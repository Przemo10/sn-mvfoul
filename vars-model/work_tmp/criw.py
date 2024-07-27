from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import gc
import os
import logging


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class MutualDistillationLoss(nn.Module):
    def __init__(self, temp=4.0, lambda_hyperparam=0.1):
        super(MutualDistillationLoss, self).__init__()
        self.temp = temp
        self.lambda_hyperparam = lambda_hyperparam

    def forward(self, logits_a, logits_b, targets):
        probs_a = F.softmax(logits_a / self.temp, dim=1)
        probs_b = F.softmax(logits_b / self.temp, dim=1)
        distillation_loss = F.kl_div(probs_a.log(), probs_b, reduction='batchmean') + \
                            F.kl_div(probs_b.log(), probs_a, reduction='batchmean')
        distillation_loss = distillation_loss * (self.temp ** 2) * self.lambda_hyperparam
        ce_loss = F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_b, targets)
        return ce_loss + distillation_loss


def trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, best_model_path,
            epoch_start, model_name, path_dataset, max_epochs=1000):
    logging.info("start training")
    early_stopping_patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epoch_start, max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        prediction_file, train_loss_action, train_loss_offence_severity, train_loss_mutual_distillation = train(
            train_loader, model, criterion, optimizer, scheduler, epoch + 1, model_name, train=True, set_name="train",
            pbar=pbar)

        prediction_file, val_loss_action, val_loss_offence_severity, val_loss_mutual_distillation = train(
            val_loader2, model, criterion, optimizer, scheduler, epoch + 1, model_name, train=False, set_name="valid")

        prediction_file, test_loss_action, test_loss_offence_severity, test_loss_mutual_distillation = train(
            test_loader2, model, criterion, optimizer, scheduler, epoch + 1, model_name, train=False, set_name="test")

        print(
            f"TRAINING: a_loss: {train_loss_action}, offence_loss: {train_loss_offence_severity}, dist loss : {train_loss_mutual_distillation}")
        print(
            f"VALIDATION: a_loss: {val_loss_action}, offence_loss: {val_loss_offence_severity}, dist loss : {val_loss_mutual_distillation}")
        print(
            f"TEST: a_loss: {test_loss_action}, offence_loss: {test_loss_offence_severity}, dist loss : {test_loss_mutual_distillation}")

        val_loss = val_loss_action + val_loss_offence_severity + val_loss_mutual_distillation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    pbar.close()
    return


def train(dataloader, model, criterion, optimizer, scheduler, epoch, model_name, train=False, set_name="train",
          pbar=None):
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0.0
    loss_total_offence_severity = 0.0
    total_loss = 0.0
    loss_total_mutual_distillation = 0.0

    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    prediction_file = f"predictions_{set_name}_epoch_{epoch}.json"
    actions = {}
    md_loss = MutualDistillationLoss(temp=4.0, lambda_hyperparam=0.1)

    with torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            view_loss_action = torch.tensor(0.0, dtype=torch.float32).cuda()
            view_loss_offence_severity = torch.tensor(0.0, dtype=torch.float32).cuda()
            view_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            mutual_distillation_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()

            if pbar is not None:
                pbar.update()

            output = model(mvclips)
            batch_size, total_views, _, _, _, _ = mvclips.shape

            for view_type in output:
                outputs_offence_severity = output[view_type]['offence_logits']
                outputs_action = output[view_type]['action_logits']
                if view_type == 'single':
                    outputs_offence_severity = einops.rearrange(outputs_offence_severity, '(b n) k -> b n k',
                                                                b=batch_size, n=total_views)
                    outputs_action = einops.rearrange(outputs_action, '(b n) k -> b n k', b=batch_size, n=total_views)
                    outputs_offence_severity = outputs_offence_severity.max(dim=1)[0]
                    outputs_action = outputs_action.max(dim=1)[0]

                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), dim=1)

                if len(action) == 1:
                    values = {"Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]}
                    values["Offence"] = "Offence" if preds_sev.item() != 0 else "No offence"
                    values["Severity"] = {1: "1.0", 2: "3.0", 3: "5.0"}.get(preds_sev.item(), "")
                    actions[action[0]] = values
                else:
                    for i in range(len(action)):
                        values = {"Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]}
                        values["Offence"] = "Offence" if preds_sev[i].item() != 0 else "No offence"
                        values["Severity"] = {1: "1.0", 2: "3.0", 3: "5.0"}.get(preds_sev[i].item(), "")
                        actions[action[i]] = values

                if len(output['mv_collection']['offence_logits'].size()) == 1:
                    outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
                if len(output['mv_collection']['action_logits'].size()) == 1:
                    outputs_action = outputs_action.unsqueeze(0)

                if view_type == 'single':
                    view_loss_offence_severity += 0.5 * criterion[0](outputs_offence_severity, targets_offence_severity)
                    view_loss_action += 0.5 * criterion[1](outputs_action, targets_action)
                    view_loss += view_loss_action + view_loss_offence_severity

                if view_type == 'mv_collection':
                    view_loss_offence_severity += criterion[0](outputs_offence_severity, targets_offence_severity)
                    view_loss_action += criterion[1](outputs_action, targets_action)
                    view_loss += view_loss_action + view_loss_offence_severity

            if md_loss is not None:
                single_offence_logits = einops.rearrange(output['single']['offence_logits'], '(b n) k -> b n k',
                                                         b=batch_size, n=total_views)
                single_action_logits = einops.rearrange(output['single']['action_logits'], '(b n) k -> b n k',
                                                        b=batch_size, n=total_views)
                mutual_distillation_offence_loss = md_loss(output['mv_collection']['offence_logits'],
                                                           single_offence_logits,
                                                           targets_offence_severity.max(dim=1)[0])
                mutual_distillation_action_loss = md_loss(output['mv_collection']['action_logits'],
                                                          single_action_logits, targets_action.max(dim=1)[0])
                mutual_distillation_loss = mutual_distillation_offence_loss + mutual_distillation_action_loss

            loss = view_loss + mutual_distillation_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 80.0)
                optimizer.step()
                scheduler.step()

            loss_total_action += float(view_loss_action)
            loss_total_offence_severity += float(view_loss_offence_severity)
            loss_total_mutual_distillation += float(mutual_distillation_loss)
            total_loss += 1

        gc.collect()
        torch.cuda.empty_cache()

    return prediction_file, loss_total_action / total_loss, loss_total_offence_severity / total_loss, loss_total_mutual_distillation / total_loss
