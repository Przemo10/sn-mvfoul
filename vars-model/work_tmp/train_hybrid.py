import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
import einops
from hybrid_model.mutulal_distilation_loss import MutualDistillationLoss


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          ):
    # switch to train mode
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

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = dict()
    data["Set"] = set_name

    actions = {}
    md_loss = MutualDistillationLoss()

    if True:
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            view_loss_action = torch.tensor(0.0)
            view_loss_offence_severity = torch.tensor(0.0)
            view_loss = torch.tensor(0.0)
            mutual_distillation_loss = torch.tensor(0.0)

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()

            if pbar is not None:
                pbar.update()

            output = model(mvclips)
            print(mvclips.shape)
            batch_size, total_views, _, _, _ = mvclips.shape

            for view_type in output:
                outputs_offence_severity = output[view_type]['offence_logits']
                outputs_action = output[view_type]['outputs_action']
                if view_type == 'single':
                    outputs_offence_severity = outputs_offence_severity.mean(dim=1)
                    outputs_action - outputs_action.mean(dim=1)
                values = {}
                actions[view_type] = {}
                if len(action) == 1:
                    preds_sev = torch.argmax(outputs_offence_severity, dim=0)
                    preds_act = torch.argmax(outputs_action, dim=0)
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                    if preds_sev.item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev.item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev.item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev.item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[view_type][action[0]] = values
                else:
                    preds_sev = torch.argmax(outputs_offence_severity, dim=1)
                    preds_act = torch.argmax(outputs_action, dim=1)

                    for i in range(len(action)):
                        values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                        if preds_sev.item() == 0:
                            values["Offence"] = "No offence"
                            values["Severity"] = ""
                        elif preds_sev.item() == 1:
                            values["Offence"] = "Offence"
                            values["Severity"] = "1.0"
                        elif preds_sev.item() == 2:
                            values["Offence"] = "Offence"
                            values["Severity"] = "3.0"
                        elif preds_sev.item() == 3:
                            values["Offence"] = "Offence"
                            values["Severity"] = "5.0"

                        actions[view_type][action[i]] = values
                if len(output['mv_collection']['offence_logits'].size()) == 1:
                    outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
                if len(output['mv_collection']['action_logits'].size()) == 1:
                    outputs_action = outputs_action.unsqueeze(0)

                # compute the loss

                view_loss_offence_severity += criterion[0](outputs_offence_severity, targets_offence_severity)
                view_loss_action += criterion[1](outputs_action, targets_action)
                view_loss = view_loss + view_loss_action + view_loss_offence_severity
            if md_loss is not None:
                single_offence_logits = einops.rearrange(
                    output['single']['offence_logits'],
                    pattern='(b n) k -> b n k',
                    b=batch_size,
                    n=total_views
                )
                single_action_logits = einops.rearrange(
                    output['single']['offence_logits'],
                    pattern='(b n) k -> b n k',
                    b=batch_size,
                    n=total_views
                )
                mutual_distillation_offence_loss = md_loss(
                    output['mv_collection']['offence_logits'],
                    single_offence_logits,
                    targets_offence_severity.mean(dim=1)  # alternative targets_offence_severity[:, 0]
                )
                mutual_distillation_action_loss = md_loss(
                    output['mv_collection']['action_logits'],
                    single_action_logits,
                    targets_action.mean(dim=1)
                )
                mutual_distillation_loss = mutual_distillation_offence_loss + mutual_distillation_action_loss
            loss = view_loss + mutual_distillation_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(view_loss_action)
            loss_total_offence_severity += float(view_loss_offence_severity)
            loss_total_mutual_distillation += float(mutual_distillation_loss)
            total_loss += 1

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile:
        json.dump(data, outfile)
    return os.path.join(model_name,
                        prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss
