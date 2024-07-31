import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from src.soccernet_evaluate import evaluate
from tqdm import tqdm
import einops
from src.custom_loss.mutulal_distilation_loss import MutualDistillationLoss
import logging
from config.label_map import OFFENCE_SEVERITY_MAP
from sklearn.metrics import confusion_matrix, classification_report

def save_checkpoint(epoch, model, optimizer, scheduler, path, filename, losses, results):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'losses': losses,
        'results': results
    }
    torch.save(state, os.path.join(path, filename))


def trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, best_model_path,
            epoch_start, model_name, path_dataset, max_epochs=1000):
    logging.info("start training")
    md_loss = MutualDistillationLoss(temp=1.0, lambda_hyperparam=10)
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    all_losses = {'train': [], 'valid': [], 'test': []}
    all_results = {'train_single': [], 'train_multi': [], 'valid_single': [], 'valid_multi': [], 'test_single': [],
                   'test_multi': []}
    writer = SummaryWriter()

    for epoch in range(epoch_start, max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
            md_loss=md_loss,
            scheduler=scheduler,
            writer=writer,
        )
        train_loss = loss_mutual_distillation + loss_offence_severity + loss_action

        results_single = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file[0])
        results_multi = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file[1])

        all_losses['train'].append(train_loss)
        all_results['train_single'].append(results_single)
        all_results['train_multi'].append(results_multi)

        print(
            f"TRAINING: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)},"
            f" loss_distil: {round(loss_mutual_distillation, 10)}")
        print(f" Single: {results_single},\n Multi : {results_multi}")

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid",
            md_loss=md_loss,
            scheduler=scheduler,
            writer=writer,
        )
        valid_loss = loss_mutual_distillation + loss_offence_severity + loss_action

        results_single = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file[0])
        results_multi = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file[1])

        all_losses['valid'].append(valid_loss)
        all_results['valid_single'].append(results_single)
        all_results['valid_multi'].append(results_multi)

        print(
            f"VALID: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)}, "
            f"loss_distil: {round(loss_mutual_distillation, 10)}")
        print(f" Single: {results_single},\n Multi : {results_multi}")

        if epoch % 2 == 0:
            prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
                md_loss=md_loss,
                scheduler=scheduler,
                writer=writer,
            )
            results_single = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file[0])
            results_multi = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file[1])

            all_losses['test'].append(
                (loss_action, loss_offence_severity, loss_mutual_distillation))
            all_results['test_single'].append(results_single)
            all_results['test_multi'].append(results_multi)

            print(
                f"TEST: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)}, "
                f"loss_distil: {round(loss_mutual_distillation, 10)}")
            print(f" Single: {results_single},\n Multi : {results_multi}")

        scheduler.step()

        if valid_loss < best_val_loss and epoch > 5:
            save_checkpoint(epoch, model, optimizer, scheduler, best_model_path, "best_valid_model.pth.tar", all_losses,
                            all_results)
            best_val_loss = valid_loss
            print('Best valid model saved.')
        if train_loss < best_train_loss:
            save_checkpoint(epoch, model, optimizer, scheduler, best_model_path, "best_train_model.pth.tar", all_losses,
                            all_results)
            best_train_loss = train_loss
            print('Best train model saved.')
        if epoch >= max_epochs - 3:
            save_checkpoint(epoch, model, optimizer, scheduler, best_model_path, "last_model.pth.tar", all_losses,
                            all_results)
            print('Last model saved.')
        if epoch % 5 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_model_path, f"epoch{epoch+1}_model.pth.tar", all_losses,
                            all_results)
            print('saved epoch model ')

        writer.flush()
        pbar.close()
    return

def train(dataloader, model, criterion, optimizer, epoch, model_name, train=False, set_name="train", pbar=None,
          md_loss=None, scheduler=None, writer=None,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # paths where we will save the results
    single_prediction_file = f"single_predictions_{set_name}_epoch_{epoch}.json"
    multi_view_prediction_file = f"multi_view_predictions_{set_name}_epoch_{epoch}.json"

    single_data = {"Set": set_name, "Actions": {}}
    multi_view_data = {"Set": set_name, "Actions": {}}

    with torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            view_loss_action = torch.tensor(0.0, device=device)
            view_loss_offence_severity = torch.tensor(0.0, device=device)
            view_loss = torch.tensor(0.0, device=device)
            mutual_distillation_loss = torch.tensor(0.0, device=device)

            targets_offence_severity = targets_offence_severity.to(device)
            targets_action = targets_action.to(device)
            mvclips = mvclips.to(device).float()

            if pbar is not None:
                pbar.update()

            output = model(mvclips)
            batch_size, total_views, _, _, _, _ = mvclips.shape

            for view_type in output:
                outputs_offence_severity = output[view_type]['offence_logits']
                outputs_action = output[view_type]['action_logits']

                if view_type == 'single':
                    outputs_offence_severity = einops.rearrange(outputs_offence_severity, ' (b n) k -> b n k',
                                                                b=batch_size, n=total_views)
                    outputs_action = einops.rearrange(outputs_action, ' (b n) k -> b n k', b=batch_size, n=total_views)
                    outputs_offence_severity = outputs_offence_severity[:,0]
                    outputs_action =  outputs_action[:,0]

                for i in range(len(action)):
                    values = {
                        "Action class": INVERSE_EVENT_DICTIONARY["action_class"][
                            torch.argmax(outputs_action.detach().cpu(), dim=1)[i].item()]
                    }
                    preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
                    offence, severity = OFFENCE_SEVERITY_MAP[preds_sev[i].item()]
                    values["Offence"] = offence
                    values["Severity"] = severity

                    if view_type == 'single':
                        single_data["Actions"][action[i]] = values
                    else:
                        multi_view_data["Actions"][action[i]] = values

                if len(output['mv_collection']['offence_logits'].size()) == 1:
                    outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
                if len(output['mv_collection']['action_logits'].size()) == 1:
                    outputs_action = outputs_action.unsqueeze(0)
                # compute the custom_loss
                if view_type == 'single':
                    view_loss_offence_severity += 0.5 * criterion[0](outputs_offence_severity, targets_offence_severity)
                    # print(outputs_action, targets_action)
                    view_loss_action += 0.5 * criterion[1](outputs_action, targets_action)
                    # print(view_loss_action)
                    view_loss = view_loss + view_loss_action + view_loss_offence_severity

                if view_type == 'mv_collection':
                    view_loss_offence_severity += criterion[0](outputs_offence_severity, targets_offence_severity)
                    view_loss_action += criterion[1](outputs_action, targets_action)
                    view_loss = view_loss + view_loss_action + view_loss_offence_severity

            if True:
                single_offence_logits = einops.rearrange(output['single']['offence_logits'], '(b n) k -> b n k',
                                                         b=batch_size, n=total_views)
                single_action_logits = einops.rearrange(output['single']['action_logits'], '(b n) k -> b n k',
                                                        b=batch_size, n=total_views)
                mutual_distillation_offence_loss = md_loss(
                    output['mv_collection']['offence_logits'],
                    single_offence_logits,targets_offence_severity[:,0]
                )
                mutual_distillation_action_loss = md_loss(
                    output['mv_collection']['action_logits'],
                    single_action_logits,
                    targets_action[:,0]
                )
                mutual_distillation_loss += mutual_distillation_offence_loss  + mutual_distillation_action_loss

            loss = view_loss + mutual_distillation_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # scheduler.step()

            loss_total_action += float(view_loss_action)
            loss_total_offence_severity += float(view_loss_offence_severity)
            loss_total_mutual_distillation += float(mutual_distillation_loss)
            total_loss += 1
            if writer is not None:
                writer.add_scalars(
                    f"Loss/{set_name}",
                    {
                        f"action - {model_name}": loss_total_action,
                        f"offence - {model_name}": loss_total_offence_severity,
                        f"total - {model_name}": loss_total_offence_severity + loss_total_action
                    },
                    epoch
                )

        gc.collect()
        torch.cuda.empty_cache()

    # save results to files
    with open(os.path.join(model_name, single_prediction_file), 'w') as f:
        json.dump(single_data, f, indent=4)

    with open(os.path.join(model_name, multi_view_prediction_file), 'w') as f:
        json.dump(multi_view_data, f, indent=4)
    print(loss_total_action, loss_total_offence_severity,loss_total_mutual_distillation , loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation /total_loss )
    return ((os.path.join(model_name,single_prediction_file),os.path.join(model_name, multi_view_prediction_file)),
            loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation/ total_loss,
            )


def sklearn_evaluation(dataloader,
          model,
          model_name="",
          set_name="train",
        ):

    prediction_file = "sklearn_summary_multi_view_hybird_" +model_name + "_" + set_name + ".json"

    model.eval()
    offence_labels = ["No offence", "Offence-no card", "Offence yellow", "Offence red"]
    action_labels = [INVERSE_EVENT_DICTIONARY["action_class"][i] for i in range(0,8)]
    targets_offence_severity_list = []
    targets_action_list = []
    pred_offence_list = []
    pred_action_list = []
    data = {}
    data["Set"] = set_name
    data["model_name"] =model_name

    with torch.no_grad():
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            targets_offence_severity_int_or_list = torch.argmax(targets_offence_severity.cpu(),1).numpy().tolist()
            targets_action_int_or_list = torch.argmax(targets_action.cpu(), 1).numpy().tolist()
            mvclips = mvclips.cuda().float()
            output = model(mvclips)
            outputs_offence_severity = output['mv_collection']['offence_logits']
            outputs_action = output['mv_collection']['action_logits']
            preds_sev = torch.argmax(outputs_offence_severity, 1)
            preds_act = torch.argmax(outputs_action, 1)
            targets_offence_severity_list.extend(targets_offence_severity_int_or_list)
            targets_action_list.extend(targets_action_int_or_list)
            pred_offence_list.extend(preds_sev.detach().cpu().numpy().tolist())
            pred_action_list.extend(preds_act.cpu().numpy().tolist())
            print(preds_sev.detach().cpu().numpy().tolist(), preds_act.cpu().numpy().tolist())
        gc.collect()
        torch.cuda.empty_cache()
    targets_offence_severity_map_list = [offence_labels[idx] for idx in targets_offence_severity_list]
    preds_offence_map_list = [offence_labels[idx] for idx in pred_offence_list]
    targets_action_map_list = [action_labels[idx] for idx in targets_action_list]
    preds_action_map_list = [action_labels[idx] for idx in pred_action_list]
    cm_offence = confusion_matrix(
        targets_offence_severity_map_list, preds_offence_map_list, labels=offence_labels
    ).tolist()

    cm_action = confusion_matrix(
        targets_action_map_list, preds_action_map_list, labels=action_labels
    ).tolist()
    data['cm_offence'] = cm_offence
    data['cm_actions'] = cm_action
    print(cm_offence)
    print(cm_action)

    cr_offence = classification_report(
        targets_offence_severity_list, pred_offence_list, target_names=offence_labels,output_dict=True
    )
    cr_action = classification_report(
        targets_action_list, pred_action_list, target_names=action_labels, output_dict=True)
    data['cr_offence'] = cr_offence
    data['cr_action'] = cr_action
    print(cr_offence)
    print(cr_action)
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)


def evaluation(dataloader,
               model,
               set_name="test",
               ):
    model.eval()

    multi_view_prediction_file = f"multi_view_predictions_{set_name}.json"
    multi_view_data = {"Set": set_name, "Actions": {}}

    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            # mvclips = mvclips.float()
            output = model(mvclips)
            multi_view_offence_output = output['mv_collection']['offence_logits']
            multi_view_action_output = output['mv_collection']['action_logits']

            for i in range(len(action)):
                values = {
                    "Action class": INVERSE_EVENT_DICTIONARY["action_class"][
                        torch.argmax(multi_view_action_output.detach().cpu(), dim=1)[i].item()]
                }
                preds_sev = torch.argmax(multi_view_offence_output.detach().cpu(), dim=1)
                offence, severity = OFFENCE_SEVERITY_MAP[preds_sev[i].item()]
                values["Offence"] = offence
                values["Severity"] = severity
                multi_view_data["Actions"][action[i]] = values
        gc.collect()
        torch.cuda.empty_cache()

    with open(multi_view_prediction_file, "w") as outfile:
        json.dump(multi_view_data, outfile)
    return multi_view_prediction_file