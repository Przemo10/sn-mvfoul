import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from src.soccernet_evaluate import evaluate
from tqdm import tqdm
import logging
from config.label_map import OFFENCE_SEVERITY_MAP
from sklearn.metrics import confusion_matrix, classification_report
from src.custom_trainers.training_history import get_best_n_metric_result, find_highest_leaderboard_index
from src.custom_trainers.training_history import update_epoch_results_dict, TRAINING_RESULT_DICT
import numpy as np
from src.custom_trainers.training_history import save_training_history, get_leaderboard_summary


def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000,
            patience=10,
            writer=None,
            ):
    logging.info("start training")
    counter = 0
    patience_counter = patience

    for epoch in range(epoch_start, max_epochs):

        print(f"Epoch {epoch + 1}/{max_epochs}")

        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
            writer=writer,
        )

        train_results = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file)
        print(f"TRAINING loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(train_results)
        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid",
            writer=writer,
        )

        valid_results = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file)
        print(f"VALIDATION: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(valid_results)
        valid_epoch_leaderboard = valid_results["leaderboard_value"]

        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
            test_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="test",
            writer=writer,
        )

        test_results = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file)
        print(f"TEST: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(test_results)
        test_epoch_leaderboard = test_results["leaderboard_value"]

        scheduler.step()

        if writer is not None:
            writer.add_scalars(
                f"Metric/train",
                train_results,
                epoch + 1
            )
            writer.add_scalars(
                f"Metric/valid",
                valid_results,
                epoch + 1
            )
            writer.add_scalars(
                f"Metric/test",
                test_results,
                epoch + 1
            )

        update_epoch_results_dict("train", train_results)
        update_epoch_results_dict("valid", valid_results)
        update_epoch_results_dict("test", test_results)

        counter += 1
        if counter > 5:
            saved_cond = np.logical_or(
                get_best_n_metric_result("valid") < valid_epoch_leaderboard,
                get_best_n_metric_result("test") < test_epoch_leaderboard
            )
            saved_cond = np.logical_or(
                saved_cond,
                epoch + 3 >= max_epochs
            )
            if saved_cond:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "history": TRAINING_RESULT_DICT
                }
                path_aux = os.path.join(best_model_path, str(epoch + 1) + "_model.pth.tar")
                torch.save(state, path_aux)
                save_training_history(best_model_path)
            # Early stopping
            if valid_epoch_leaderboard > np.max(TRAINING_RESULT_DICT['valid']['leaderboard_value'][:-1]):
                patience = patience_counter
            else:
                patience -= 1
                if patience == 0:
                    break

    writer.flush()
    pbar.close()

    # Finding the highest leaderboard value index for 'valid' and 'test' sets
    highest_valid_index = find_highest_leaderboard_index(TRAINING_RESULT_DICT, 'valid')
    highest_test_index = find_highest_leaderboard_index(TRAINING_RESULT_DICT, 'test')

    print(f"Highest leaderboard value index for valid set: {highest_valid_index}")
    print(f"Highest leaderboard value index for test set: {highest_test_index}")
    print(f"Training result dict: {TRAINING_RESULT_DICT}")

    leaderboard_summary = get_leaderboard_summary(highest_valid_index, highest_test_index)
    print(leaderboard_summary)

    return leaderboard_summary


def train(dataloader, model, criterion, optimizer, epoch, model_name, train=False, set_name="train", pbar=None,
          md_loss=None, scheduler=None, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0.0
    loss_total_offence_severity = 0.0
    total_loss = 0.0

    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    # paths where we will save the results
    single_prediction_file = f"single_predictions_{set_name}_epoch_{epoch}.json"
    multi_view_prediction_file = f"multi_view_predictions_{set_name}_epoch_{epoch}.json"

    # single_data = {"Set": set_name, "Actions": {}}
    multi_view_data = {"Set": set_name, "Actions": {}}

    with torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            single_view_loss_action = torch.tensor(0.0, device=device)
            single_view_loss_offence_severity = torch.tensor(0.0, device=device)

            targets_offence_severity = targets_offence_severity.to(device)
            targets_action = targets_action.to(device)
            mvclips = mvclips.to(device).float()

            if pbar is not None:
                pbar.update()

            output = model(mvclips)
            batch_size, total_views, _, _, _, _ = mvclips.shape

            # single_view_transformation
            for view_id in range(total_views):
                single_label_id = f"single_{view_id}"
                # single_view output
                single_view_offence_output = output[single_label_id]["offence_logits"]
                single_view_action_output = output[single_label_id]['action_logits']

                if len(single_view_offence_output.size()) == 1:
                    single_view_offence_output = single_view_offence_output.unsqueeze(0)
                    single_view_action_output = single_view_action_output.unsqueeze(0)

                # compute total_single_view_loss
                single_view_loss_offence_severity += criterion[0](single_view_offence_output, targets_offence_severity)
                single_view_loss_action += criterion[1](single_view_action_output, targets_action)

            if 'mv_collection' in list(output.keys()):
                multi_view_offence_output = output['mv_collection']['offence_logits']
                multi_view_action_output = output['mv_collection']['action_logits']

                # evaluation result ...
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

                if len(multi_view_offence_output.size()) == 1:
                    multi_view_offence_output = multi_view_offence_output.unsqueeze(0)
                    multi_view_action_output = multi_view_action_output.unsqueeze(0)

                # multi-view-loss
                multi_view_loss_offence_severity = criterion[0](multi_view_offence_output, targets_offence_severity)
                multi_view_view_loss_action = criterion[1](multi_view_action_output, targets_action)
                multi_view_loss = multi_view_loss_offence_severity + multi_view_view_loss_action

            loss = single_view_loss_offence_severity + single_view_loss_offence_severity + multi_view_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

            loss_total_action += float(multi_view_view_loss_action + single_view_loss_action)
            loss_total_offence_severity += float(multi_view_loss_offence_severity + single_view_loss_offence_severity)
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

    with open(os.path.join(model_name, multi_view_prediction_file), 'w') as f:
        json.dump(multi_view_data, f, indent=4)
    return (os.path.join(model_name, multi_view_prediction_file),
            loss_total_action / total_loss,
            loss_total_offence_severity / total_loss
            )


def sklearn_evaluation(dataloader,
                       model,
                       model_name="",
                       set_name="train",
                       ):
    prediction_file = "sklearn_summary_multi_view_hybird_" + model_name + "_" + set_name + ".json"

    model.eval()
    offence_labels = ["No offence", "Offence-no card", "Offence yellow", "Offence red"]
    action_labels = [INVERSE_EVENT_DICTIONARY["action_class"][i] for i in range(0, 8)]
    targets_offence_severity_list = []
    targets_action_list = []
    pred_offence_list = []
    pred_action_list = []
    data = {}
    data["Set"] = set_name
    data["model_name"] = model_name

    with torch.no_grad():
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
            targets_offence_severity_int_or_list = torch.argmax(targets_offence_severity.cpu(), 1).numpy().tolist()
            targets_action_int_or_list = torch.argmax(targets_action.cpu(), 1).numpy().tolist()
            mvclips = mvclips.cuda().float()
            output = model(mvclips)
            outputs_offence_severity = output['mv_collection']['offence_logits']
            outputs_action = output['mv_collection']['action_logits']
            targets_offence_severity_int_or_list = torch.argmax(targets_offence_severity.cpu(), 1).numpy().tolist()
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
        targets_offence_severity_list, pred_offence_list, target_names=offence_labels, output_dict=True
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