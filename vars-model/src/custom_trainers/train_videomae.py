import logging
import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from tqdm import tqdm
from src.soccernet_evaluate import evaluate
from config.label_map import OFFENCE_SEVERITY_MAP
from src.custom_trainers.training_history import get_best_n_metric_result, find_highest_leaderboard_index
from src.custom_trainers.training_history import update_epoch_results_dict, TRAINING_RESULT_DICT
import  numpy as np
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


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          writer=None,
          ):
    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name)

        # path where we will save the results
    prediction_file = f"predicitions_{set_name}_epoch_{epoch}.json"

    data = {}
    data["Set"] = set_name

    actions = {}

    with torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()

            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:

                preds_sev = torch.argmax(outputs_offence_severity, 0)  # dla video-mae
                preds_act = torch.argmax(outputs_action, 0)
                # preds_sev = torch.argmax(outputs_offence_severity, 0)
                # preds_act = torch.argmax(outputs_action,0)  # dla mvit-

                values = {}
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
                actions[action[0]] = values
            else:

                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values

            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)

                # compute the custom_loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
            if writer is not None:
                writer.add_scalars(
                    f"Loss/{set_name}",
                    {
                        f"action - {model_name}": loss_total_action,
                        f"offence - {model_name}": loss_total_offence_severity,
                        f"total - {model_name}": loss_total_offence_severity + loss_total_action
                    },
                    epoch + 1
                )

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile:
        json.dump(data, outfile)
    return os.path.join(model_name,
                        prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss


# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
               model,
               set_name="test",
               ):
    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            # mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            for i in range(len(action)):
                if len(action) == 1:
                    pred_idx = 0
                values = {
                    "Action class": INVERSE_EVENT_DICTIONARY["action_class"][
                        torch.argmax(outputs_action.detach().cpu(), dim=1)[pred_idx].item()]
                }
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
                offence, severity = OFFENCE_SEVERITY_MAP[preds_sev[i].item()]
                values["Offence"] = offence
                values["Severity"] = severity

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)
    return prediction_file