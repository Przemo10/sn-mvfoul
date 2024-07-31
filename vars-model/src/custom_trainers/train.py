import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from tqdm import tqdm
from src.soccernet_evaluate import evaluate
from sklearn.metrics import confusion_matrix, classification_report


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
            max_epochs=1000
            ):
    logging.info("start training")
    counter = 0
    writer = SummaryWriter()

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

        results = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file)
        print(f"TRAINING loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(results)
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

        results = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file)
        print(f"VALIDATION: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(results)

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

        results = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file)
        print(f"TEST: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ")
        print(results)

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch + 1) + "_model.pth.tar")
            torch.save(state, path_aux)

    writer.flush()
    pbar.close()
    return


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
                    epoch
                )

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile:
        json.dump(data, outfile)
    return os.path.join(model_name,
                        prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss


def sklearn_evaluation(dataloader,
                       model,
                       model_name="",
                       set_name="train",
                       ):
    prediction_file = "sklearn_summary_" + model_name + "_" + set_name + ".json"

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
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)
                # add elements into list
                targets_offence_severity_list.extend(targets_offence_severity_int_or_list)
                targets_action_list.extend(targets_action_int_or_list)
                pred_offence_list.append(preds_sev.detach().cpu().numpy().tolist())
                pred_action_list.append(preds_act.cpu().numpy().tolist())
            else:
                preds_sev = torch.argmax(outputs_offence_severity, 1)
                preds_act = torch.argmax(outputs_action, 1)
                targets_offence_severity_list.extend(targets_offence_severity_int_or_list)
                targets_action_list.extend(targets_action_int_or_list)
                pred_offence_list.extend(preds_sev.detach().cpu().numpy().tolist())
                pred_action_list.extend(preds_act.cpu().numpy().tolist())
        gc.collect()
        torch.cuda.empty_cache()

    targets_offence_severity_map_list = [offence_labels[idx] for idx in targets_offence_severity_list]
    preds_offence_map_list = [offence_labels[idx] for idx in pred_offence_list]
    targets_action_map_list = [action_labels[idx] for idx in targets_action_list]
    preds_action_map_list = [action_labels[idx] for idx in pred_action_list]
    cm_offence = confusion_matrix(
        targets_offence_severity_map_list, preds_offence_map_list, labels=offence_labels
    ).tolist()

    print(targets_action_list)
    print("**********")
    print(pred_action_list)

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


# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
               model,
               set_name="test",
               ):
    model.eval()

    prediction_file = f"predicitions_{set_name}.json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            # mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

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

        gc.collect()
        torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)
    return prediction_file