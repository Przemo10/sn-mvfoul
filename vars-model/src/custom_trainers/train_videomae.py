import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from tqdm import tqdm
from src.soccernet_evaluate import evaluate
from config.label_map import OFFENCE_SEVERITY_MAP


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
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            targets_offence_severity = targets_offence_severity
            targets_action = targets_action
            mvclips = mvclips.float()

            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            for i in range(len(action)):
                if len(action) == 1:
                    dim_idx = 0
                else:
                    dim_idx = 1
                values = {
                    "Action class": INVERSE_EVENT_DICTIONARY["action_class"][
                        torch.argmax(outputs_action.detach().cpu(), dim=dim_idx)[i].item()]
                }
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=dim_idx)
                offence, severity = OFFENCE_SEVERITY_MAP[preds_sev[i].item()]
                values["Offence"] = offence
                values["Severity"] = severity

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
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