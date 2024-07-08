import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
# from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from soccernet_evaluate import evaluate
from tqdm import tqdm
import einops
from hybrid_model.mutulal_distilation_loss import MutualDistillationLoss
import logging

OFFENCE_SEVERITY_MAP = {
    0: ("No offence", ""),
    1: ("Offence", "1.0"),
    2: ("Offence", "3.0"),
    3: ("Offence", "5.0")
}

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
    md_loss = MutualDistillationLoss(temp=4.0, lambda_hyperparam=0.1)
    best_val_loss = 1000
    best_train_loss = 1000

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
            md_loss = md_loss,
            scheduler=scheduler
        )
        train_loss = loss_mutual_distillation + loss_offence_severity + loss_action

        results_single = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file[0])
        results_multi = evaluate(os.path.join(path_dataset, "train", "annotations.json"),prediction_file[1])

        print(f"TRAINING: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ,"
              f"loss_distil: {round(loss_mutual_distillation,10)}")
        print(f" Single: {results_single},\n"
              f" Multi : {results_multi}")
        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation= train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid",
            md_loss=md_loss,
            scheduler=scheduler
        )
        valid_loss = loss_mutual_distillation + loss_offence_severity + loss_action

        results_single = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file[0])
        results_multi = evaluate(os.path.join(path_dataset, "valid", "annotations.json"),prediction_file[1])

        print(f"VALID: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ,"
              f"loss_distil: {round(loss_mutual_distillation,10)}")
        print(f" Single: {results_single},\n"
              f" Multi : {results_multi}")


        if epoch % 5 == 0:
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
                scheduler=scheduler
            )
            results_single = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file[0])
            results_multi = evaluate(os.path.join(path_dataset, "test", "annotations.json"),prediction_file[1])
            print(f"TEST: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ,"
                  f"loss_distil: {round(loss_mutual_distillation, 10)}")
            print(f" Single: {results_single},\n"
                  f" Multi : {results_multi}")



        scheduler.step()

        counter += 1

        if counter > 10 and valid_loss < best_val_loss:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch + 1) + "best_valid_model.pth.tar")
            torch.save(state, path_aux)
            best_val_loss = valid_loss
        elif counter > 3 and counter % 5 == 0 and train_loss < best_train_loss:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch + 1) + "best_train_model.pth.tar")
            torch.save(state, path_aux)
            best_train_loss = train_loss
        elif counter >= (max_epochs-1):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch + 1) + "last_model.pth.tar")
            torch.save(state, path_aux)

    pbar.close()
    return


def train(dataloader, model, criterion, optimizer, epoch, model_name, train=False, set_name="train", pbar=None,
          md_loss=None, scheduler=None):
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
                    outputs_action = outputs_action[:,0]

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

                # compute the loss
                if view_type == 'single':
                    view_loss_offence_severity += criterion[0](outputs_offence_severity, targets_offence_severity)
                    view_loss_action += criterion[1](outputs_action, targets_action)
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
                mutual_distillation_offence_loss = md_loss(output['mv_collection']['offence_logits'],
                                                           single_offence_logits, targets_offence_severity[:,0])
                mutual_distillation_action_loss = md_loss(output['mv_collection']['action_logits'],
                                                          single_action_logits, targets_action[:,0])
                mutual_distillation_loss = mutual_distillation_offence_loss + mutual_distillation_action_loss

            loss = view_loss + mutual_distillation_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                optimizer.step()

            loss_total_action += float(view_loss_action)
            loss_total_offence_severity += float(view_loss_offence_severity)
            loss_total_mutual_distillation += float(mutual_distillation_loss)
            total_loss += 1

        gc.collect()
        torch.cuda.empty_cache()

    # save results to files
    with open(os.path.join(model_name, single_prediction_file), 'w') as f:
        json.dump(single_data, f, indent=4)

    with open(os.path.join(model_name, multi_view_prediction_file), 'w') as f:
        json.dump(multi_view_data, f, indent=4)
    print(loss_total_action, loss_total_offence_severity,loss_total_mutual_distillation , loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation )
    return ((os.path.join(model_name,single_prediction_file),os.path.join(model_name, multi_view_prediction_file)),
            loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation
            )