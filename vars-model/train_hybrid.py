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
        )

        results = evaluate(os.path.join(path_dataset, "train", "annotations.json"), prediction_file)
        print(f"TRAINING: loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ,"
              f"loss_distil: {round(loss_mutual_distillation,10)}")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid"
        )
        results = evaluate(os.path.join(path_dataset, "valid", "annotations.json"), prediction_file)
        print(f"VALIDATION loss_action: {round(loss_action, 6)}, loss_offence: {round(loss_offence_severity, 6)} ,"
              f"loss_distil: {round(loss_mutual_distillation, 10)}")
        print(results)

        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity, loss_mutual_distillation = train(
            test_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="test",
        )

        results = evaluate(os.path.join(path_dataset, "test", "annotations.json"), prediction_file)
        print(f"TEST: loss_action: {round(loss_action,6)}, loss_offence: {round(loss_offence_severity,6)},"
              f" loss_distil: {round(loss_mutual_distillation, 10)}")
        print(results)

        # scheduler.step()

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
    md_loss = MutualDistillationLoss(temp=10, lambda_hyperparam=0.4)

    with torch.set_grad_enabled(train):
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            view_loss_action = torch.tensor(0.0, dtype=torch.float16).cuda()
            view_loss_offence_severity = torch.tensor(0.0, dtype=torch.float16).cuda()
            view_loss = torch.tensor(0.0, dtype=torch.float16).cuda()
            mutual_distillation_loss = torch.tensor(0.0, dtype=torch.float16).cuda()

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            #criterion[0] = criterion[0].cpu()
            # criterion[1] = criterion[1].cpu()
            # print(mvclips.shape), dtype=torch.float16

            if pbar is not None:
                pbar.update()


            output = model(mvclips)
            # print(output)
            # print(mvclips.shape)
            batch_size, total_views, _, _, _,_ = mvclips.shape

            for view_type in output:
                outputs_offence_severity = output[view_type]['offence_logits']
                outputs_action = output[view_type]['action_logits']
                if view_type == 'single':
                    # print(1.0, outputs_offence_severity.shape, outputs_action.shape)
                    outputs_offence_severity = einops.rearrange(
                        outputs_offence_severity,
                        pattern='(b n) k -> b n k', b=batch_size, n=total_views)
                    outputs_action = einops.rearrange(
                        outputs_action,
                        pattern='(b n) k -> b n k', b=batch_size, n=total_views)

                    # print(2.0, outputs_offence_severity.shape, outputs_action.shape)
                    outputs_offence_severity = outputs_offence_severity.max(dim=1)[0]
                    outputs_action = outputs_action.max(dim=1)[0]
                    # print(3.0, outputs_offence_severity.shape, outputs_action.shape)

                if len(action) == 1:
                    values = {}
                    preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
                    preds_act = torch.argmax(outputs_action.detach().cpu(), dim=1)
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
                    preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
                    preds_act = torch.argmax(outputs_action.detach().cpu(), dim=1)
                    # print(f'{view_type} preds_sev shape: {preds_sev.shape}, {preds_sev}')
                    # print(f'{view_type} preds_act shape: {preds_sev}, {preds_sev}')
                    # print(action)

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
                if len(output['mv_collection']['offence_logits'].size()) == 1:
                    outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
                if len(output['mv_collection']['action_logits'].size()) == 1:
                    outputs_action = outputs_action.unsqueeze(0)

                # compute the loss

                # print(view_type, outputs_offence_severity.shape, targets_offence_severity.shape)
                # print(view_type, outputs_action.shape, targets_action.shape)
                # print( outputs_offence_severity.get_device(), targets_offence_severity.get_device())
                if view_type == 'single':
                    view_loss_offence_severity += torch.tensor(0.1).cuda() * criterion[0](outputs_offence_severity, targets_offence_severity)
                    view_loss_action += torch.tensor(0.3).cuda() * criterion[1](outputs_action, targets_action)
                    view_loss = view_loss + view_loss_action + view_loss_offence_severity

                if view_type == 'mv_collection':
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
                    output['single']['action_logits'],
                    pattern='(b n) k -> b n k',
                    b=batch_size,
                    n=total_views
                )
                mutual_distillation_offence_loss = md_loss(
                    output['mv_collection']['offence_logits'],
                    single_offence_logits,
                    targets_offence_severity.max(dim=1)[0]  # alternative targets_offence_severity[:, 0]
                )
                mutual_distillation_action_loss = md_loss(
                    output['mv_collection']['action_logits'],
                    single_action_logits,
                    targets_action.max(dim=1)[0]
                )
                mutual_distillation_loss = mutual_distillation_offence_loss + mutual_distillation_action_loss
            loss = view_loss + mutual_distillation_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 80.0)
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
    print(loss_total_action, loss_total_offence_severity,loss_total_mutual_distillation , loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation )
    return (os.path.join(model_name,prediction_file),
            loss_total_action / total_loss,
            loss_total_offence_severity / total_loss,
            loss_total_mutual_distillation
            )

