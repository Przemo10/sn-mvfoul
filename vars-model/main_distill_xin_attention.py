import os
import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from src.custom_loss.custom_step_lr_scheduler import  CustomStepLRScheduler
from torch.utils.tensorboard import SummaryWriter
from src.custom_dataset.hybrid_dataset import MultiViewDatasetHybrid
from src.custom_trainers.train_xin_distill import trainer, evaluation, sklearn_evaluation
from src.custom_loss.loss_selector import select_training_loss
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
from src.custom_model.model_selector import XIN_NET_VERSION
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
from torchvision.models.video import MViT_V2_S_Weights
from torchvision.models.video import swin3d_s,  Swin3D_S_Weights, swin3d_t, Swin3D_T_Weights
from datetime import datetime


def checkArguments():
    # args.num_views
    if args.num_views > 5 or args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.weighted_loss
    if args.weighted_loss not in ["Base", "No", "Exp", "Yes", "Focal", "FocalCE", "BaseExp", "WeightedFocal"]:
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Base, No, Exp, Yes, Focal, FocalCE")
        exit()

    # args.start_frame
    if args.start_frame > 124 or args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()


def main(*args):
    if args:
        args = args[0]
        LR = args.LR
        gamma = args.gamma
        step_size = args.step_size
        start_frame = args.start_frame
        end_frame = args.end_frame
        weight_decay = args.weight_decay
        video_shift_aug = args.video_shift_aug
        model_name = f"{args.model_name}_s{args.net_version_s}_t{args.net_version_t}_kd_{args.kd_temp}_{args.kd_lambda}"
        net_version_s = args.net_version_s
        net_version_t = args.net_version_t
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / (
                    (args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        pre_model_s = args.pre_model_s
        pre_model_t = args.pre_model_t
        pooling_type_s = args.pooling_type_s
        pooling_type_t = args.pooling_type_t
        weighted_loss = args.weighted_loss
        weight_exp_alpha = args.weight_exp_alpha
        weight_exp_bias = args.weight_exp_bias
        weight_exp_gamma = args.weight_exp_gamma
        focal_alpha = args.focal_alpha
        focal_gamma = args.focal_gamma
        ce_weight = args.ce_weight
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        only_evaluation = args.only_evaluation
        path_to_model_weights_s = args.path_to_model_weights_s
    else:
        print("EXIT")
        exit()

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    model_output_dirname = f"{LR}/B_{batch_size}F{number_of_frames}_G{gamma}_S{step_size}_mv{net_version_s}"
    model_output_dirname = f"{model_output_dirname}_t{pre_model_t}_s{pre_model_s}"
    best_model_path = os.path.join(
        "models",
        os.path.join(model_name,
                     os.path.join(str(num_views), os.path.join(pre_model_s, os.path.join(model_output_dirname)))
                     )
    )
    os.makedirs(best_model_path, exist_ok=True)

    log_path = os.path.join(best_model_path, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    # Initialize the data augmentation
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                          transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                          transforms.RandomRotation(degrees=5),
                                          transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                          transforms.RandomHorizontalFlip()
                                          ])
    else:
        transformAug = None

    if pre_model_s == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model_s == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()
    elif pre_model_s == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()
    elif pre_model_s == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model_s == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    elif pre_model_s == "swin3d_s":
        transforms_model = Swin3D_S_Weights.KINETICS400_V1.transforms()
        print("swin3d_s")
    elif pre_model_s == "swin3d_t":
        transforms_model = Swin3D_T_Weights.KINETICS400_V1.transforms()
    else:
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        print("Warning: Could not find the desired pretrained model")
        print("Possible options are: r3d_18, s3d, mc3_18, mvit_v2_s and r2plus1d_18")
        print("We continue with r2plus1d_18")

    if only_evaluation == 0:
        print('Evaluation mode 0 - only test set ...')

        dataset_test2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='test',
                                     num_views=num_views,
                                     transform_model=transforms_model)

        test_loader2 = torch.utils.data.DataLoader(dataset_test2,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)

    elif only_evaluation == 3:
        dataset_train = MultiViewDatasetHybrid(
            path=path,
            start=start_frame,
            end=end_frame,
            fps=fps,
            split='train',
            num_views=num_views,
            transform=transformAug,
            transform_model=transforms_model,
            video_shift_aug=video_shift_aug,
            weight_exp_alpha=weight_exp_alpha,
            weight_exp_bias=weight_exp_bias,
            weight_exp_gamma=weight_exp_gamma

        )
        dataset_valid2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='valid',
                                                num_views=5,
                                                transform_model=transforms_model)

        dataset_test2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='test',
                                               num_views=5,
                                               transform_model=transforms_model)
        print('Dataset initialization- finished')

        print('Dataloaders initalization ...')

        # Create the dataloaders for train validation and test datasets
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                  num_workers=max_num_worker, pin_memory=True)

        val_loader2 = DataLoader(dataset_valid2,
                                 batch_size=1, shuffle=False,
                                 num_workers=max_num_worker, pin_memory=True)

        test_loader2 = DataLoader(dataset_test2,
                                  batch_size=1, shuffle=False,
                                  num_workers=max_num_worker, pin_memory=False)

        print('Dataloaders initalization - finished')
    ###################################
    #       LOADING THE MODEL         #
    ###################################
    xin_network = XIN_NET_VERSION.get(net_version_s)
    print(xin_network, pooling_type_s)
    student_model = xin_network(num_views=num_views, net_name = pre_model_s, agr_type = pooling_type_s).cuda()

    if path_to_model_weights_s != "":
        path_model = os.path.join(path_to_model_weights_s)
        load = torch.load(path_model)
        student_model.load_state_dict(load['state_dict'])
        print("Weights student has been read.")
    else:
        print("Training student from scratch.")

    if only_evaluation == 3:

        optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR,
                                      betas=(0.92, 0.999), eps=1e-07,
                                      weight_decay=weight_decay, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        #scheduler = CustomStepLRScheduler(optimizer, step_size=step_size, gamma=gamma)
        epoch_start = 0

        teacher_model_path_list = [
            "models/VARS_XIN_reg01_bq23,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/11_model.pth.tar",
            "models/VARS_XIN_reg01_bq23b,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/14_model.pth.tar",
            "models/VARS_XIN_reg01_new_cs,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/26_model.pth.tar"
        ]
        teacher_models_list = []

        xin_network = XIN_NET_VERSION.get(net_version_t)

        teacher_model = xin_network(num_views=num_views, net_name = pre_model_s, agr_type = pooling_type_t).cuda()

        for path_to_model_weights_t in teacher_model_path_list:
            path_model = os.path.join(path_to_model_weights_t)
            load = torch.load(path_model)
            teacher_model.load_state_dict(load['state_dict'])
            teacher_model.eval()
            print(f"Weights teacher {path_to_model_weights_t} has been read.")
            teacher_models_list.append(teacher_model)

        criterion = select_training_loss(
            weighted_loss=weighted_loss,
            dataset_train=dataset_train,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            ce_weight=ce_weight,
        )

        run_label = model_output_dirname.replace("/", "_")
        current_date = datetime.now().strftime("%Y%b%d_%H%M")
        writer = SummaryWriter(f"runs/{current_date}_Xin_Distill_t{net_version_t}_s{net_version_s}_attention_{model_name}_{pre_model_s}_{run_label}")
        start_time = time.time()
        leadearboard_summary = trainer(
            train_loader, val_loader2, test_loader2,
            teacher_models_list=teacher_models_list,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            best_model_path=best_model_path,
            epoch_start=epoch_start,
            model_name=model_name,
            path_dataset=path,
            max_epochs=max_epochs,
            writer=writer, patience= args.patience,
            kd_temp = args.kd_temp,
            kd_lambda = args.kd_lambda
        )
        end_time = time.time()
        leadearboard_summary["training_time"] = round((end_time - start_time) / 3600, 4)
        hyperparams = {attr: str(value) for attr, value in vars(args).items()}

        for attr, value in hyperparams.items():
            writer.add_text(f'Hyperparameters/{attr}', value)

        # Alternatively, log all hyperparameters at once using add_hparams (if supported)
        writer.add_hparams(hyperparams, leadearboard_summary)
        writer.close()
        print(f"Training finished. Training time:{leadearboard_summary['training_time']}")
        
    if only_evaluation == 0:
        print("Only evaluation 0")
        evaluation_results = sklearn_evaluation(
            test_loader2, student_model, set_name="test", model_name = model_name,
        )
        print(evaluation_results)
        prediction_file = evaluation(
            test_loader2,
            student_model,
            set_name="test",
        )
        results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 4:
        print("Only evaluation 4")
        evaluation_results = sklearn_evaluation(
            train_loader, student_model, set_name="train", model_name = model_name,
        )
        print(evaluation_results)
        evaluation_results = sklearn_evaluation(
            val_loader2, student_model, set_name="valid", model_name = model_name,
        )
        print(evaluation_results)
        evaluation_results = sklearn_evaluation(
            test_loader2, student_model, set_name="test", model_name = model_name,
        )
        print(evaluation_results)
        prediction_file = evaluation(
            test_loader2,
            student_model,
            set_name="test",
        )
        results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    return 0


if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--max_epochs', required=False, type=int, default=60, help='Maximum number of epochs')
    parser.add_argument('--model_name', required=False, type=str, default="Xin_DISTILL", help='named of the model to save')
    parser.add_argument('--batch_size', required=False, type=int, default=2, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-04, help='Learning Rate')
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int, default=1, help='number of worker to load data')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--video_shift_aug", required=False, type=int, default=0, help="Number of video shifted clips")
    parser.add_argument("--pre_model_s", required=False, type=str, default="mvit_v2_s",
                        help="Name of the pretrained model")
    parser.add_argument("--pre_model_t", required=False, type=str, default="mvit_v2_s",
                        help="Name of the pretrained model")
    parser.add_argument("--net_version_s", required=False, type=int, default=25, help="MvAggregateModelVersion")
    parser.add_argument("--net_version_t", required=False, type=int, default=25, help="MvAggregateModelVersion")
    parser.add_argument("--pooling_type_s", required=False, type=str, default="attention",
                        help="Student model pooling type")
    parser.add_argument("--pooling_type_t", required=False, type=str, default="attention",
                        help="Teacher model pooling type")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Base",
                        help="Weighted loss version")
    parser.add_argument("--weight_exp_alpha", required=False, type=float, default=6.0,
                        help="weight_exp_hyperparam")
    parser.add_argument("--weight_exp_bias", required=False, type=float, default=0.1,
                        help="weighed exp bias hyper")
    parser.add_argument("--weight_exp_gamma", required=False, type=float, default=1.0,
                        help="weighted exp gamma hyper")
    parser.add_argument("--focal_alpha", required=False, type=float, default=1.0, help="focal_alpha")
    parser.add_argument("--focal_gamma", required=False, type=float, default=2.0,help="focal_gamma")
    parser.add_argument("--ce_weight", required=False, type=float, default=0.75, help="ce_weight")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=5, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.3, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=1e-3, help="Weight decacy")
    parser.add_argument("--patience", required=False, type=int, default=15, help="Earlystopping starting from 5 epoch.")
    parser.add_argument("--only_evaluation", required=False, type=int, default=3,
                        help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights_s", required=False, type=str, default="", help="Path to the student weights")
    parser.add_argument("--kd_temp", required=False, type=float, default=4.0, help="distill_temp")
    parser.add_argument("--kd_lambda", required=False, type=float, default=0.5,  help="Weight decacy")

    args = parser.parse_args()

    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Start the main training function
    start = time.time()
    logging.info('Starting main function')
    main(args, False)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
