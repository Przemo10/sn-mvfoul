import os
import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from torch.nn.functional import prelu
from torch.utils.tensorboard import SummaryWriter
from src.custom_dataset.hybrid_dataset import MultiViewDatasetHybrid
from src.custom_trainers.train_xin import trainer, evaluation, sklearn_evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
from src.custom_model.model_selector import select_xin_net
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

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean' and args.pooling_type != 'attention':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()

    # args.weighted_loss
    if args.weighted_loss not in ["Base", "No", "Exp", "Yes"]:
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Base, No, Exp")
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
        model_name = f"{args.model_name},_v{args.net_version}"
        net_version = args.net_version
        pre_model = args.pre_model
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / (
                    (args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        pooling_type = "no-pooling"
        weighted_loss = args.weighted_loss
        weight_exp_alpha = args.weight_exp_alpha
        weight_exp_bias = args.weight_exp_bias
        weight_exp_gamma = args.weight_exp_gamma
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
    else:
        print("EXIT")
        exit()

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    output_dir_name = f"{LR}/B_{batch_size}F{number_of_frames}_G{gamma}_S{step_size}_mv{net_version}_p{pooling_type}"

    best_model_path = os.path.join(
        "models",
        os.path.join(model_name,
                     os.path.join(str(num_views), os.path.join(pre_model, os.path.join(output_dir_name)))
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

    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    elif pre_model == "swin3d_s":
        transforms_model = Swin3D_S_Weights.KINETICS400_V1.transforms()
        print("swin3d_s")
    elif pre_model == "swin3d_t":
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
    xin_network = select_xin_net(net_version)
    model = xin_network(num_views=num_views, net_name = pre_model).cuda()

    if path_to_model_weights != "":
        path_model = os.path.join(path_to_model_weights)
        load = torch.load(path_model)
        model.load_state_dict(load['state_dict'])

    if only_evaluation == 3:

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      betas=(0.9, 0.95), eps=1e-04,
                                      weight_decay=weight_decay, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        epoch_start = 0

        if continue_training:
            print(2048)
            path_model = os.path.join(log_path, 'model.pth.tar')
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])
            optimizer.load_state_dict(load['optimizer'])
            scheduler.load_state_dict(load['scheduler'])
            epoch_start = load['epoch']

        if weighted_loss == 'Exp':
            print(dataset_train.getExpotentialWeight())
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_train.getExpotentialWeight()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_train.getExpotentialWeight()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        elif weighted_loss in ['Base', 'Yes']:
            print(dataset_train.getWeights())
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        else:
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            criterion = [criterion_offence_severity, criterion_action]

        run_label = output_dir_name.replace("/", "_")
        current_date = datetime.now().strftime("%Y%b%d_%H%M")
        writer = SummaryWriter(f"runs/{current_date}_Xin{net_version}_attention_{model_name}_{pre_model}_{run_label}")
        start_time = time.time()
        leadearboard_summary = trainer(
            train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion,
            best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs,
            writer=writer, patience= args.patience
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
            test_loader2, model, set_name="test", model_name = model_name,
        )
        print(evaluation_results)
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
        )
        results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 4:
        print("Only evaluation 4")
        evaluation_results = sklearn_evaluation(
            train_loader, model, set_name="train", model_name = model_name,
        )
        print(evaluation_results)
        evaluation_results = sklearn_evaluation(
            val_loader2, model, set_name="valid", model_name = model_name,
        )
        print(evaluation_results)
        evaluation_results = sklearn_evaluation(
            test_loader2, model, set_name="test", model_name = model_name,
        )
        print(evaluation_results)
        prediction_file = evaluation(
            test_loader2,
            model,
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
    parser.add_argument('--model_name', required=False, type=str, default="Xin_VAR2", help='named of the model to save')
    parser.add_argument('--batch_size', required=False, type=int, default=2, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-04, help='Learning Rate')
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int, default=1, help='number of worker to load data')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=2, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--video_shift_aug", required=False, type=int, default=0, help="Number of video shifted clips")
    parser.add_argument("--pre_model", required=False, type=str, default="mvit_v2_s",
                        help="Name of the pretrained model")
    parser.add_argument("--net_version", required=False, type=int, default=3, help="XinNetVersion")
    parser.add_argument("--pooling_type", required=False, type=str, default="max",
                        help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Base",
                        help="Weighted loss version")
    parser.add_argument("--weight_exp_alpha", required=False, type=float, default=8.0,
                        help="weight_exp_hyperparam")
    parser.add_argument("--weight_exp_bias", required=False, type=float, default=0.02,
                        help="weighed exp bias hyper")
    parser.add_argument("--weight_exp_gamma", required=False, type=float, default=2.0,
                        help="weighted exp gamma hyper")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=5, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.3, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3,
                        help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="",
                        help="Path to the model weights")

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
