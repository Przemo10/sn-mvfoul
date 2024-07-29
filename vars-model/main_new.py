import os
import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from src.custom_dataset.baseline_dataset import MultiViewDataset
from src.custom_dataset.video_mae_dataset import MultiViewMAEDataset
from src.custom_dataset.hybrid_dataset import MultiViewDatasetHybrid
from torch.utils.data import  DataLoader
from src.custom_trainers.train import trainer, evaluation
from src.custom_trainers.train_xin import trainer as xin_trainer, evaluation as xin_evaluation
from src.custom_trainers.train_hybrid import trainer as hibrid_trainer
from src.custom_trainers.train_videomae import trainer as mae_trainer, evaluation as mae_evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from src.custom_model.baseline_model import MVNetwork
from src.custom_model.xin_net2 import XinMultimodalNet2
from src.custom_model.video_mae import VideoMAENetwork
from src.custom_model.hybrid_mvit_v2 import MultiVideoHybridMVit2
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
from torchvision.models.video import MViT_V2_S_Weights
from torchvision.models.video import swin3d_s,  Swin3D_S_Weights


def checkArguments():

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type not in ['max', 'mean', 'attention', 'max_mean_alpha', 'max_mean_weight']:
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()

    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or  args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or  args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or  args.fps < 1:
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
        model_name = args.model_name
        pre_model = args.pre_model
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        video_shift_aug = args.video_shift_aug
        path = args.path
        pooling_type = args.pooling_type
        weighted_loss = args.weighted_loss
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


    best_model_path = os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size))))))
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
    if args.custom_model == "xin":
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
                video_shift_aug=video_shift_aug)
            dataset_valid2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='valid',
                                                    num_views=num_views,
                                                    transform_model=transforms_model)

            dataset_test2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='test',
                                                   num_views=num_views,
                                                   transform_model=transforms_model)
            print('Dataset initialization- finished')

            print('Dataloaders initalization ...')

            # Create the dataloaders for train validation and test datasets
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      num_workers=max_num_worker, pin_memory=True)

            val_loader2 = DataLoader(dataset_valid2,
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=max_num_worker, pin_memory=True)

            test_loader2 = DataLoader(dataset_test2,
                                      batch_size=batch_size, shuffle=False,
                                      num_workers=max_num_worker, pin_memory=False)

            print('Dataloaders initalization - finished')
    elif args.custom_model == "mae":
        if data_aug == 'Yes':
            transformAug = transforms.Compose([
                                              transforms.RandomRotation(degrees=5),
                                              transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                              transforms.RandomHorizontalFlip()
                                              ])
        else:
            transformAug = None

        dataset_Train = MultiViewMAEDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='train',
                                            num_views=num_views, transform=transformAug)
        dataset_Valid2 = MultiViewMAEDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='valid',
                                             num_views=5)
        dataset_Test2 = MultiViewMAEDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='test', num_views=5)

        print('Dataset initialization- finished')

        print('Dataloaders initalization ...')
        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
                                                   batch_size=batch_size, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
    
        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=max_num_worker, pin_memory=True)
    
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
    elif args.custom_model == "hibrid":
        if data_aug == 'Yes':
            transformAug = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomHorizontalFlip()
            ])
        else:
            transformAug = None
    
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        dataset_Train = MultiViewDatasetHybrid(path=path,
                                         start=start_frame,
                                         end=end_frame,
                                         fps=fps,
                                         split='train',
                                         num_views=num_views,
                                         transform=transformAug,
                                         transform_model=transforms_model,
                                         video_shift_aug=video_shift_aug)
    
        dataset_Valid2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='valid',
                                          num_views=num_views,
                                          transform_model=transforms_model)
    
        dataset_Test2 = MultiViewDatasetHybrid(path=path, start=start_frame, end=end_frame, fps=fps, split='test',
                                         num_views=num_views,
                                         transform_model=transforms_model)
    
    
    
        print('Dataset initialization- finished')
    
        print('Dataloaders initalization ...')
    
        # Create the dataloaders for train validation and test datasets
        train_loader = DataLoader(dataset_Train, batch_size=batch_size, shuffle=True,
                                  num_workers=max_num_worker, pin_memory=True)
    
        val_loader2 = DataLoader(dataset_Valid2,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=max_num_worker, pin_memory=True)
    
        test_loader2 = DataLoader(dataset_Test2,
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=max_num_worker,  pin_memory=False)
    
        print('Dataloaders initalization - finished')
    else:
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
        else:
            transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
            print("Warning: Could not find the desired pretrained model")
            print("Possible options are: r3d_18, s3d, mc3_18, mvit_v2_s and r2plus1d_18")
            print("We continue with r2plus1d_18")

        if only_evaluation == 0:
            print('Evaluation mode 0 - only test set ...')
            dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='test', num_views = 5,
            transform_model=transforms_model)

            test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                batch_size=1, shuffle=False,
                num_workers=max_num_worker, pin_memory=True)
        elif only_evaluation == 1:
            print('Evaluation mode 1 - only chall set ...')
            dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='chall', num_views = 5,
            transform_model=transforms_model)

            chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
                batch_size=1, shuffle=False,
                num_workers=max_num_worker, pin_memory=True)
        elif only_evaluation == 2:
            print('Evaluation mode 2 ...')
            dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='test', num_views = 5,
            transform_model=transforms_model)
            dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='chall', num_views = 5,
            transform_model=transforms_model)

            test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                batch_size=1, shuffle=False,
                num_workers=max_num_worker, pin_memory=True)

            chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
                batch_size=1, shuffle=False,
                num_workers=max_num_worker, pin_memory=True)
        else:

            print('Dataset initialization- starts... ')
            # Create Train Validation and Test datasets
            dataset_Train = MultiViewDataset(
                path=path, start=start_frame, end=end_frame, fps=fps, split='train', num_views = num_views,
                transform=transformAug, transform_model=transforms_model,video_shift_aug=video_shift_aug)
            dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='valid', num_views = 5,
                transform_model=transforms_model)
            dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='test', num_views = 5,
                transform_model=transforms_model)

            print('Dataset initialization- finished')

            print('Dataloaders initalization ...')

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

        print('Dataloaders initalization - finished')

    ###################################
    #       LOADING THE MODEL         #
    ###################################
    if args.custom_model == "xin":
        model = XinMultimodalNet2(num_views=num_views, net_name = pre_model).cuda()

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

            if weighted_loss == 'Yes':
                criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[0].cuda())
                criterion_action = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[1].cuda())
                criterion = [criterion_offence_severity, criterion_action]
            else:
                criterion_offence_severity = nn.CrossEntropyLoss()
                criterion_action = nn.CrossEntropyLoss()
                criterion = [criterion_offence_severity, criterion_action]

            xin_trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion,
                    best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs)

        if only_evaluation == 0:

            prediction_file = xin_evaluation(
                test_loader2,
                model,
                set_name="test",
            )
            results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
            print("TEST")
            print(results)

        return 0
    elif args.custom_model == "mae":
        model = VideoMAENetwork()

        if path_to_model_weights != "":
            path_model = os.path.join(path_to_model_weights)
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])

        if only_evaluation == 3:

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                          betas=(0.9, 0.999), eps=1e-07,
                                          weight_decay=weight_decay, amsgrad=False)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            epoch_start = 0

            if continue_training:
                path_model = os.path.join(log_path, 'model.pth.tar')
                load = torch.load(path_model)
                model.load_state_dict(load['state_dict'])
                optimizer.load_state_dict(load['optimizer'])
                scheduler.load_state_dict(load['scheduler'])
                epoch_start = load['epoch']

            if weighted_loss == 'Yes':
                criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
                criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
                criterion = [criterion_offence_severity, criterion_action]
            else:
                criterion_offence_severity = nn.CrossEntropyLoss()
                criterion_action = nn.CrossEntropyLoss()
                criterion = [criterion_offence_severity, criterion_action]

        # Start training or evaluation
        if only_evaluation == 0:
            prediction_file = mae_evaluation(
                test_loader2,
                model,
                set_name="test",
            )
            results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
            print("TEST")
            print(results)
        elif only_evaluation == 2:
            pass
        else:
            mae_trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion,
                    best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs)

        return 0
    elif args.custom_model == "hibrid":
        model = MultiVideoHybridMVit2(num_views=num_views).cuda()

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
                path_model = os.path.join(log_path, 'model.pth.tar')
                load = torch.load(path_model)
                model.load_state_dict(load['state_dict'])
                optimizer.load_state_dict(load['optimizer'])
                scheduler.load_state_dict(load['scheduler'])
                epoch_start = load['epoch']

            if weighted_loss == 'Yes':
                criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
                criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
                criterion = [criterion_offence_severity, criterion_action]
            else:
                criterion_offence_severity = nn.CrossEntropyLoss()
                criterion_action = nn.CrossEntropyLoss()
                criterion = [criterion_offence_severity, criterion_action]

            hibrid_trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion,
                    best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs)

        return 0
    else:
        model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()

        if path_to_model_weights != "":
            path_model = os.path.join(path_to_model_weights)
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])

        if only_evaluation == 3:

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                        betas=(0.9, 0.999), eps=1e-07, 
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


            if weighted_loss == 'Yes':
                criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
                criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
                criterion = [criterion_offence_severity, criterion_action]
            else:
                criterion_offence_severity = nn.CrossEntropyLoss()
                criterion_action = nn.CrossEntropyLoss()
                criterion = [criterion_offence_severity, criterion_action]


        # Start training or evaluation
        if only_evaluation == 0:
            prediction_file = evaluation(
                test_loader2,
                model,
                set_name="test",
            ) 
            results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
            print("TEST")
            print(results)

        elif only_evaluation == 1:
            prediction_file = evaluation(
                chall_loader2,
                model,
                set_name="chall",
            )

            results = evaluate(os.path.join(path, "chall", "annotations.json"), prediction_file)
            print("CHALL")
            print(results)

        elif only_evaluation == 2:
            prediction_file = evaluation(
                test_loader2,
                model,
                set_name="test",
            )

            results = evaluate(os.path.join(path, "test", "annotations.json"), prediction_file)
            print("TEST")
            print(results)

            prediction_file = evaluation(
                chall_loader2,
                model,
                set_name="chall",
            )

            results = evaluate(os.path.join(path, "chall", "annotations.json"), prediction_file)
            print("CHALL")
            print(results)
        else:
            trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, 
                    best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs)

        return 0



if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=60,     help='Maximum number of epochs' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VARS_shift3",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--video_shift_aug", required=False, type=int, default=0, help="Number of video shifted clips")
    parser.add_argument("--pre_model", required=False, type=str, default="mvit_v2_s", help="Name of the pretrained model")
    parser.add_argument("--pooling_type", required=False, type=str, default="mean", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the custom_loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.3, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")
    parser.add_argument("--custom_model", required=False, type=str, default="MVNetwork", help="Custom model: xin, mae, hibrid")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3, help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to the model weights")

    args = parser.parse_args()

    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args, False)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
