from src.custom_loss.focal_cross_entropy  import FocalLoss
from src.custom_loss.weighted_focal_ce_loss import WeightedFocalCELoss
from src.custom_loss.weighted_focal_loss import WeightedFocalLoss
from typing import Union, Type, List
from src.custom_dataset.baseline_dataset import MultiViewDataset
from src.custom_dataset.hybrid_dataset import MultiViewDatasetHybrid
from src.custom_dataset.video_mae_dataset import MultiViewMAEDataset
import torch.nn as nn


def select_training_loss(
        weighted_loss: str,
        dataset_train: Union[MultiViewDataset, MultiViewDatasetHybrid, MultiViewMAEDataset],
        focal_alpha =  1.0,
        focal_gamma = 2.0,
        ce_weight = 0.75,
) -> List:
    if weighted_loss == 'Exp':
        print(dataset_train.getExpotentialWeight())
        criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_train.getExpotentialWeight()[0].cuda())
        criterion_action = nn.CrossEntropyLoss(weight=dataset_train.getExpotentialWeight()[1].cuda())

    elif weighted_loss in ['Base', 'Yes']:
        print(dataset_train.getWeights())
        criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[0].cuda())
        criterion_action = nn.CrossEntropyLoss(weight=dataset_train.getWeights()[1].cuda())
    elif weighted_loss == 'Focal':
        criterion_offence_severity = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=None)
        criterion_action = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=None)

    elif weighted_loss == 'FocalCE':
        criterion_offence_severity = WeightedFocalCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ce_weight=ce_weight,
            weight=dataset_train.getExpotentialWeight()[0].cuda()
        )
        criterion_action = WeightedFocalCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ce_weight=ce_weight,
            weight=dataset_train.getExpotentialWeight()[1].cuda()
        )
    elif weighted_loss == 'WeightedFocal':
        criterion_offence_severity = WeightedFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            weights=dataset_train.getWeights()[0].cuda()
        )
        criterion_action = WeightedFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            weights=dataset_train.getWeights()[1].cuda()
        )
    else:
        criterion_offence_severity = nn.CrossEntropyLoss()
        criterion_action = nn.CrossEntropyLoss()
    criterion = [criterion_offence_severity, criterion_action]
    return criterion