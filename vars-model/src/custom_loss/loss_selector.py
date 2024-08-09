from focal_cross_entropy import FocalLoss
from weighted_focal_ce_loss import WeightedFocalCELoss
from typing import Union, Type
from src.custom_dataset.baseline_dataset import MultiViewDataset
from src.custom_dataset.hybrid_dataset import MultiViewDatasetHybrid
from src.custom_dataset.video_mae_dataset import MultiViewMAEDataset
from torch import nn


def loss_selector(
        weighted_loss: str,
        dataset_train: Type[Union[MultiViewDataset, MultiViewDatasetHybrid, MultiViewMAEDataset]]
):
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