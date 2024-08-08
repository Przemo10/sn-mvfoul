from src.custom_model.mvaggregate import MVAggregate, MVAggregate2
from typing import Union, Type
from torch import nn
from src.custom_model.xin_net import XinMultimodalNet
from src.custom_model.xin_net2 import XinMultimodalNet2
from src.custom_model.xin_net3 import XinMultimodalNet3
from src.custom_model.pooling_attention import WeightedAggregate, ViewAvgAggregate, ViewMaxAggregate
from src.custom_model.pooling_attention import ViewMaxMeanWeightAggregate, ViewMaxMeanAlphaAggregate


def select_baseline_mv_aggregate(mv_aggregate_version: int) -> Type[Union[MVAggregate, MVAggregate2]]:
    if mv_aggregate_version == 2:
        return MVAggregate2
    else:
        return MVAggregate


def select_xin_net(net_version: int) -> Type[Union[XinMultimodalNet, XinMultimodalNet2, XinMultimodalNet3]]:
    if net_version == 2:
        return XinMultimodalNet2
    elif net_version == 3:
        return XinMultimodalNet3
    else:
        return XinMultimodalNet


