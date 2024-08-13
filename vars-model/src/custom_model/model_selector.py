from src.custom_model.mvaggregate import MVAggregate, MVAggregate2
from typing import Union, Type
from torch import nn
from src.custom_model.xin_net import XinMultimodalNet
from src.custom_model.xin_net_n1 import  XinMultimodalNetN10, XinMultimodalNetN15, XinMultimodalNetN16
from src.custom_model.xin_net_n2 import  XinMultimodalNetN20, XinMultimodalNetN25


def select_baseline_mv_aggregate(mv_aggregate_version: int) -> Type[Union[MVAggregate, MVAggregate2]]:
    if mv_aggregate_version == 2:
        return MVAggregate2
    else:
        return MVAggregate

#  first number 1 concat
#  first number 2 attention

XIN_NET_VERSION = {

    10: XinMultimodalNetN10,
    15: XinMultimodalNetN15,
    16: XinMultimodalNetN16,
    20: XinMultimodalNetN20,
    25: XinMultimodalNetN25,
     1: XinMultimodalNet

}

PRETRAINED_VIDEO_MAE  = {

    1: "MCG-NJU/videomae-base-finetuned-kinetics",
    2: "MCG-NJU/videomae-large-finetuned-kinetics",
    3: "MCG-NJU/videomae-base-finetuned-ssv2",
    4: "marekk/video_soccer_goal_detection",
    5: "anirudhmu/videomae-base-finetuned-soccer-action-recognition",
}



