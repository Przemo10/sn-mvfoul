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

PRETRAINED_VIDEO_MAE = {

    1: "MCG-NJU/videomae-base-finetuned-kinetics",
    2: "MCG-NJU/videomae-large-finetuned-kinetics",
    3: "MCG-NJU/videomae-base-finetuned-ssv2",
    4: "marekk/video_soccer_goal_detection",
    5: "anirudhmu/videomae-base-finetuned-soccer-action-recognition",
}


TEACHERS_CONFIG = {
    'attention_rand': [
        "models/VARS_XIN_reg01_bq23,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/11_model.pth.tar",
        "models/VARS_XIN_reg01_bq23b,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/14_model.pth.tar,"
        "models/VARS_XIN_reg01_new_cs,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pattention/26_model.pth.tar"
    ],
    "max_pool":[
        "models/VARS_XIN_pred_max_pool1,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/24_model.pth.tar",
        "models/VARS_XIN_pred_max_pool2cs,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/28_model.pth.tar",
        "models/VARS_XIN_pred_max_pool3csB_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/26_model.pth.tar"
    ],
    "max_pool_weak": [
        "models/VARS_XIN_pred_max_pool1,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/11_model.pth.tar",
        "models/VARS_XIN_pred_max_pool2cs,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/11_model.pth.tar",
        "models/VARS_XIN_pred_max_pool3csB_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/11_model.pth.tar"
    ]

}
STUDENT_CONFIG = {
    "n1":
        "models/VARS_XIN_v2,_v15/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv15_pattention/6_model.pth.tar",
    "n2_max_pool":
        "models/VARS_XIN_pred_max_pool1,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/6_model.pth.tar",
    "n2_max_pool_v2":
        "models/VARS_XIN_pred_max_pool1,_v25/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv25_pmax/6_model.pth.tar",
    "n1_max_pool":
        "models/VARS_XIN_pred_max_poool_max_n1_v15/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv15_pmax/6_model.pth.tar",
    "n1_max_pool_v2":
        "models/VARS_XIN_pred_max_poool_max_n1_v15/5/mvit_v2_s/5e-05_WeightedFocal/B_4F16_G0.5_S3_mv15_pmax/9_model.pth.tar",

}


