import torch
from src.custom_model.model_selector import select_baseline_mv_aggregate
from src.custom_model.select_feature_extract_net import get_feature_network
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.video import swin3d_s, Swin3D_S_Weights
from transformers import VideoMAEForVideoClassification


class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18',
                 agr_type='max',
                 lifting_net=torch.nn.Sequential(),
                 mv_aggregate_version=0):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net

        network, self.feat_dim = get_feature_network(self.net_name)
        network.fc = torch.nn.Sequential()

        selcted_mv_aggregate_model = select_baseline_mv_aggregate(mv_aggregate_version)

        self.mvnetwork = selcted_mv_aggregate_model(
            model=network, agr_type=self.agr_type, feat_dim=self.feat_dim, lifting_net=self.lifting_net)

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)
