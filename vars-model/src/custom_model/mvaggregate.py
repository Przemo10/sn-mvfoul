from scipy.sparse import random
from six import print_
from src.utils import batch_tensor, unbatch_tensor
import torch
from torch import nn
from src.custom_model.pooling_attention import select_pooling_attention


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=model,
            lifting_net=lifting_net,
            feat_dim=feat_dim)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages) # dla mvit (4,400),  (4,2,400)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention


class MVAggregate10(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 8)
        )

        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=model,
            lifting_net=lifting_net,
            feat_dim=feat_dim
        )

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages) # dla mvit (4,400),  (4,2,400)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention


class MVAggregate2(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 8)
        )

        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=model,
            lifting_net=lifting_net,
            feat_dim=feat_dim
        )

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages) # dla mvit (4,400),  (4,2,400)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention