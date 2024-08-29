from interface.src.utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        num_heads = 8
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )

        self.relu = nn.ReLU()

    def forward(self, mvimages, aux=None):
        if not aux:
            B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
            aux = self.lifting_net(
                unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
            )
        else:
            aux = mvimages
            B, V, _ = aux.shape

        ##################### VIEW ATTENTION #####################

        # S = source length
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)

        aux_sum = torch.sum(torch.reshape(relu_res, (B, V * V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V * V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages, aux=None):
        if not aux:
            B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
            aux = self.lifting_net(
                unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
            )
        else:
            aux = mvimages
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages, aux=None):
        if not aux:
            B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
            aux = self.lifting_net(
                unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
            )
        else:
            aux = mvimages
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class ViewMaxMeanAlphaAggregate(nn.Module):

    # source article https://s3.ap-northeast-2.amazonaws.com/journal-home/journal/jips/fullText/302/13.pdf
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        random_value = torch.rand(1)
        self.alpha = nn.Parameter(random_value)
        self.beta = nn.Parameter(1 - random_value)
        self.relu = nn.ReLU()

    def forward(self, mvimages, aux=None):
        if not aux:
            B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
            aux = self.lifting_net(
                unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
            )
        else:
            aux = mvimages
        max_pool = torch.max(aux, dim=1)[0]
        mean_pool = torch.mean(aux, dim=1)
        pooled_view = self.alpha * mean_pool + self.beta * max_pool
        pooled_view = self.relu(pooled_view)

        return pooled_view.squeeze(), aux


class ViewMaxMeanWeightAggregate(nn.Module):

    # source article https://s3.ap-northeast-2.amazonaws.com/journal-home/journal/jips/fullText/302/13.pdf
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim
        w1 = torch.rand(self.feat_dim)
        w2 = torch.rand(self.feat_dim)
        self.max_pool_weight = nn.Parameter(torch.div(w1, torch.sum(w1)))
        self.mean_pool_weight = nn.Parameter(torch.div(w2, torch.sum(w2)))
        self.relu = nn.ReLU()

    def forward(self, mvimages, aux=None):
        if not aux:
            B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
            aux = self.lifting_net(
                unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
            )
        else:
            aux = mvimages
        max_pool = torch.max(aux, dim=1)[0]
        mean_pool = torch.mean(aux, dim=1)
        pooled_view = 0.5 * torch.mul(self.max_pool_weight, max_pool) + 0.5 * torch.mul(self.mean_pool_weight,
                                                                                        mean_pool)
        pooled_view = self.relu(pooled_view)

        return pooled_view.squeeze(), aux


def select_pooling_attention(agr_type: str, model: nn.Module, lifting_net=nn.Sequential(), feat_dim=400):
    if agr_type == "max":
        aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
    elif agr_type == "mean":
        aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
    elif agr_type == "max_mean_alpha":
        aggregation_model = ViewMaxMeanAlphaAggregate(model=model, lifting_net=lifting_net)
    elif agr_type == "max_mean_weight":
        aggregation_model = ViewMaxMeanWeightAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
    else:
        aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
    return aggregation_model
