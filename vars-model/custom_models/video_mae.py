import torch
from transformers import VideoMAEForVideoClassification
from torch import  nn
from utils import batch_tensor, unbatch_tensor


class VideoMAENetwork(torch.nn.Module):

    def __init__(self, agr_type='max'):
        super().__init__()

        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.feet_dim = 400
        self.lifting_net = nn.Sequential()
        self.inter = nn.Sequential(
            nn.LayerNorm(self.feet_dim),
            nn.Linear(self.feet_dim,self.feet_dim),
            nn.Linear(self.feet_dim, self.feet_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.feet_dim),
            nn.Linear(self.feet_dim, self.feet_dim),
            nn.Linear(self.feet_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(self.feet_dim),
            nn.Linear(self.feet_dim, self.feet_dim),
            nn.Linear(self.feet_dim, 8)
        )

    def forward(self, mvimages):
        print(mvimages.shape)
        B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(
            unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))

        pooled_view = torch.max(aux, dim=1)[0]
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, aux
