import torch
from transformers import VideoMAEForVideoClassification
from torch import  nn
from src.utils import batch_tensor, unbatch_tensor


class VideoMAENetwork(torch.nn.Module):

    def __init__(self, agr_type='max'):
        super().__init__()

        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model.classifier = nn.Identity()
        self.feet_dim = 400
        self.norm_dim = 768
        self.flatten_shape = 3072
        self.lifting_net = nn.Sequential()
        self.inter = nn.Sequential(
            nn.LayerNorm(self.norm_dim),
            nn.Linear(self.norm_dim ,self.feet_dim),
            nn.Linear(self.feet_dim, self.feet_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.norm_dim ),
            nn.Linear(self.norm_dim,  self.feet_dim),
            nn.ReLU(),
            nn.Linear(self.feet_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(self.norm_dim ),
            nn.Linear(self.norm_dim ,self.feet_dim),
            nn.ReLU(),
            nn.Linear(self.feet_dim, 8)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc_action:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.fc_offence:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, mvimages):
        # print(mvimages.shape) torch.Size([4, 2, 16, 3, 224, 224])
        B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height,
        output = self.model(batch_tensor(mvimages, dim=1, squeeze=True))[0]
        # print(output.shape) (8, 400)
        aux = self.lifting_net(unbatch_tensor(output, B, dim=1, unsqueeze=True))
        # print(output.shape, aux.shape) (8, 1, 768), (4,2 ,768) aux
        pooled_view = torch.mean(aux, dim=1)
        # poolded_view (4, 400), aux (4, 2, 400) (B,views, 400)
        pred_action = self.fc_action(pooled_view)
        pred_offence_severity = self.fc_offence(pooled_view )
        return pred_offence_severity, pred_action, aux
