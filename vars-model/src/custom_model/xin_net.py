import torch

from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from transformers import VideoMAEForVideoClassification
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torch import  nn
from src.utils import batch_tensor, unbatch_tensor
from src.custom_model.select_feature_extract_net import get_feature_network


class XinMultimodalNet(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s'):
        super().__init__()

        self.model,  self.feat_dim = get_feature_network(net_name=net_name)
        self.model.head = nn.Identity()
        self.num_views = num_views
        # self.feat_dim = 400
        self.token_dim = 768
        self.drop_layer = nn.Dropout(0.2)
        self.lifting_net = nn.Sequential()
        self.res_perceptor_block = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.Sigmoid(),
            nn.LayerNorm(self.token_dim),
            nn.Dropout(0.2),
        )
        self.fc_offence = nn.Sequential(
            nn.Linear(self.token_dim, 4)
        )
        self.fc_action = nn.Sequential(
            nn.Linear(self.token_dim, 8)
        )
        self.intern = nn.Sequential(
            nn.Linear(self.token_dim * num_views,  self.token_dim * num_views),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim * num_views),
            nn.Dropout(0.2)
        )
        self.fc_mv_offence = nn.Linear(self.token_dim * num_views, 4)
        self.fc_mv_actions = nn.Linear(self.token_dim * num_views, 8)

    def forward(self, mvimages):
        output_dict = {'mv_collection': {}}
        B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
        # aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        # x = self.drop_layer(aux)
        features_extractor_list = []
        for i in range(V):
            aux = self.lifting_net(self.model(mvimages[:, i]))
            x = self.drop_layer(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            x1 = self.res_perceptor_block(x)
            x = torch.add(x, x1)
            features_extractor_list.append(x)

        x = torch.cat(features_extractor_list, dim=1)
        x = self.intern(x)
        mv_offence = self.fc_mv_offence(x)
        mv_actions = self.fc_mv_actions(x)
        output_dict["mv_collection"]["offence_logits"] = mv_offence
        output_dict["mv_collection"]["action_logits"] = mv_actions
        # print(x.shape)

        return output_dict

"""
model = XinMultimodalNet(num_views=2)

for  x,y in model.named_parameters():
    print(x, y.shape)

videos = torch.randn(4, 2, 3, 16, 224, 224)

output = model(videos)
print(output)
print(output.keys(), model.num_views)
for i in range(model.num_views):
    print(f"single_{i}")

print(torch.argmax(output['single_1']['action_logits'], dim=1))
"""