import torch
from torch import  nn
from src.utils import batch_tensor, unbatch_tensor
from src.custom_model.select_feature_extract_net import get_feature_network


class XinMultimodalNet3(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s', freeze_layers=0):
        super().__init__()

        network, self.feat_dim, self.freeze_up_to = get_feature_network(net_name=net_name)

        if freeze_layers > 0:
            self.model = self.freezee_net_layer(network)
        else:
            self.model = network

        self.model,  self.feat_dim, self.freeze_up_to  = get_feature_network(net_name=net_name)
        self.model.head = nn.Identity()
        self.num_views = num_views
        # self.feat_dim = 400
        # self.token_dim = 768
        self.token_dim = 400
        self.last_extractor_block = nn.Sequential(
            nn.Linear(768, self.token_dim),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim),
            nn.Dropout(0.2)
        )
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
        self.intern= nn.Sequential(
            nn.Linear(self.token_dim,  self.token_dim),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim),
            nn.Dropout(0.2),
        )

    def freezee_net_layer(self, network):
        freeze = True
        for name, param in network.named_parameters():
            if freeze:
                param.requires_grad = False
            if name.startswith(self.freeze_up_to):
                freeze = False  # Stop freezing after the last parameter of blocks.10

        return  network


    def forward(self, mvimages):
        output_dict = {'mv_collection': {}}
        B, V, C, D, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
        # aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        # x = self.drop_layer(aux)
        features_extractor_list = []
        for i in range(V):
            aux = self.lifting_net(self.model(mvimages[:, i]))
            x = self.last_extractor_block(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            x1 = self.res_perceptor_block(x)
            x = torch.add(x, x1)
            x = x.unsqueeze(1)
            features_extractor_list.append(x)

        x = torch.cat(features_extractor_list, dim=1)
        x = 0.5 * torch.max(x, dim=1)[0] + 0.5 * torch.mean(x, dim=1)
        x = self.intern(x)
        mv_offence = self.fc_offence(x)
        mv_actions = self.fc_action(x)
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