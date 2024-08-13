import torch
from torch import  nn
from src.utils import batch_tensor, unbatch_tensor
from src.custom_model.select_feature_extract_net import get_feature_network
from src.custom_model.pooling_attention import  select_pooling_attention


class XinMultimodalNetN10(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s',freeze_layers = 0):
        super().__init__()

        network,  self.feat_dim, self.freeze_up_to  = get_feature_network(net_name=net_name)

        if freeze_layers > 0:
            self.model = self.freezee_net_layer(network)
        else:
            self.model = network
        self.model.head = nn.Identity()
        self.num_views = num_views
        # self.feat_dim = 400
        self.token_dim = 768
        self.drop_layer = nn.Dropout(0.2)
        self.lifting_net = nn.Sequential()
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
            x = self.drop_layer(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            features_extractor_list.append(x)
        x = torch.cat(features_extractor_list, dim=1)
        x = self.intern(x)
        mv_offence = self.fc_mv_offence(x)
        mv_actions = self.fc_mv_actions(x)
        output_dict["mv_collection"]["offence_logits"] = mv_offence
        output_dict["mv_collection"]["action_logits"] = mv_actions
        # print(x.shape)

        return output_dict


class XinMultimodalNetN15(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s',freeze_layers = 0, agr_type = "max"):
        super().__init__()

        network,  self.feat_dim, self.freeze_up_to  = get_feature_network(net_name=net_name)


        if freeze_layers > 0:
            self.model = self.freezee_net_layer(network)
        else:
            self.model = network
        self.num_views = num_views
        self.feat_dim = 400
        self.token_dim = 768
        self.drop_layer = nn.Dropout(0.2)
        self.drop_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
        )
        self.lifting_net = nn.Sequential()
        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=self.model,
            lifting_net=self.lifting_net,
            feat_dim=self.feat_dim
        )

        self.fc_offence = nn.Sequential(
            nn.Linear(self.feat_dim, 4)
        )
        self.fc_action = nn.Sequential(
            nn.Linear(self.feat_dim, 8)
        )
        self.intern = nn.Sequential(
            nn.Linear(self.feat_dim,  self.feat_dim),
            nn.LayerNorm(self.feat_dim),
        )
        self.fc_mv_offence = nn.Linear(self.feat_dim, 4)
        self.fc_mv_actions = nn.Linear(self.feat_dim, 8)
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
            x = self.drop_layer(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            features_extractor_list.append(x)
        x = torch.stack(features_extractor_list, dim=1)
        x, _ = self.aggregation_model(mvimages=x, aux=1)
        x = self.intern(x)
        mv_offence = self.fc_mv_offence(x)
        mv_actions = self.fc_mv_actions(x)
        output_dict["mv_collection"]["offence_logits"] = mv_offence
        output_dict["mv_collection"]["action_logits"] = mv_actions
        # print(x.shape)

        return output_dict

class XinMultimodalNetN16(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s',freeze_layers = 0, agr_type = "max"):
        super().__init__()

        network,  self.feat_dim, self.freeze_up_to  = get_feature_network(net_name=net_name)


        if freeze_layers > 0:
            self.model = self.freezee_net_layer(network)
        else:
            self.model = network
        self.num_views = num_views
        self.feat_dim = 400
        self.token_dim = 400
        self.drop_layer = nn.Dropout(0.2)
        self.lifting_net = nn.Sequential()
        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=self.model,
            lifting_net=self.lifting_net,
            feat_dim=self.feat_dim
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 8)
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 4)
        )
        self.inter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Linear(self.feat_dim, 4)
        )

        self.fc_mv_actions= nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Linear(self.feat_dim, 8)
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
            x = self.drop_layer(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            features_extractor_list.append(x)
        x = torch.stack(features_extractor_list, dim=1)
        x, _ = self.aggregation_model(mvimages=x, aux=1)
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