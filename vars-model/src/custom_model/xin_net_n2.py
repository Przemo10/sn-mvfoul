import torch
from torch import  nn
from src.utils import batch_tensor, unbatch_tensor
from src.custom_model.select_feature_extract_net import get_feature_network
from src.custom_model.pooling_attention import  select_pooling_attention


class XinMultimodalNetN20(torch.nn.Module):

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


class XinMultimodalNetN25(torch.nn.Module):

    # work based on article

    def __init__(self, num_views: int,  net_name='mvit_v2_s',freeze_layers = 0, agr_type = 'max'):
        super().__init__()

        network,  self.feat_dim, self.freeze_up_to  = get_feature_network(net_name=net_name)

        if freeze_layers > 0:
            self.model = self.freezee_net_layer(network)
        else:
            self.model = network
        self.model.head = nn.Identity()
        self.num_views = num_views
        self.feat_dim = 256
        self.token_dim = 768
        self.drop_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.token_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.2),
        )
        self.lifting_net = nn.Sequential()
        self.aggregation_model = select_pooling_attention(
            agr_type=agr_type,
            model=self.model,
            lifting_net=self.lifting_net,
            feat_dim=self.feat_dim
        )
        self.res_perceptor_block = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Sigmoid(),
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.2),
        )
        self.fc_offence = nn.Sequential(
            nn.Linear(self.feat_dim, 4)
        )
        self.fc_action = nn.Sequential(
            nn.Linear(self.feat_dim,  8)
        )
        self.intern = nn.Sequential(
            nn.Linear(self.feat_dim,  self.feat_dim),
            nn.ReLU(),
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.2)
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
            x = self.drop_block(aux)
            # print(x.shape)
            single_offence = self.fc_offence(x)
            single_action = self.fc_action(x)
            output_dict[f"single_{i}"] = {}
            output_dict[f"single_{i}"]['offence_logits'] = single_offence
            output_dict[f"single_{i}"]['action_logits'] = single_action
            x1 = self.res_perceptor_block(x)
            x = torch.add(x, x1)
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
