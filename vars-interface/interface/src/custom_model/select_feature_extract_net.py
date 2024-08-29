from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.video import swin3d_s, Swin3D_S_Weights, swin3d_t, Swin3D_T_Weights


FREEZE_LAYERS = {
    "mvit_v2_s": "blocks.10",
    "swin3d_s": "features.4",
    "swin3d_t": "features.4",
}


def get_feature_network(net_name='r2plus1d_18', feat_dim = 400):
    if net_name == "r3d_18":
        weights_model = R3D_18_Weights.DEFAULT
        network = r3d_18(weights=weights_model)

    elif net_name == "s3d":
        weights_model = S3D_Weights.DEFAULT
        network = s3d(weights=weights_model)
        feat_dim = 400
    elif net_name == "mc3_18":
        weights_model = MC3_18_Weights.DEFAULT
        network = mc3_18(weights=weights_model)
    elif net_name == "r2plus1d_18":
        weights_model = R2Plus1D_18_Weights.DEFAULT
        network = r2plus1d_18(weights=weights_model)
    elif net_name == "mvit_v2_s":
        weights_model = MViT_V2_S_Weights.DEFAULT
        network = mvit_v2_s(weights=weights_model)
        feat_dim = 400
    elif net_name == "swin3d_s":
        weights_model = Swin3D_S_Weights.DEFAULT
        network = swin3d_s(weights=weights_model)
        feat_dim = 400
    elif net_name == "swin3d_t":
        weights_model = Swin3D_T_Weights.DEFAULT
        network = swin3d_t(weights=weights_model)
        feat_dim = 400
    else:
        weights_model = R2Plus1D_18_Weights.DEFAULT
        network = r2plus1d_18(weights=weights_model)

    freeze_layer_name = FREEZE_LAYERS.get(net_name, "no_freeze")

    return  network, feat_dim, freeze_layer_name
