import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from hybrid_model.mutulal_distilation_loss import MutualDistillationLoss


class MultiImageHybrid(nn.Module):

    def __init__(self, arch, num_classes, n, pretrained_weights=True):

        super().__init__()

        self.n = n
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        drop_rate = .0 if 'tiny' in arch else .1
        self.model = timm.create_model(arch, pretrained=self.pretrained_weights, num_classes=self.num_classes,
                                       drop_rate=drop_rate)
        for block in self.model.blocks:
            block.attn.fused_attn = False

        self.embed_dim = self.model.embed_dim

        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)

        # nn.init.zeros_(self.model.head.weight)
        # nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image):

        x = einops.rearrange(x, '(b n) s c -> b (n s) c', b=batch_size, n=self.n)
        print(f"After rearrange in format_multi_image_tokens: {x.shape}")

        first_img_token_idx = 0
        if self.model.cls_token is not None:
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_image + 1
                print(f"Excess CLS token index: {excess_cls_index}")
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
                print(f"After removing excess CLS token at step {i}: {x.shape}")
            first_img_token_idx = 1

        print(f"Image embeddings shape before normalization: {self.img_embed_matrix.shape}")
        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        print(f"Image embeddings shape after normalization: {image_embeddings.shape}")

        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        print(f"After adding image embeddings: {x.shape}")
        return x

    def forward(self, x):

        batch_size = len(x)
        output_dict = {'single': {}}
        if self.n > 1:
            output_dict['mv_collection'] = {}

        print(f"Input shape: {x.shape}")
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        print(f"After rearrange in forward: {x.shape}")

        x = self.model.patch_embed(x)
        print(f"After patch embedding: {x.shape}")

        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)
        print(f"After positional embedding: {x.shape}")

        for view_type in output_dict:

            tokens = x.clone()
            if view_type == 'mv_collection':
                tokens = self.format_multi_image_tokens(tokens, batch_size, tokens_per_image)
                print(f"After format_multi_image_tokens: {tokens.shape}")
            tokens = self.model.blocks(tokens)
            print(f"After blocks: {tokens.shape}")

            tokens = self.model.norm(tokens)
            print(f"After normalization: {tokens.shape}")

            output_dict[view_type]['logits'] = self.model.forward_head(tokens)
            print(f"Logits shape: {output_dict[view_type]['logits'].shape}")

        return output_dict


# Example usage:
if __name__ == "__main__":
    # Assuming an architecture name 'vit_base_patch16_224' and 1000 classes
    model = MultiImageHybrid(arch='vit_base_patch16_224', num_classes=8, n=4)

    for x1, x2 in model.named_parameters():
        print(x1, x2.shape)

    # Create a batch of 3 sets of images, each set containing 4 images of size 224x224 with 3 channels
    input_tensor = torch.randn(3, 4, 3, 224, 224)

    # Forward pass
    output = model(input_tensor)
    print("-----------------")
    print(f" Single logits shape {output['single']['logits'].shape}")
    print(f" MV logits shape {output['mv_collection']['logits'].shape}")
    print( output['mv_collection']['logits'][:, 0].shape )
    print(output['single']['logits'].flatten().shape)
    #print(torch.argmax(output['single']['logits'], dim=1))
    # print(torch.argmax(output['mv_collection']['logits'], dim=1))
    single_view_logits = einops.rearrange(output['single']['logits'], '(b n) k -> b n k', b=3, n=4)
    print('*******')
    print(single_view_logits.shape)
    print(single_view_logits.mean(dim=1).shape)
    #print(single_view_logits)
    #print(single_view_logits.flatten())

    # Your tensor
    tensor = torch.tensor([
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.]
    ])

    # Repeat the tensor 3 times along the first dimension
    repeated_tensor = tensor.repeat_interleave(3, dim=0).view(4, 3, 8)
    print(tensor.shape,  repeated_tensor.shape)
    print(repeated_tensor.flatten())

    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    print(loss(repeated_tensor.flatten(), single_view_logits.flatten())/4.0)

    mdloss = MutualDistillationLoss()

    print(output['mv_collection']['logits'][:, 0])
    print(torch.mean(output['mv_collection']['logits'], dim=1))
    t2 = mdloss(output['mv_collection']['logits'],single_view_logits, output['mv_collection']['logits'][1:, 0])
    print(t2  )
    print(output['mv_collection']['logits'])

