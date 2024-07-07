import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class MultiVideoHybridMVit2(nn.Module):
    """
    A hybrid model for handling multiple frames per video using MVit_v2_s.

    Args:
        n (int): Number of views per sample.
        pretrained_weights (str): Path to the pretrained weights.
    """

    def __init__(self, num_views: int, pretrained_weights=None):
        super(MultiVideoHybridMVit2, self).__init__()

        self.n = num_views

        # Initialize the base MVit_v2_s model
        weights = MViT_V2_S_Weights.DEFAULT if pretrained_weights is None else pretrained_weights
        self.model = mvit_v2_s(weights=weights)
        self.embed_dim = 96
        self.feet_dim = 400
        self.org_dim = self.model.head[1].in_features

        # Initialize the learnable image embedding matrix
        self.img_embed_matrix = nn.Parameter(torch.zeros(1, self.n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)

        # Reuse the model head layer for classification tasks
        self.tmp_head = self.model.head[1]

        self.offence_head = self.tmp_head
        self.action_head = self.tmp_head

        # Initialize the classification head
        # self.fc_offence = nn.Linear(self.feet_dim, out_features=4)
        # self.fc_action = nn.Linear(self.feet_dim, out_features=8)

        self.fc_offence = nn.Sequential(
            nn.Identity(),
            nn.Dropout(p=0.1),
            nn.Linear(768, 4)
        )
        self.fc_action = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 8)
        )

    def freeze_blocks(self, freeze: bool):
        for param in self.model.blocks.parameters():
            param.requires_grad = freeze

    def format_multi_frame_tokens(self, x, batch_size, tokens_per_frame):
        """
        Formats the tokens for multiple frames.

        Args:
            x (Tensor): Input tensor with shape [batch_size * n, tokens, embed_dim].
            batch_size (int): Original batch size.
            tokens_per_frame (int): Number of tokens per frame.

        Returns:
            Tensor: Formatted tensor.
        """
        # Initial shape of x: [batch_size * n, tokens, embed_dim]
        # Example initial shape: [8, 18817, 96]
        # print(f"Initial shape of x: {x.shape}")

        # Rearrange the tensor to merge batch and frame dimensions
        x = einops.rearrange(x, '(b n) s c -> b (n s) c', b=batch_size, n=self.n)
        # Shape after rearrange: [4, 37634, 96] if n=2 (concatenating tokens for all frames per batch)
        # print(f"Shape after rearrange: {x.shape}")

        first_img_token_idx = 0

        # Handle cls_token if present
        if hasattr(self.model.pos_encoding, 'class_token'):
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_frame + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
                # print(f"Shape after removing cls token at frame {i}: {x.shape}")
            first_img_token_idx = 1
            # Shape after removing excess cls tokens: [4, 37633, 96] if n=2 and cls tokens are removed

        # Normalize and add image embeddings
        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        # image_embeddings shape: [1, 2, 96] if n=2
        #print(f"Image embeddings shape: {image_embeddings.shape}")

        # Repeat embeddings to match the number of tokens per frame
        repeated_embeddings = torch.repeat_interleave(image_embeddings, tokens_per_frame - first_img_token_idx, dim=1)
        # print(f"Repeated embeddings shape: {repeated_embeddings.shape}")

        x[:, first_img_token_idx:] += repeated_embeddings
        # Shape after adding image embeddings: [4, 37633, 96] (no change in shape, just adding embeddings)
        # print(f"Shape after adding image embeddings: {x.shape}")

        return x

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape [batch_size, n, channels, depth, height, width].

        Returns:
            dict: A dictionary with logits for single and multi-view collections.
        """
        batch_size = x.shape[0]
        output_dict = {'single': {}}
        if self.n > 1:
            output_dict['mv_collection'] = {}

        # Input shape: [batch_size, num_views, channels, depth, height, width]
        # Example input shape: [4, 2, 3, 11, 224, 224]

        # Flatten the views into individual images
        x = einops.rearrange(x, 'b n c d h w -> (b n) c d h w')
        # Shape after rearrange: [8, 3, 11, 224, 224]
        # print(f"Shape after rearrange (views to individual images): {x.shape}")

        # Pass through the initial convolutional layers of MVIT to get patch embeddings
        x = self.model.conv_proj(x)
        # Shape after conv_proj: [8, 96, 6, 56, 56]
        # print(f"Shape after conv_proj: {x.shape}")

        # Get the shape for temporal, height, and width dimensions
        init_thw_shape = x.shape[2:]  # Shape: [6, 56, 56]
        thw_shape = init_thw_shape
        # print(f"THW shape after conv_proj: {thw_shape}")

        # Flatten the spatial dimensions and bring channels to the last dimension
        B, C, D, H, W = x.shape
        # print(B, C, D, H, W)
        x = x.view(B, C, D * H * W).transpose(1, 2)  # Now x has shape [batch_size, num_tokens, embed_dim]
        # Shape after view and transpose: [8, 18816, 96]
        # print(f"Shape after view and transpose: {x.shape}")

        # Add positional encoding
        x = self.model.pos_encoding(x)
        # print(f"Shape after adding positional encoding: {x.shape}")

        tokens_per_frame = x.shape[1]  # Number of tokens per frame

        for view_type in output_dict:
            tokens = x.clone()
            if view_type == 'mv_collection':
                tokens = self.format_multi_frame_tokens(tokens, batch_size, tokens_per_frame)
                # Shape after format_multi_frame_tokens: [4, 37632, 96] if n=2
                # print(f"Shape after format_multi_frame_tokens: {tokens.shape}")
                # Update thw_shape after merging frames
                thw_shape = (D * self.n, H, W)
                # print(f"Updated thw_shape for mv_collection: {thw_shape}")
                self.freeze_blocks(freeze=False)
            else:
                self.freeze_blocks(freeze=False)

            # Sequentially pass the tokens through each block with the thw argument
            for block in self.model.blocks:
                tokens, thw_shape = block(tokens, thw_shape)
                # print(tokens.shape, thw_shape)  # Debug print statement for shape tracking
                # Shape after each block will be [batch_size, num_tokens, embed_dim]

            tokens = self.model.norm(tokens)
            # Shape after normalization: [4, 295, 768] if final number of tokens is 295 and embed_dim is 768
            # print(f"Shape {view_type} tokens after normalization: {tokens.shape}")

            selected_tokens = torch.mean(tokens, dim=1)  # torch.max(tokens, dim=1)[0] #tokens[:, 0] # TO DO MIX

            offence_logits = self.fc_offence(selected_tokens)
            action_logits = self.fc_action(selected_tokens)
            # Shape of logits: [batch_size, num_classes], e.g., [4, 10]
            # print(f"Offence logits shape: {offence_logits.shape}")
            # print(f"Action logits shape: {action_logits.shape}")

            output_dict[view_type]['offence_logits'] = offence_logits
            output_dict[view_type]['action_logits'] = action_logits

        self.freeze_blocks(freeze=False)

        return output_dict

# Usage example:
# Initialize the model
# model = MultiVideoHybridMVit2(num_views=2)
# Example input: [batch_size, num_views, channels, depth, height, width]
#videos = torch.randn(4, 2, 3, 16, 224, 224)
"""
 (output_dict['single']['offence_logits'],output_dict['single']['action_logits'], 
                output_dict['mv_collection']['offence_logits'], output_dict['mv_collection']['action_logits'])
"""

#print(list(output.keys()))
#print(list(output['single'].keys()))
#print(output)
# model(videos)
