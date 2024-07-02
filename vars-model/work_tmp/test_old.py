import torch
import einops
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# Define input tensor shape: (batch_size, views, channels, depth, height, width)
# Example: batch_size=4, views=2, channels=3, depth=16, height=112, width=112
input_data = torch.randn(4, 2, 3, 16, 224, 224)

# Reshape the input tensor to match the model's expected input shape
reshaped_input = einops.rearrange(input_data, 'b n c d h w -> (b n) c d h w')

# Load the pre-trained model
weights_model = MViT_V2_S_Weights.DEFAULT
model = mvit_v2_s(weights=weights_model)

# Ensure the model is in evaluation mode
model.eval()

# Perform the forward pass
with torch.no_grad():  # Disable gradient calculation
    output = model(reshaped_input)

print(output.shape)