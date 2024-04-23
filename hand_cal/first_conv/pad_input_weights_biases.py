import torch
import torch.nn.functional as F
import numpy as np

# Load the image from the .npy file
input_image = np.load('input_image.npy')
conv_weight = np.load('first_conv_conv_weight_weights.npy')
biases = np.load('first_conv_conv_bias_biases.npy')

# Convert numpy array to PyTorch tensor
original_input_tensor = torch.from_numpy(input_image).bfloat16()
original_weight_tensor = torch.from_numpy(conv_weight).bfloat16()
original_bias_tensor = torch.from_numpy(biases).bfloat16()

zero_channel = torch.zeros(1, 1, 160, 160, dtype=original_input_tensor.dtype)
input_tensor = torch.cat((original_input_tensor, zero_channel), dim=1)  # dim=1 is the channel dimension

zero_weights = torch.zeros((original_weight_tensor.shape[0], 1, original_weight_tensor.shape[2], original_weight_tensor.shape[3]), dtype=original_weight_tensor.dtype)
weight_tensor = torch.cat((original_weight_tensor, zero_weights), dim=1)

expanded_bias_tensor = original_bias_tensor[:, None, None]  # Adds two dimensions, now shape is (32, 1, 1)
expanded_bias_tensor = expanded_bias_tensor.expand(-1, 80, 80)  # Expands to (32, 80, 80)

# Apply the specified padding
# Padding format: (left, right, top, bottom)
padded_input = F.pad(input_tensor, (1, 0, 1, 0), mode='constant')

# Convert the padded tensor back to a numpy array
padded_input_numpy = padded_input.to(torch.float32).numpy()
padded_weight_numpy = weight_tensor.to(torch.float32).numpy()
expanded_bias_numpy = expanded_bias_tensor.to(torch.float32).numpy()

# Save the padded input to a .npy file
np.save('padded_input_image.npy', padded_input_numpy)
np.save('padded_weight_numpy.npy', padded_weight_numpy)
np.save('expanded_bias_numpy.npy', expanded_bias_numpy)