import torch
import torch.nn.functional as F
import numpy as np

# Load the data from .npy files
input_image = np.load('padded_input_image.npy')
conv_weight = np.load('padded_weight_numpy.npy')
conv_bias = np.load('first_conv_conv_bias_biases.npy')

# # Load the data from .npy files
# input_image = np.load('input_image_fp32.npy')
# conv_weight = np.load('first_conv_conv_weight_weights_fp32.npy')
# conv_bias = np.load('first_conv_conv_bias_biases_fp32.npy')

# Convert numpy arrays to PyTorch tensors
input_tensor = torch.from_numpy(input_image).bfloat16()
weight_tensor = torch.from_numpy(conv_weight).bfloat16()
bias_tensor = torch.from_numpy(conv_bias).bfloat16()

# # Convert numpy arrays to PyTorch tensors
# input_tensor = torch.from_numpy(input_image)
# weight_tensor = torch.from_numpy(conv_weight)
# bias_tensor = torch.from_numpy(conv_bias)

# Create a convolutional layer
# PyTorch expects weights in the shape (out_channels, in_channels, kernel_height, kernel_width)
# Numpy files are assumed to be in the shape (out_channels, in_channels, height, width) already
conv_layer = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True)

# Manually set the weights and biases of the convolutional layer
conv_layer.weight = torch.nn.Parameter(weight_tensor)
conv_layer.bias = torch.nn.Parameter(bias_tensor)

conv_layer.to('cuda')

# Apply the convolutional layer to the input tensor

output = conv_layer(input_tensor.to('cuda'))

# Save the output tensor to a numpy file
output_numpy = output.detach().cpu().to(torch.float32).numpy()
# output_numpy = output.detach().cpu().numpy()
np.save('output_image.npy', output_numpy)

print(output.shape)  # To verify the output shape
print(output)        # To see the output tensor

# Load the gold standard file
gold_output = np.load('first_conv_conv_gold.npy')
# gold_output = np.load('first_conv_conv_gold_fp32.npy')

# Compare the generated output with the gold standard
difference = np.abs(output_numpy - gold_output)
print("Difference between output and gold standard:", difference)
print("Maximum difference:", np.max(difference))
print("Mean difference:", np.mean(difference))

# Check if the outputs are identical within a certain tolerance
if np.allclose(output_numpy, gold_output, atol=1e-3):
    print("The outputs are effectively identical within the tolerance.")
else:
    print("There are significant differences between the outputs.")
