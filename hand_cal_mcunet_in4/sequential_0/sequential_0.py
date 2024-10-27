import torch
import torch.nn.functional
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    input_npy = numpy.load(ACT_PTH + 'input_image.npy')
    weight_npy = numpy.load(WEIGHT_PTH + 'first_conv_conv_weight_weights.npy')

    # Convert numpy array to PyTorch tensor
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()

    # Pad input tensor for Halide inputs
    zero_channel_tensor = torch.zeros(1, 1, input_tensor.shape[2], input_tensor.shape[3], dtype=input_tensor.dtype)
    processed_input_tensor = torch.nn.functional.pad(
        torch.cat((input_tensor[:, :, :, :], zero_channel_tensor), dim=1),  # dim=1 is the channel dimension
        # 160x160 -> 169x169
        (1, 8, 1, 8),  # Padding format: (left, right, top, bottom)
        mode='constant'
    )

    # Pad weight tensor for Halide inputs
    zero_weights_tensor = torch.zeros((weight_tensor.shape[0], 1, weight_tensor.shape[2], weight_tensor.shape[3]), dtype=weight_tensor.dtype)
    processed_weight_tensor = torch.cat((weight_tensor, zero_weights_tensor), dim=1)

    # Convert the padded tensor back to a numpy array
    input_host_stencil = processed_input_tensor.to(torch.float32).numpy()
    kernel_host_stencil = processed_weight_tensor.to(torch.float32).numpy()

    # Save the padded input to a .npy file
    numpy.save(HALIDE_PTH + 'input_host_stencil.npy', input_host_stencil)
    print(f"input_host_stencil shape: {input_host_stencil.shape}")
    print(HALIDE_PTH + "input_host_stencil.npy saved\n")

    # Save the padded kernel to a .npy file
    numpy.save(HALIDE_PTH + 'kernel_host_stencil.npy', kernel_host_stencil)
    print(f"kernel_host_stencil shape: {kernel_host_stencil.shape}")
    print(HALIDE_PTH + "kernel_host_stencil.npy saved\n")

def layer(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Load the data from .npy files
    input_npy = numpy.load(HALIDE_PTH + 'input_host_stencil.npy')
    weight_npy = numpy.load(HALIDE_PTH + 'kernel_host_stencil.npy')

    # Convert numpy arrays to PyTorch tensors
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()

    # Create a convolutional layer
    first_conv = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False)

    # Manually set the weights of the convolutional layer
    first_conv.weight = torch.nn.Parameter(weight_tensor)

    # Map layers to cuda
    first_conv.to('cuda')

    # Apply the convolutional layers
    output = first_conv(input_tensor.to('cuda'))

    # Save the output npy file
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)  # To verify the output shape
    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)
    print(HALIDE_PTH + "hw_output.npy saved\n")

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):

    # Load the Numpy array from the .npy file
    input_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'input_host_stencil.npy'), axis=0)
    kernel_npy = numpy.load(HALIDE_PTH + 'kernel_host_stencil.npy')
    output_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'hw_output.npy'), axis=0)

    # Save the Numpy array to a .mat file
    convert_array_to_mat(input_npy, 'input_host_stencil', HALIDE_PTH)
    convert_array_to_mat(kernel_npy, 'kernel_host_stencil', HALIDE_PTH)
    convert_array_to_mat(output_npy, 'hw_output', HALIDE_PTH)

def main():
    WORK_DIR = '/home/yuchenm/gitrepos/mcunet/'
    ACT_PTH = WORK_DIR + 'activations_dump/'
    WEIGHT_PTH = WORK_DIR + 'weights_biases_dump/'
    HALIDE_PTH = 'halide_data/'

    preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH)
    layer(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH)
    save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH)

if __name__ == '__main__':
    main()