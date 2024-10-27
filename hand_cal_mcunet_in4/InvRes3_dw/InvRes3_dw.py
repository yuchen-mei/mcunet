import torch
import torch.nn.functional
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

INVRES_BLOCK_NAME = "blocks_2_mobile_inverted_conv_depth_conv"
PREVIOUS_LAYER = "InvRes3_pw_exp_bias_relu6"

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    input_npy = numpy.load(f'../{PREVIOUS_LAYER}/halide_data/hw_output.npy')
    weight_npy = numpy.load(WEIGHT_PTH + f'{INVRES_BLOCK_NAME}_conv_weight_weights.npy')

    # Convert numpy array to PyTorch tensor
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()

    # Nothing to preprocess for this layer
    processed_input_tensor = input_tensor
    processed_weight_tensor = weight_tensor

    # Convert the processed tensor back to a numpy array
    input_host_stencil = processed_input_tensor.to(torch.float32).numpy()
    kernel_host_stencil = processed_weight_tensor.to(torch.float32).numpy()

    # Save the processed input to a .npy file
    numpy.save(HALIDE_PTH + 'input_host_stencil.npy', input_host_stencil)
    print(f"input_host_stencil shape: {input_host_stencil.shape}")
    print(HALIDE_PTH + "input_host_stencil.npy saved\n")

    # Save the processed kernel to a .npy file
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
    dw_conv = torch.nn.Conv2d(in_channels=120, out_channels=120, kernel_size=3, stride=1, padding=0, groups=120, bias=False)

    # Manually set the weights and biases of the convolutional layer
    dw_conv.weight = torch.nn.Parameter(weight_tensor)

    # Map layers to cuda
    dw_conv.to('cuda')

    # Apply the convolutional layers
    output = dw_conv(input_tensor.to('cuda'))

    # Save the output npy file
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)
    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)
    print(HALIDE_PTH + "hw_output.npy saved\n")

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):

    # Load the Numpy array from the .npy file
    input_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'input_host_stencil.npy'), axis=0)
    kernel_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'kernel_host_stencil.npy'), axis=1)
    output_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'hw_output.npy'), axis=0)

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