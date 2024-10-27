import torch
import torch.nn.functional
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

PREVIOUS_LAYER = 'InvRes17_pw_sq_bias'
INPUT_CHANNEL_SIZE = 4

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):

    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    # Load the image from the .npy file
    input_npy = numpy.load(f"../{PREVIOUS_LAYER}/halide_data/hw_output.npy")

    # Select the first 8 channels out of 320
    input_npy = input_npy[:, :INPUT_CHANNEL_SIZE, :, :]

    # Convert numpy array to PyTorch tensor in bfloat16
    input_tensor = torch.from_numpy(input_npy).bfloat16()

    # Nothing to preprocess for this layer
    processed_input_tensor = input_tensor

    # Convert the processed tensor back to a numpy array
    input_host_stencil = processed_input_tensor.to(torch.float32).numpy()

    # Save the padded input to a .npy file
    numpy.save(HALIDE_PTH + 'input_host_stencil.npy', input_host_stencil)
    print(f"input_host_stencil shape: {input_host_stencil.shape}")
    print(HALIDE_PTH + "input_host_stencil.npy saved\n")

def layer(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
   
   # Load the data from .npy files
    input_npy = numpy.load(HALIDE_PTH + 'input_host_stencil.npy')
    weight_npy = numpy.full((INPUT_CHANNEL_SIZE, 1, input_npy.shape[2], input_npy.shape[3]), 1/25)

    # Convert numpy arrays to PyTorch tensors
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()

    # Create an avgpooling layer using depthwise conv schedule
    avg_pool = torch.nn.Conv2d(in_channels=INPUT_CHANNEL_SIZE, out_channels=INPUT_CHANNEL_SIZE, kernel_size=11, stride=1, padding=0, groups=INPUT_CHANNEL_SIZE, bias=False)

    # Manually set the weights
    avg_pool.weight = torch.nn.Parameter(weight_tensor)
    
    # Map layers to cuda
    avg_pool.to('cuda')

    # Apply the convolutional layers
    output = avg_pool(input_tensor.to('cuda'))
    output = output.squeeze(2).squeeze(2) # Remove the spatial dimensions since they are all 1

    # Save the output npy file
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)

    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):

    # Load the Numpy array from the .npy file
    input_npy = numpy.load(HALIDE_PTH + 'input_host_stencil.npy')
    output_npy = numpy.load(HALIDE_PTH + 'hw_output.npy')

    convert_array_to_mat(input_npy, 'input_host_stencil', HALIDE_PTH)
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