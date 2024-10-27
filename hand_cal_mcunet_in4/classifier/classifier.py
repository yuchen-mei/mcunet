import torch
import torch.nn.functional
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

PREVIOUS_LAYER = 'avgpool'
BLOCK_NAME = "classifier_linear"

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    input_npy = numpy.load(WEIGHT_PTH + f"{BLOCK_NAME}_weight_weights.npy") # (1000, 320)
    weight_npy = numpy.load(f"../{PREVIOUS_LAYER}/halide_data/hw_output.npy") # (1, 320)

    # Convert numpy array to PyTorch tensor in bfloat16
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()

    # Reshape the input tensor
    processed_input_tensor = input_tensor.transpose(0, 1)
    processed_weight_tensor = weight_tensor

    # Convert the processed tensor back to a numpy array
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

    # Create a matrix multiplication layer
    output = torch.matmul(input_tensor.transpose(0, 1), weight_tensor.transpose(0, 1))

    # Save the output npy file
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)
    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)
    print(HALIDE_PTH + "hw_output.npy saved\n")

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):

    # Load the Numpy array from the .npy file
    input_npy = numpy.load(HALIDE_PTH + 'input_host_stencil.npy')
    kernel_npy = numpy.load(HALIDE_PTH + 'kernel_host_stencil.npy')
    output_npy = numpy.load(HALIDE_PTH + 'hw_output.npy')

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