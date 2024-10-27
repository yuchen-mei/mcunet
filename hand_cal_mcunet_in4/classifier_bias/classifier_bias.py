import torch
import torch.nn.functional as F
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

INVRES_BLOCK_NAME = 'classifier_linear'
PREVIOUS_LAYER = 'classifier'

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    input_npy = numpy.load(f"../{PREVIOUS_LAYER}/halide_data/hw_output.npy")
    bias_npy = numpy.load(WEIGHT_PTH + f"{INVRES_BLOCK_NAME}_bias_biases.npy")

    # Convert numpy array to PyTorch tensor in bfloat16
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    bias_tensor = torch.from_numpy(bias_npy).bfloat16()
    
    # Don't need preprocess
    processed_input_tensor = input_tensor.squeeze(1)
    processed_bias_tensor = bias_tensor

    # Convert the processed tensor back to a numpy array
    hw_input_stencil = processed_input_tensor.to(torch.float32).numpy()
    hw_bias_stencil = processed_bias_tensor.to(torch.float32).numpy()

    # Save the processed input to a .npy file
    numpy.save(HALIDE_PTH + 'hw_input_stencil.npy', hw_input_stencil)
    print(f"hw_input_stencil shape: {hw_input_stencil.shape}")
    print(HALIDE_PTH + "hw_input_stencil.npy saved\n")

    # Save the processed bias to a .npy file
    numpy.save(HALIDE_PTH + 'hw_bias_stencil.npy', hw_bias_stencil)
    print(f"hw_bias_stencil shape: {hw_bias_stencil.shape}")
    print(HALIDE_PTH + "hw_bias_stencil.npy saved\n")

def layer(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    
    # Load the data from .npy files
    input_npy = numpy.load(HALIDE_PTH + 'hw_input_stencil.npy')
    bias_npy = numpy.load(HALIDE_PTH + 'hw_bias_stencil.npy')

    # Convert numpy arrays to PyTorch tensors and map to cuda
    input_tensor = torch.from_numpy(input_npy).bfloat16().to('cuda')
    bias_tensor = torch.from_numpy(bias_npy).bfloat16().to('cuda')

    # Apply the bias layer
    output = input_tensor + bias_tensor
    
    # Load the output as npy for comparison
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)  # To verify the output shape

    # Load the gold standard file
    gold_output_npy = numpy.load(ACT_PTH + f"classifier/classifier_out.npy")

    # Set the new relative tolerance slightly higher than the default
    atol = 4.0e-02
    rtol = 5.0e-02

    # Calculate the combined tolerance using the formula from numpy.allclose
    difference = numpy.abs(output_npy - gold_output_npy)
    tolerance = atol + rtol * numpy.abs(gold_output_npy)

    # Find locations where the absolute difference exceeds the calculated tolerance
    out_of_tolerance = numpy.where(difference > tolerance)

    # If there are values exceeding the tolerance
    if out_of_tolerance[0].size > 0:
        print("\n[FAIL] There are significant differences between the outputs.\n")

        # Get indices of the maximum absolute and relative differences
        max_adiff_idx = numpy.unravel_index(numpy.argmax(difference), difference.shape)

        # Print the indices, test, and gold values for largest adiff
        adiff = difference[max_adiff_idx]
        print(f"Largest absolute difference at index {max_adiff_idx}:")
        print(f"Test output: {output_npy[max_adiff_idx]}, Gold output: {gold_output_npy[max_adiff_idx]}, adiff={adiff}")

    else:
        print("\n[PASS] The outputs are effectively identical within the tolerance.\n")
    
    # Save the padded gold output to a .npy file
    numpy.save(HALIDE_PTH + 'hw_output.npy', gold_output_npy)

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Load the Numpy array from the .npy file
    input_npy = numpy.load(HALIDE_PTH + 'hw_input_stencil.npy')
    bias_npy = numpy.load(HALIDE_PTH + 'hw_bias_stencil.npy')
    output_npy = numpy.load(HALIDE_PTH + 'hw_output.npy')

    # Save the data to a .mat file
    convert_array_to_mat(input_npy, 'hw_input_stencil', HALIDE_PTH)
    convert_array_to_mat(bias_npy, 'hw_bias_stencil', HALIDE_PTH)
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