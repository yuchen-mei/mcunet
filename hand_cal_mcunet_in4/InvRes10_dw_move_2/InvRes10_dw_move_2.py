import torch
import torch.nn.functional as F
import numpy
from scipy.io import savemat
import struct
import os
import sys
sys.path.append('..')
from utils import *

PAD_O_LEFT = 0
PAD_O_RIGHT = 0
TRUNC_SIZE = 0
PREVIOUS_LAYER = "InvRes10_pw_exp_bias_relu6"

def preprocess(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Clean halide path first and recreate it
    if os.path.exists(HALIDE_PTH): 
        print(f"cleaning {HALIDE_PTH}\n")
        os.system('rm -rf ' + HALIDE_PTH)
    os.mkdir(HALIDE_PTH)

    # Load the image from the .npy file
    input_npy = numpy.load(f"../{PREVIOUS_LAYER}/halide_data/hw_output.npy")
    input_npy_glb0 = input_npy[:, 0::4, :, :]
    input_npy_glb1 = input_npy[:, 1::4, :, :]
    input_npy_glb2 = input_npy[:, 2::4, :, :]
    input_npy_glb3 = input_npy[:, 3::4, :, :]
    input_npy = numpy.zeros((input_npy_glb2.shape[0], input_npy_glb2.shape[1]+input_npy_glb3.shape[1], input_npy_glb2.shape[2], input_npy_glb2.shape[3]))
    input_npy[:, 0::2, :, :] = input_npy_glb2[:, :, :, :]
    input_npy[:, 1::2, :, :] = input_npy_glb3[:, :, :, :]

    # Convert numpy array to PyTorch tensor in bfloat16
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    
    # Expand the bias tensor to match the input tensor shape
    processed_input_tensor = input_tensor

    # Convert the processed tensor back to a numpy array
    hw_input_stencil = processed_input_tensor.to(torch.float32).numpy()

    # Save the processed input to a .npy file
    numpy.save(HALIDE_PTH + 'hw_input_stencil.npy', hw_input_stencil)
    print(f"hw_input_stencil shape: {hw_input_stencil.shape}")
    print(HALIDE_PTH + "hw_input_stencil.npy saved\n")

def layer(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    
    # Load the data from .npy files
    input_npy = numpy.load(HALIDE_PTH + 'hw_input_stencil.npy')

    # Convert numpy arrays to PyTorch tensors and map to cuda
    input_tensor = torch.from_numpy(input_npy).bfloat16().to('cuda')

    # Create layer and define output padding and trunc size
    pad_o_left = PAD_O_LEFT
    pad_o_right = PAD_O_RIGHT
    padding = (pad_o_left, pad_o_right, pad_o_left, pad_o_right)
    trunc_size = TRUNC_SIZE

    # Apply the bias layer
    output = input_tensor

    # Pad the output
    output_padded = F.pad(output, padding)

    # Write zeros to last few rows and cols
    if trunc_size > 0:
        output_padded[..., -trunc_size:, :] = 0
        output_padded[..., :, -trunc_size:] = 0
    
    # Load the output as npy for comparison
    output_npy = output_padded.detach().cpu().to(torch.float32).numpy()
    print("output_npy shape: ", output_npy.shape)  # To verify the output shape
    
    # Save the padded gold output to a .npy file
    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)
    print(HALIDE_PTH + "hw_output.npy saved\n")

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Load the Numpy array from the .npy file
    input_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'hw_input_stencil.npy'), axis=0)
    output_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'hw_output.npy'), axis=0)

    # Save the data to a .mat file
    convert_array_to_mat(input_npy, 'hw_input_stencil', HALIDE_PTH)
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