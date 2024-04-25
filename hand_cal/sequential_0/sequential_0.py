import torch
import torch.nn.functional
import numpy
from scipy.io import savemat
import struct
import os

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
    zero_channel_tensor = torch.zeros(1, 1, 160, 160, dtype=input_tensor.dtype)
    processed_input_tensor = torch.nn.functional.pad(
        torch.cat((input_tensor[:, :, :, :], zero_channel_tensor), dim=1),  # dim=1 is the channel dimension
        (1, 1, 1, 1),  # Padding format: (left, right, top, bottom)
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
    bias_npy = numpy.load(WEIGHT_PTH + 'first_conv_conv_bias_biases.npy')

    # Convert numpy arrays to PyTorch tensors
    input_tensor = torch.from_numpy(input_npy).bfloat16()
    weight_tensor = torch.from_numpy(weight_npy).bfloat16()
    bias_tensor = torch.from_numpy(bias_npy).bfloat16()

    # Create a convolutional layer
    first_conv = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True)
    relu6 = torch.nn.ReLU6()

    # Manually set the weights and biases of the convolutional layer
    first_conv.weight = torch.nn.Parameter(weight_tensor)
    first_conv.bias = torch.nn.Parameter(bias_tensor)

    # Map layers to cuda
    first_conv.to('cuda')
    relu6.to('cuda')

    # Apply the convolutional layers
    output = relu6(first_conv(input_tensor.to('cuda')))

    # Save the output tensor to a numpy file
    output_npy = output.detach().cpu().to(torch.float32).numpy()
    numpy.save(HALIDE_PTH + 'hw_output.npy', output_npy)
    print("output_npy shape: ", output_npy.shape)  # To verify the output shape

    # Load the gold standard file
    gold_output_npy = numpy.load(ACT_PTH + 'first_conv/first_conv.npy')

    # Compare the generated output with the gold standard
    difference = numpy.abs(output_npy - gold_output_npy)
    print("Difference between output and gold standard:", difference)
    print("Maximum difference:", numpy.max(difference))
    print("Mean difference:", numpy.mean(difference))

    # Check if the outputs are identical within a certain tolerance
    if numpy.allclose(output_npy, gold_output_npy, atol=1e-3):
        print("The outputs are effectively identical within the tolerance.\n")
    else:
        print("There are significant differences between the outputs.\n")

def float2bfbin(fnum):
    if fnum == "NaN":
        sign = "0"
        exp = "11111111"
        lfrac = "11111111"
    elif fnum == "-NaN":
        sign = "1"
        exp = "11111111"
        lfrac = "11111111"
    elif fnum == "Inf" or fnum > 3.402823466e+38:
        sign = "0"
        exp = "11111111"
        lfrac = "00000000"
    elif fnum == "-Inf" or fnum < -3.402823466e+38:
        sign = "1"
        exp = "11111111"
        lfrac = "00000000"
    else:
        fstr = "".join("{:08b}".format(elem) for elem in struct.pack("!f", fnum))
        sign = fstr[0]
        exp = fstr[1:9]
        lfrac = "0" + fstr[9:16]
        hfrac = fstr[16:]
        # Enable rounding
        if (hfrac[0] == "1" and (hfrac[1] == "1" or hfrac[2] == "1")) or (lfrac[7] == "1" and hfrac[0] == "1"):  
            # bit 8 of the float mantissa is set, so round up
            if lfrac[1:8] == "1111111":  # roll over mantissa and increase exp if needed
                exp = "{:08b}".format((int(exp, 2) + 1))  # exp overflow?
            lfrac = "{:08b}".format((int(lfrac, 2) + 1))
    return sign + exp + lfrac[1:8]

# Function to convert each element to bf16 stored as uint16
def convert_to_bf16_uint16(arr):
    vec_func = numpy.vectorize(float2bfbin)
    binary_strings = vec_func(arr)
    uint16_data = numpy.vectorize(lambda bin_str: numpy.uint16(int(bin_str, 2)))(binary_strings)
    return uint16_data.reshape(arr.shape)

def convert_array_to_mat(npy_arr, mat_name, HALIDE_PTH):
    print(f'{mat_name} has shape {npy_arr.shape}')
    bf16_npy_array = convert_to_bf16_uint16(npy_arr)
    mat_dict = {mat_name: bf16_npy_array}
    savemat(HALIDE_PTH + f'{mat_name}.mat', mat_dict)
    print(f'saving {HALIDE_PTH}{mat_name}.mat\n')

def save_mat(WORK_DIR, ACT_PTH, WEIGHT_PTH, HALIDE_PTH):
    # Load the Numpy array from the .npy file
    input_npy = numpy.squeeze(numpy.load(HALIDE_PTH + 'input_host_stencil.npy'), axis=0)
    kernel_npy = numpy.load(HALIDE_PTH + 'kernel_host_stencil.npy')
    bias_npy = numpy.load(WEIGHT_PTH + 'first_conv_conv_bias_biases.npy')
    output_npy = numpy.pad(
        numpy.squeeze(numpy.load(HALIDE_PTH + 'hw_output.npy'), axis=0),
        pad_width=((0, 0), (0, 0), (0, 0)), # ((dim0_left, dim0_right), (dim1_left, dim1_right), ...)
        mode='constant',
        constant_values=0
    )

    convert_array_to_mat(input_npy, 'input_host_stencil', HALIDE_PTH)
    convert_array_to_mat(kernel_npy, 'kernel_host_stencil', HALIDE_PTH)
    convert_array_to_mat(bias_npy, 'bias_host_stencil', HALIDE_PTH)
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