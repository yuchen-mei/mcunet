import numpy as np
from scipy.io import savemat
import struct

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
    vec_func = np.vectorize(float2bfbin)
    binary_strings = vec_func(arr)
    uint16_data = np.vectorize(lambda bin_str: np.uint16(int(bin_str, 2)))(binary_strings)
    return uint16_data.reshape(arr.shape)


# Load the Numpy array from the .npy file
input_npy = np.squeeze(np.load('padded_input_image.npy'), axis=0)
kernel_npy = np.load('padded_weight_numpy.npy')
bias_npy = np.load('first_conv_conv_bias_biases.npy')
output_npy = np.squeeze(np.load('first_conv_gold.npy'), axis=0)

input_name = 'input_host_stencil'
kernel_name = 'kernel_host_stencil'
bias_name = 'bias_host_stencil'
output_name = 'hw_output'

print("Input mat shape:", input_npy.shape)
print("Kernel mat shape:", kernel_npy.shape)
print("Bias mat shape:", bias_npy.shape)
print("Output mat shape:", output_npy.shape)

bf16_input = convert_to_bf16_uint16(input_npy)
bf16_kernel = convert_to_bf16_uint16(kernel_npy)
bf16_bias = convert_to_bf16_uint16(bias_npy)
bf16_output = convert_to_bf16_uint16(output_npy)

# Create a dictionary to store the data, where the key is the name of the variable in MATLAB
input_mat_dict = {input_name: bf16_input}
kernel_mat_dict = {kernel_name: bf16_kernel}
bias_mat_dict = {bias_name: bf16_bias}
output_mat_dict = {output_name: bf16_output}

print(f"{input_name}: ", bf16_input)
print(f"{kernel_name}: ", bf16_kernel)
print(f"{bias_name}: ", bf16_bias)
print(f"{output_name}: ", bf16_output)

# Save the dictionary to a .mat file
savemat(f'{input_name}.mat', input_mat_dict)
savemat(f'{kernel_name}.mat', kernel_mat_dict)
savemat(f'{bias_name}.mat', bias_mat_dict)
savemat(f'{output_name}.mat', output_mat_dict)