import numpy
import struct
from scipy.io import savemat
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