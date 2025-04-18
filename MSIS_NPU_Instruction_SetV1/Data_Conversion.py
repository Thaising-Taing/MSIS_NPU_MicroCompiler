import numpy as np

CONCAT = 4

def uint32_to_hexa(uint32_data):
    if isinstance(uint32_data, np.ndarray):
        return np.vectorize(lambda x: hex(x)[2:].upper().zfill(8))(uint32_data)
    else:
        return hex(uint32_data)[2:].upper().zfill(8)

def uint16_to_hexa(uint16_data):
    if isinstance(uint16_data, np.ndarray):
        return np.vectorize(lambda x: hex(x)[2:].upper().zfill(4))(uint16_data)
    else:
        return hex(uint16_data)[2:].upper().zfill(4)

def uint8_to_hexa(uint8_data):
    if isinstance(uint8_data, np.ndarray):
        return np.vectorize(lambda x: hex(x)[2:].upper().zfill(2))(uint8_data)
    else:
        return hex(uint8_data)[2:].upper().zfill(2)


def float32_to_uint16(float32_data):
    return np.uint16(float32_data.view(np.uint32)>>16)

def integer_to_uint16(integer_data):
    return np.uint16(integer_data)

def integer_to_uint8(integer_data):
    return np.uint8(integer_data)

def Activation_SW2HW(batch, channel, height, width, data_ar, DEBUG=False, file_name=''):
    data_ar = data_ar.reshape(batch, channel, height, width)
    
    if(channel % CONCAT == 0):
        zero_channel = 0        
    else:
        zero_channel = CONCAT - (channel % CONCAT)
    channel = channel + zero_channel

    zero_ar = np.zeros(shape=(batch, zero_channel, height, width), dtype=np.uint8)
    data_ar = np.concatenate((data_ar, zero_ar), axis=1)
    data_ar = data_ar.reshape(batch, channel//CONCAT, CONCAT, -1).transpose(0,1,3,2).reshape(-1,CONCAT)
    data_ar = data_ar[:,::-1]

    if(DEBUG):
        data_save = data_ar[:,::-1]
        data_save = np.vectorize(uint8_to_hexa)(data_save)
        np.savetxt(file_name, data_save, fmt='%s', delimiter='')

    return data_ar


def Activation_HW2SW(batch, channel, height, width, data_ar, DEBUG=False, file_name=''):
    if(channel % CONCAT == 0):
        zero_channel = 0        
    else:
        zero_channel = CONCAT - (channel % CONCAT)

    channel_shape = channel + zero_channel

    # data_ar = data_ar.reshape(-1,CONCAT)

    data_ar = data_ar.reshape(batch, channel_shape//CONCAT, -1, CONCAT).transpose(0,1,3,2)
    data_ar = data_ar.reshape(batch, channel_shape, height, width)
    
    return data_ar[:,:channel,:,:]


def Weight_SW2HW(out_channel, in_channel, kernel_size, data_ar, DEBUG=False, file_name=''):
    data_ar = data_ar.reshape(out_channel, in_channel, kernel_size, kernel_size)
    
    if(in_channel != 1):
        if(kernel_size == 1 and CONCAT < 16):
            CONCAT_K1 = 16
        else:
            CONCAT_K1 = CONCAT
        
        if(in_channel % CONCAT_K1 == 0):
            zero_in_channel = 0        
        else:
            zero_in_channel = CONCAT_K1 - (in_channel % CONCAT_K1)
        in_channel = in_channel + zero_in_channel    
        
        zero_ar = np.zeros(shape=(out_channel, zero_in_channel, kernel_size, kernel_size), dtype=np.uint8)
        data_ar = np.concatenate((data_ar, zero_ar), axis=1)
    
    if(out_channel % CONCAT == 0):
        zero_out_channel = 0        
    else:
        zero_out_channel = CONCAT - (out_channel % CONCAT)  
    out_channel = out_channel + zero_out_channel
    
    zero_ar = np.zeros(shape=(zero_out_channel, in_channel, kernel_size, kernel_size), dtype=np.uint8)
    data_ar = np.concatenate((data_ar, zero_ar), axis=0)
    
    if(kernel_size == 1):
        if(in_channel != 1):
            zero_kernel_size = 1
        else:
            zero_kernel_size = 2
    elif(kernel_size == 2):
        zero_kernel_size = 1
    elif(kernel_size == 3):
        zero_kernel_size = 0
    
    if((kernel_size == 1) & (in_channel != 1)):
        zero_ar = np.zeros(shape=(out_channel, in_channel//2, zero_kernel_size, kernel_size), dtype=np.uint8)
        data_ar = data_ar.reshape(out_channel, in_channel//2, kernel_size*2, kernel_size)
        data_ar = np.concatenate((data_ar, zero_ar), axis=2)
    else:
        zero_ar = np.zeros(shape=(out_channel, in_channel, zero_kernel_size, kernel_size), dtype=np.uint8)
        data_ar = np.concatenate((data_ar, zero_ar), axis=2)
    
    data_ar = data_ar.reshape(out_channel//CONCAT, CONCAT, -1).transpose(0,2,1).reshape(-1,CONCAT)
    data_ar = data_ar[:,::-1]

    if(DEBUG):
        data_save = data_ar[:,::-1]
        data_save = np.vectorize(uint8_to_hexa)(data_save)
        np.savetxt(file_name, data_save, fmt='%s', delimiter='')

    return data_ar


def FCWeight_SW2HW(out_channel, in_channel, height, width, data_ar, DEBUG=False, file_name=''):
    
    data_ar = data_ar.reshape(out_channel, in_channel, height, width)
    data_ar = data_ar.reshape(out_channel, in_channel//4, 4, -1).transpose(0,1,3,2)
    
    kernel_size = 1
    in_channel = in_channel * height * width
    data_ar = data_ar.reshape(out_channel, in_channel, kernel_size, kernel_size)

    if(in_channel != 1):
        if(kernel_size == 1 and CONCAT < 16):
            CONCAT_K1 = 16
        else:
            CONCAT_K1 = CONCAT
        
        if(in_channel % CONCAT_K1 == 0):
            zero_in_channel = 0        
        else:
            zero_in_channel = CONCAT_K1 - (in_channel % CONCAT_K1)
        in_channel = in_channel + zero_in_channel    
        
        zero_ar = np.zeros(shape=(out_channel, zero_in_channel, kernel_size, kernel_size), dtype=np.uint8)
        data_ar = np.concatenate((data_ar, zero_ar), axis=1)
    
    if(out_channel % CONCAT == 0):
        zero_out_channel = 0        
    else:
        zero_out_channel = CONCAT - (out_channel % CONCAT)  
    out_channel = out_channel + zero_out_channel
    
    zero_ar = np.zeros(shape=(zero_out_channel, in_channel, kernel_size, kernel_size), dtype=np.uint8)
    data_ar = np.concatenate((data_ar, zero_ar), axis=0)
    
    if(kernel_size == 1):
        if(in_channel != 1):
            zero_kernel_size = 1
        else:
            zero_kernel_size = 2
    elif(kernel_size == 2):
        zero_kernel_size = 1
    elif(kernel_size == 3):
        zero_kernel_size = 0
    
    if((kernel_size == 1) & (in_channel != 1)):
        zero_ar = np.zeros(shape=(out_channel, in_channel//2, zero_kernel_size, kernel_size), dtype=np.uint8)
        data_ar = data_ar.reshape(out_channel, in_channel//2, kernel_size*2, kernel_size)
        data_ar = np.concatenate((data_ar, zero_ar), axis=2)
    else:
        zero_ar = np.zeros(shape=(out_channel, in_channel, zero_kernel_size, kernel_size), dtype=np.uint8)
        data_ar = np.concatenate((data_ar, zero_ar), axis=2)
    
    data_ar = data_ar.reshape(out_channel//CONCAT, CONCAT, -1).transpose(0,2,1).reshape(-1,CONCAT)
    data_ar = data_ar[:,::-1]

    if(DEBUG):
        data_save = data_ar[:,::-1]
        data_save = np.vectorize(uint8_to_hexa)(data_save)
        np.savetxt(file_name, data_save, fmt='%s', delimiter='')

    return data_ar


def Bias_SW2HW(channel, data_ar1, DEBUG=False, file_name=''):
    data_ar1 = data_ar1.reshape(channel)
    
    if(channel % CONCAT == 0):
        zero_channel = 0        
    else:
        zero_channel = CONCAT - (channel % CONCAT)
    
    zero_ar = np.zeros(shape=(zero_channel), dtype=np.uint16)
    data_ar1 = np.concatenate((data_ar1, zero_ar), axis=0)
    
    data_ar = data_ar1.reshape(-1, CONCAT//2)
    data_ar = data_ar[:,::-1]

    if(DEBUG):
        data_save = data_ar[:,::-1]
        data_save = np.vectorize(uint16_to_hexa)(data_save)
        np.savetxt(file_name, data_save, fmt='%s', delimiter='')
    
    return data_ar


def Parameter2_SW2HW(channel, data_ar1, data_ar2, DEBUG=False, file_name=''):
    data_ar1 = data_ar1.reshape(channel)
    data_ar2 = data_ar2.reshape(channel)
    
    if(channel % CONCAT == 0):
        zero_channel = 0        
    else:
        zero_channel = CONCAT - (channel % CONCAT)
    
    zero_ar = np.zeros(shape=(zero_channel), dtype=np.uint16)
    data_ar1 = np.concatenate((data_ar1, zero_ar), axis=0)
    data_ar2 = np.concatenate((data_ar2, zero_ar), axis=0)
    
    data_ar1 = data_ar1.reshape(-1, CONCAT)
    data_ar2 = data_ar2.reshape(-1, CONCAT)
    data_ar = np.concatenate((data_ar1, data_ar2), axis=1).reshape(-1, CONCAT//2)
    data_ar = data_ar[:,::-1]

    if(DEBUG):
        data_save = data_ar[:,::-1]
        data_save = np.vectorize(uint16_to_hexa)(data_save)
        np.savetxt(file_name, data_save, fmt='%s', delimiter='')

    return data_ar


def hex_file_to_numpy_array(file_path):
    with open(file_path, 'r') as f:
        hex_strings = f.read().splitlines()
    
    # 헥사 스트링을 32비트 정수로 변환
    uint32_array = np.array([int(hex_str, 16) for hex_str in hex_strings], dtype=np.uint32)
    
    return uint32_array


def hex_file_to_list(file_path):
    with open(file_path, 'r') as f:
        uint32_list = [int(line.strip(), 16) for line in f.readlines()]
    return uint32_list

