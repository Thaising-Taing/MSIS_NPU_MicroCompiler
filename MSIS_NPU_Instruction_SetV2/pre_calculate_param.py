class layer_address:
    def __init__(self, weight_adres, IN1_adres, IN2_adres, Param_adres, out_adres):
        self.weight_adres = weight_adres
        self.IN1_adres = IN1_adres
        self.IN2_adres = IN2_adres
        self.Param_adres = Param_adres
        self.out_adres = out_adres

class LayerParams:
    def __init__(self, total_width, total_height, tile_width, tile_height, 
                 total_in_ch, total_out_ch, tile_in_ch, tile_out_ch,
                 kernel_size, stride, l_pad=0, r_pad=0, t_pad=0, b_pad=0, maxpool_stride=0, C_CONCAT=2,
                 post_process_cfg=None, pad_cfg=None):
        self.total_width = total_width
        self.total_height = total_height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.total_in_ch = total_in_ch
        self.total_out_ch = total_out_ch
        self.tile_in_ch = tile_in_ch
        self.tile_out_ch = tile_out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.l_pad = l_pad
        self.r_pad = r_pad
        self.t_pad = t_pad
        self.b_pad = b_pad
        self.maxpool_stride = maxpool_stride
        self.C_CONCAT = C_CONCAT
        self.post_process_cfg = post_process_cfg or {'prcs1': '', 'prcs2': '', 'prcs3': '', 'prcs4': ''}
        self.pad_cfg = pad_cfg or {'l_ovlp': 0, 't_ovlp': 0}
        
        # Calculate kernel_2d_size with padding
        extra_kernel_size = 0 if kernel_size % 3 == 0 else 3 - (kernel_size % 3)
        self.kernel_2d_size = kernel_size * (kernel_size + extra_kernel_size)
        
        # Calculate total_2d_size
        self.total_2d_size = total_width * total_height

def calculate_layer_params(params, operation):
    """
    Calculate all layer parameters based on operation type
    
    Args:
        params (LayerParams): Layer parameters
        operation (str): Operation type ('D2_CONV', 'DW_CONV', 'LINEAR', 'RESIZE', 'MATMUL', 'EWADDER')
    
    Returns:
        dict: Dictionary containing all calculated parameters
    """
    # Calculate write_out_ch and read_out_ch
    write_out_ch = (params.tile_out_ch >> params.C_CONCAT) + (params.tile_out_ch & ((1 << params.C_CONCAT) - 1))
    read_out_ch = (params.tile_out_ch >> params.C_CONCAT) + (params.tile_out_ch & ((1 << params.C_CONCAT) - 1))
    
    # Calculate read_in_ch
    read_in_ch = (params.tile_in_ch >> params.C_CONCAT) + (params.tile_in_ch & ((1 << params.C_CONCAT) - 1))
    
    # Calculate mcm_dataflow flag based on operation type
    if operation == 'D2_CONV':
        if params.total_in_ch != params.tile_in_ch:
            mcm_dataflow = 0
        else:
            if params.total_out_ch <= params.tile_out_ch:
                mcm_dataflow = 2
            elif params.total_out_ch == (params.tile_out_ch << 1):
                mcm_dataflow = 3
            else:
                mcm_dataflow = 1
    elif operation in ['DW_CONV', 'LINEAR']:
        mcm_dataflow = 2
    elif operation in ['MATMUL']:     
        mcm_dataflow = 0  
    else:
        mcm_dataflow = 0
    
    # Calculate operation flag (DW_CONV=01, MATMUL=10)
    operation_flag = 0
    if operation == 'DW_CONV':
        operation_flag = 0b01
    elif operation == 'MATMUL':
        operation_flag = 0b10
    elif operation == 'EWADDER':
        operation_flag = 0b10
    # Combine mcm_dataflow (MSB 16 bits) and operation_flag (LSB 16 bits)
    combined_flow_flag = (mcm_dataflow << 16) | operation_flag
    mcm_flag = f"0x{combined_flow_flag:08x}"    
    # Calculate ADDER/MUL flag (MSB 16 bits)
    adder_mul_flag = 0
    has_adder = False
    has_mul = False
    
    # Check for ADDER
    if params.post_process_cfg['prcs1'] == 'ADDER' or params.post_process_cfg['prcs2'] == 'ADDER' or \
       params.post_process_cfg['prcs3'] == 'ADDER' or params.post_process_cfg['prcs4'] == 'ADDER':
        has_adder = True
    
    # Check for MUL
    if params.post_process_cfg['prcs1'] == 'MUL' or params.post_process_cfg['prcs2'] == 'MUL' or \
       params.post_process_cfg['prcs3'] == 'MUL' or params.post_process_cfg['prcs4'] == 'MUL':
        has_mul = True
    
    # Set flag based on presence of ADDER and MUL
    if has_adder and has_mul:
        adder_mul_flag = 0b10  # Both present
    elif has_adder or has_mul:
        adder_mul_flag = 0b01  # Either one present
    else:
        adder_mul_flag = 0b00  # Neither present
    
    # Calculate overlap flag (LSB 16 bits)
    overlap_flag = 0
    if params.pad_cfg['l_ovlp'] == 1:
        overlap_flag |= 0b0000000000000001  # Set bit 0 for l_ovlp
    if params.pad_cfg['t_ovlp'] == 1:
        overlap_flag |= 0b0000001000000000  # Set bit 9 for t_ovlp
    
    # Combine flags into 32-bit value
    # Shift ADDER/MUL flag to MSB 16 bits (bits 16-31)
    # Keep overlap flag in LSB 16 bits (bits 0-15)
    combined_flags = (adder_mul_flag << 16) | overlap_flag
    
    # Convert to hex string
    combined_flags_hex = f"0x{combined_flags:08x}"
    
    # Calculate burst parameters 
    if params.total_width == params.tile_width and params.total_height == params.tile_height:
        if operation == 'MATMUL':
            burst_num_write = 16
        else:
            burst_num_write = 1
    elif params.total_width == params.tile_width:
        burst_num_write = write_out_ch
    else:
        burst_num_write = write_out_ch * params.tile_height
    
    # Calculate total_out_width and total_out_height based on operation and stride
    if operation in ['D2_CONV', 'DW_CONV']:
        if params.stride == 2:
            total_out_width = ((params.total_width + params.l_pad + params.r_pad - params.kernel_size) >> 1) + 1
            total_out_height = ((params.total_height + params.t_pad + params.b_pad - params.kernel_size) >> 1) + 1
        else:
            total_out_width = (params.total_width + params.l_pad + params.r_pad - params.kernel_size) + 1
            total_out_height = (params.total_height + params.t_pad + params.b_pad - params.kernel_size) + 1
    elif operation == 'LINEAR':
        total_out_width = 1
        total_out_height = 1
    elif operation == 'MATMUL':
        total_out_width = params.total_width
        total_out_height = params.total_height
    elif operation == 'EWADDER':
        if params.stride == 1:
            total_out_width = 1
            total_out_height = 1
        else:
            total_out_width = params.total_width
            total_out_height = params.total_height
    elif operation == 'RESIZE':
        if params.stride & 0x4:  
            total_out_width = params.total_width // params.kernel_size
            total_out_height = params.total_height // params.kernel_size
        else:
            total_out_width = params.total_width * params.kernel_size
            total_out_height = params.total_height * params.kernel_size
    else:
        raise ValueError(f"Unsupported operation type: {operation}")
    
    # Apply maxpool stride if specified
    if params.maxpool_stride > 0:
        total_out_width = total_out_width >> params.maxpool_stride
        total_out_height = total_out_height >> params.maxpool_stride
    
    # Calculate total_out_2d_size
    total_out_2d_size = total_out_width * total_out_height
    
    # Calculate kernel 3D sizes
    if operation == 'D2_CONV':
        if params.kernel_2d_size != 3:
            total_kernel_3d = params.kernel_2d_size * params.total_in_ch
            tile_kernel_3d = params.kernel_2d_size * params.tile_in_ch
        else:
            total_kernel_3d = params.kernel_2d_size * (params.total_in_ch >> 1)
            tile_kernel_3d = params.kernel_2d_size * (params.tile_in_ch >> 1)
    elif operation == 'DW_CONV':
        total_kernel_3d = params.kernel_2d_size * params.total_out_ch
        tile_kernel_3d = params.kernel_2d_size * params.tile_out_ch
    else:  # MATMUL, LINEAR, RESIZE, or EWADDER
        if params.kernel_2d_size != 3:
            total_kernel_3d = params.total_2d_size * params.total_in_ch
            tile_kernel_3d = params.total_2d_size * params.tile_in_ch
        else:
            total_kernel_3d = params.total_2d_size * (params.total_in_ch >> 1) * 3
            tile_kernel_3d = params.total_2d_size * (params.tile_in_ch >> 1) * 3
    
    # Convert height parameters to hex format
    try:
        total_height_bin = format(params.total_height, '011b')
        tile_height_bin = format(params.tile_height, '011b')
        combined_height_bin = "0" * 10 + total_height_bin + tile_height_bin
        combined_height_hex = hex(int(combined_height_bin, 2))[2:].lower()
        height_hex = f"0x{combined_height_hex.zfill(8)}"
    except ValueError:
        height_hex = "0x00000000"
    
    # Convert width parameters to hex format
    try:
        total_width_bin = format(params.total_width, '011b')
        tile_width_bin = format(params.tile_width, '011b')
        combined_width_bin = "0" * 10 + total_width_bin + tile_width_bin
        combined_width_hex = hex(int(combined_width_bin, 2))[2:].lower()
        width_hex = f"0x{combined_width_hex.zfill(8)}"
    except ValueError:
        width_hex = "0x00000000"
    
    # Convert total in channel parameters to hex format
    try:
        total_in_ch_bin = format(params.total_in_ch, '011b')
        tile_in_ch_bin = format(params.tile_in_ch, '011b')
        combined_in_ch_bin = "0" * 10 + total_in_ch_bin + tile_in_ch_bin
        combined_in_ch_hex = hex(int(combined_in_ch_bin, 2))[2:].lower()
        in_ch_hex = f"0x{combined_in_ch_hex.zfill(8)}"
    except ValueError:
        in_ch_hex = "0x00000000"
    
    # Convert total out channel parameters to hex format
    try:
        total_out_ch_bin = format(params.total_out_ch, '011b')
        tile_out_ch_bin = format(params.tile_out_ch, '011b')
        combined_out_ch_bin = "0" * 10 + total_out_ch_bin + tile_out_ch_bin
        combined_out_ch_hex = hex(int(combined_out_ch_bin, 2))[2:].lower()
        out_ch_hex = f"0x{combined_out_ch_hex.zfill(8)}"
    except ValueError:
        out_ch_hex = "0x00000000"
    
    # Calculate width/height flag
    width_height_flag = 0
    if params.total_width == params.tile_width and params.total_height == params.tile_height:
        if operation == 'MATMUL':
            width_height_flag = 0b1000  # Both equal matmul
        else:
            width_height_flag = 0b0010  # Both equal
    elif params.total_width == params.tile_width:
        width_height_flag = 0b0001  # Only width equal
    else:
        width_height_flag = 0b0100  # Neither equal
    
    # Combine width_height_flag (MSB 16 bits) with read_out_ch (LSB 16 bits)
    combined_read_out_ch = (width_height_flag << 16) | read_out_ch
    
    # Convert to hex string
    combined_read_out_ch_hex = f"0x{combined_read_out_ch:08x}"
    
    return {
        'write_out_ch': write_out_ch,
        'read_out_ch': read_out_ch,
        'combined_read_out_ch': combined_read_out_ch_hex,
        'read_in_ch': read_in_ch,
        'burst_num_write': burst_num_write,
        'total_out_width': total_out_width,
        'total_out_height': total_out_height,
        'total_out_2d_size': total_out_2d_size,
        'total_kernel_3d': total_kernel_3d,
        'tile_kernel_3d': tile_kernel_3d,
        'kernel_2d_size': params.kernel_2d_size,
        'height_hex': height_hex,
        'width_hex': width_hex,
        'in_ch_hex': in_ch_hex,
        'out_ch_hex': out_ch_hex,
        'total_2d_size': params.total_2d_size,
        'total_width': params.total_width,
        'tile_in_ch': params.tile_in_ch,
        'tile_width': params.tile_width,
        'mcm_dataflow': mcm_dataflow,
        'operation_flag': operation_flag,
        'mcm_flag': mcm_flag,
        'combined_flags': combined_flags_hex
    }

def write_params_to_file(results, address, f, layer_num):
    """
    Write calculated parameters to a file in the specified format
    
    Args:
        results (dict): Dictionary containing calculated parameters
        f: File object to write to
        layer_num (int): Layer number to write as header
    """
    f.write(f"LAYER{layer_num}:\n")
    f.write(f"HB 0x00000001\n")#1-4
    f.write(f"HB {results['out_ch_hex']}\n") 
    f.write(f"HB {results['in_ch_hex']}\n")
    f.write(f"HB {results['width_hex']}\n")
    f.write(f"HB {results['height_hex']}\n") 
    f.write(f"HB 0x00000000\n")#in2_adress
    f.write(f"DB {results['total_2d_size']}\n")
    f.write(f"DB {results['total_kernel_3d']}\n")
    f.write(f"HB {results['combined_flags']}\n")
    f.write(f"DB {results['kernel_2d_size']}\n")
    f.write(f"DB {results['read_in_ch']}\n")
    f.write(f"DB {results['total_width']}\n")
    f.write(f"HB {results['combined_read_out_ch']}\n")
    f.write(f"DB {results['tile_width']}\n")   
    f.write(f"HB {results['mcm_flag']}\n")
    f.write(f"DB {results['total_out_width']}\n")
    f.write(f"DB {results['total_out_2d_size']}\n")    
    f.write(f"DB {results['write_out_ch']}\n")   
    f.write(f"HB 0x00000000\n")
    f.write(f"HB 0x00000000\n")

def generate_multiple_layers(layer_configs):
    """
    Generate parameters for multiple layers with different configurations
    
    Args:
        layer_configs (list): List of dictionaries containing layer configurations
            Each dictionary should have:
            - params: LayerParams object
            - address: layer_address object
            - operation: str (operation type)
            - output_file: str (name of output file)
    """
    # Open a single file for all layers
    with open('all_layers_params.txt', 'w') as f:
        for i, config in enumerate(layer_configs, 1):
            # Calculate parameters for the layer
            results = calculate_layer_params(config['params'], config['operation'])
            
            # Write results to file
            write_params_to_file(results, config['address'], f, i)
            
            # Print results for this layer
            print(f"\nResults for Layer {i}:")
            print(f"\nChannel Information:")
            print(f"Total In Channels: {config['params'].total_in_ch}")
            print(f"Total Out Channels: {config['params'].total_out_ch}")
            print(f"Tile In Channels: {config['params'].tile_in_ch}")
            print(f"Tile Out Channels: {config['params'].tile_out_ch}")
            print(f"\nRead/Write Information:")
            print(f"write_out_ch: {results['write_out_ch']}")
            print(f"read_out_ch: {results['read_out_ch']}")
            print(f"read_in_ch: {results['read_in_ch']}")
            print(f"burst_num: {results['burst_num_write']}")
    #          print(f"burst_length: {results['burst_length']}")
 #           print(f"\nHeight, Width and Channel parameters in hex format:")
            print(f"HB {results['height_hex']}")
            print(f"WB {results['width_hex']}")
            print(f"CB {results['in_ch_hex']}")
            print(f"CB {results['out_ch_hex']}")
            print(f"\nKernel sizes:")
            print(f"Kernel 2D size : {results['kernel_2d_size']}")
            print(f"Total kernel 3D size: {results['total_kernel_3d']}")
            print(f"Tile kernel 3D size: {results['tile_kernel_3d']}")

def create_layer_from_config(config):
    """
    Create LayerParams and layer_address objects from configuration dictionary
    
    Args:
        config (dict): Layer configuration dictionary
    
    Returns:
        dict: Dictionary with LayerParams and layer_address objects
    """
    # Extract parameters from the new format
    params = config['params']
    
    # Create LayerParams object with the new format
    layer_params = LayerParams(
        total_width=params['tile_width']['total'],
        total_height=params['tile_height']['total'],
        tile_width=params['tile_width']['tile'],
        tile_height=params['tile_height']['tile'],
        total_in_ch=params['tile_in_ch']['total'],
        total_out_ch=params['tile_out_ch']['total'],
        tile_in_ch=params['tile_in_ch']['tile'],
        tile_out_ch=params['tile_out_ch']['tile'],
        kernel_size=params['operation_cfg']['kernel_size'],
        stride=params['operation_cfg']['stride'],
        l_pad=params['pad_cfg']['l_pad'],
        r_pad=params['pad_cfg']['r_pad'],
        t_pad=params['pad_cfg']['t_pad'],
        b_pad=params['pad_cfg']['b_pad'],
        maxpool_stride=params['post_process_cfg']['mp_stride'],
        C_CONCAT=params['C_CONCAT'],
        post_process_cfg=params['post_process_cfg'],
        pad_cfg=params['pad_cfg']
    )
    
    # Create layer_address object with default values if not present
    address = layer_address(
        weight_adres=params.get('weight_offset', 0x00000000),
        IN1_adres=params.get('IN1_offset', 0x01000000),
        IN2_adres=params.get('IN2_offset', 0x00000000),
        Param_adres=params.get('param_offset', 0x02000000),
        out_adres=params.get('output1_offset', 0x03000000)
    )
    
    return {
        'params': layer_params,
        'address': address,
        'operation': config['operation'],
        'output_file': config['output_file']
    }

def calculate_addresses(num_layers, total_outch, total_inch, total_width, total_height, kernel_size, num_param):
    """
    Calculate memory addresses for weights, inputs, parameters and outputs for all layers
    
    Args:
        num_layers (int): Number of layers
        total_outch (list): List of output channels for each layer
        total_inch (list): List of input channels for each layer
        total_width (list): List of widths for each layer
        total_height (list): List of heights for each layer
        kernel_size (list): List of kernel sizes for each layer
        num_param (list): List of parameters for each layer
    
    Returns:
        list: List of layer_address objects for each layer
    """
    # Calculate weight addresses
    wgt_offset_address = []
    wgt_offset_address.append(0x00000000)
    for layer in range(1, num_layers):
        nxt_offset_address = wgt_offset_address[layer-1] + total_outch[layer-1] * total_inch[layer-1] * kernel_size[layer-1]**2
        wgt_offset_address.append(nxt_offset_address)

    # Calculate input addresses
    in1_offset_address = []
    in1_offset_address.append(0x01000000)
    for layer in range(1, num_layers):
        nxt_offset_address = in1_offset_address[layer-1] + total_inch[layer-1] * total_width[layer-1] * total_height[layer-1]
        in1_offset_address.append(nxt_offset_address)

    # Calculate IN2 addresses (all zeros)
    in2_offset_address = [0x00000000] * num_layers

    # Calculate parameter addresses
    param_offset_address = []
    param_offset_address.append(0x02000000)
    for layer in range(1, num_layers):
        nxt_offset_address = param_offset_address[layer-1] + total_outch[layer-1] * num_param[layer-1] * 2
        param_offset_address.append(nxt_offset_address)

    # Calculate output addresses
    out1_offset_address = []
    for layer in range(0, num_layers-1):
        out1_offset_address.append(in1_offset_address[layer+1])
    out1_offset_address.append(0x03000000)

    # Create layer_address objects
    addresses = []
    for layer in range(num_layers):
        addr = layer_address(
            wgt_offset_address[layer],
            in1_offset_address[layer],
            in2_offset_address[layer],
            param_offset_address[layer],
            out1_offset_address[layer]
        )
        addresses.append(addr)

    return addresses

def main():
    """Main function to handle layer generation from configuration file"""
    try:
        # Import layer configurations
        from layer_configs_yolov10 import layer_configs
        
        # Convert configurations to layer objects
        processed_configs = [create_layer_from_config(config) for config in layer_configs]
        
        # Generate parameters for all layers
        generate_multiple_layers(processed_configs)
        
        print("\nAll layer parameters have been generated")
        
    except ImportError:
        print("Error: CONFIG file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
