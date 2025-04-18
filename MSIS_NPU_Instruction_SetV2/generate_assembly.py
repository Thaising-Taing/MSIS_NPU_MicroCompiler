from pre_calculate_param import LayerParams, calculate_layer_params

def convert_operation_to_assembly(op_cfg):
    """Convert operation configuration to assembly format"""
    op = op_cfg['operation']
    if op == 'MAIN_PRCS':
        # For MAIN_PRCS, return empty string as it will be handled in post_process but we need to clear the previous parameters
        return f"OP {'MAIN_PRCS'} {'0'} {'0'} {'0'} {'0'} {'0'}"
    elif op == 'D2_CONV':
        op = '2D_CONV'
    elif op == 'DW_CONV':
        op = 'DW_CONV'
    elif op == 'MATMUL':
        op = 'MATMUL'
    elif op == 'EWADDER':
        op = 'EW_ADDER'
 
    return f"OP {op} {op_cfg['kernel_size']} {op_cfg['stride']} {op_cfg['post_valid']} {op_cfg['branch']} {op_cfg['q_method']}"

def convert_post_process_to_assembly(post_cfg, op_cfg):
    """Convert post process configuration to assembly format"""
    def get_process_name(prcs):
        if prcs == 0:
            return '0'
        elif prcs == 'ADD':
            return 'ADDER'
        elif prcs == 'MUL':
            return 'MUL'
        elif prcs == 'ACTIVE':
            return 'ACTIV'
        elif prcs == 'MAXPOOL':
            return 'POOL'
        elif prcs == 'SOFTMAX':
            return 'SOFTMAX'
        return '0'
    
    if op_cfg['operation'] == 'MAIN_PRCS':
        # For MAIN_PRCS, generate different format
        return f"MAIN_PRCS {op_cfg['branch']} {op_cfg['q_method']} {post_cfg['active_slope']} {post_cfg['mp_stride']} {get_process_name(post_cfg['prcs1'])} {get_process_name(post_cfg['prcs2'])} {get_process_name(post_cfg['prcs3'])} {get_process_name(post_cfg['prcs4'])}"
    else:
        return f"POST_PR {post_cfg['active_slope']} {post_cfg['mp_stride']} {get_process_name(post_cfg['prcs1'])} {get_process_name(post_cfg['prcs2'])} {get_process_name(post_cfg['prcs3'])} {get_process_name(post_cfg['prcs4'])}"

def convert_pad_to_assembly(pad_cfg):
    """Convert padding configuration to assembly format"""
    return f"PAD_CFG {pad_cfg['t_ovlp']} {pad_cfg['b_ovlp']} {pad_cfg['l_ovlp']} {pad_cfg['r_ovlp']} {pad_cfg['pad_type']} {pad_cfg['t_pad']} {pad_cfg['b_pad']} {pad_cfg['l_pad']} {pad_cfg['r_pad']}"

def convert_quant_to_assembly(quant_params):
    """Convert quantization parameters to assembly format"""
    lines = []
    # Add quant_param1 through quant_param4
    for i in range(1, 5):
        param_key = f'quant_param{i}'
        if param_key in quant_params:
            lines.append(f"QUAN_CFG {i} {quant_params[param_key]['shift']} {quant_params[param_key]['scale']}")
    
    # Add quant_param_qk
    if 'quant_param_qk' in quant_params:
        lines.append(f"QUAN_CFG 5 {quant_params['quant_param_qk']['shift']} {quant_params['quant_param_qk']['scale']}")
    return '\n'.join(lines)

def convert_to_hex_format(total, tile):
    """Convert total and tile values to combined hex format"""
    try:
        total_bin = format(total, '011b')
        tile_bin = format(tile, '011b')
        combined_bin = "0" * 10 + total_bin + tile_bin
        combined_hex = hex(int(combined_bin, 2))[2:].lower()
        return f"0x{combined_hex.zfill(8)}"
    except ValueError:
        return "0x00000000"

def write_layer_params(f, layer_num, params,avld):
    """Write layer parameters in hex format"""
    # Create LayerParams object
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
    
    # Calculate all parameters
    results = calculate_layer_params(layer_params, params['operation_cfg']['operation'])
    
    # Write out channel parameters
    f.write(f"HB {results['out_ch_hex']}\n")
    # Write in channel parameters
    f.write(f"HB {results['in_ch_hex']}\n")
    # Write width parameters
    f.write(f"HB {results['width_hex']}\n")
    # Write height parameters
    f.write(f"HB {results['height_hex']}\n")
    # Write IN2_address if it exists
    if 'IN2_offset' in params:
        in2_addr = params['IN2_offset']
        if isinstance(in2_addr, int):
            in2_addr = f"0x{in2_addr:08x}"
        elif isinstance(in2_addr, str) and not in2_addr.startswith('0x'):
            in2_addr = f"0x{in2_addr}"
        f.write(f"HB {in2_addr}\n")
    if avld:
    # Write pre-calculated parameters
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
    else:
        f.write(f"HB 0x00000000\n")
        f.write(f"HB 0x00000000\n")
    

def write_addresses(f, layer_num, params):
    """Write addresses in hex format for a layer"""
    # Define the order of addresses
    address_order = [
        'weight_offset',
        'IN1_offset',
        'param_offset',
        'output1_offset',
        'output2_offset',
        'output3_offset',
        'output4_offset'
    ]
    
    # Write each address in the specified order
    for addr_name in address_order:
        addr_value = params.get(addr_name, 0)
        # Convert to hex string if it's an integer
        if isinstance(addr_value, int):
            addr_value = f"0x{addr_value:08x}"  # Convert to 8-digit hex with leading zeros
        elif isinstance(addr_value, str) and not addr_value.startswith('0x'):
            addr_value = f"0x{addr_value}"
        f.write(f"HB {addr_value}\n")

def write_movs_instructions(f, params):
    """Write MOVS instructions for the layer"""
    # Write MOVS instructions for addresses that exist in the config
    if 'weight_offset' in params:
        f.write("SHR R2 R2 4\n")
        f.write(f"MOVS WT_ADRS R2\n")
    if 'IN1_offset' in params:
        f.write("SHR R3 R3 4\n")
        f.write(f"MOVS IN1_ADRS R3\n")
    if 'IN2_offset' in params:
        f.write("SHR R13 R13 4\n")
        f.write(f"MOVS IN2_ADRS R13\n")
    if 'param_offset' in params:
        f.write("SHR R4 R4 4\n")
        f.write(f"MOVS PARAM_ADRS R4\n")
    if 'output1_offset' in params:
        f.write("SHR R5 R5 4\n")
        f.write(f"MOVS OUT_ADRS R5 1 \n")
    if 'output2_offset' in params:
        f.write("SHR R6 R6 4\n")
        f.write(f"MOVS OUT_ADRS R6 2 \n")
    if 'output3_offset' in params:
        f.write("SHR R7 R7 4\n")
        f.write(f"MOVS OUT_ADRS R7 3 \n")
    if 'output4_offset' in params:
        f.write("SHR R8 R8 4\n")
        f.write(f"MOVS OUT_ADRS R8 4 \n")
    
    # Write MOVS instructions for channel and dimension parameters in correct order
    if 'tile_out_ch' in params:
        f.write(f"MOVS OC R9\n")
    if 'tile_in_ch' in params:
        f.write(f"MOVS IC R10\n")
    if 'tile_width' in params:
        f.write(f"MOVS IW R11\n")
    if 'tile_height' in params:
        f.write(f"MOVS IH R12\n")

def read_address_cal_instructions():
    """Read address calculation instructions from address_cal.txt"""
    try:
        with open('address_cal.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Warning: address_cal.txt not found. ")
        return ''

def generate_assembly_instructions(layer_configs, output_file, layer_selection='all', avld=False):
    """
    Generate assembly instructions for each layer and write to file
    
    Args:
        layer_configs (list): List of layer configurations
        output_file (str): Output file path
        layer_selection (str or int): Layer selection. Can be:
            - 'all': Generate all layers
            - int: Generate only that specific layer (1-based index)
            - tuple: (start, end) Generate layers from start to end (inclusive)
        avld (bool): Whether to include address calculation instructions
    """
    # Determine which layers to process
    if layer_selection == 'all':
        layers_to_process = layer_configs
        layer_indices = range(1, len(layer_configs) + 1)
    elif isinstance(layer_selection, int):
        if layer_selection < 1 or layer_selection > len(layer_configs):
            raise ValueError(f"Layer {layer_selection} does not exist. Total layers: {len(layer_configs)}")
        layers_to_process = [layer_configs[layer_selection - 1]]
        layer_indices = [layer_selection]
    elif isinstance(layer_selection, tuple):
        start, end = layer_selection
        if start < 1 or end > len(layer_configs):
            raise ValueError(f"Layer range {start}-{end} is invalid. Total layers: {len(layer_configs)}")
        layers_to_process = layer_configs[start-1:end]
        layer_indices = range(start, end + 1)
    else:
        raise ValueError("Invalid layer selection. Use 'all', a layer number, or (start, end)")
    
    with open(output_file, 'w') as f:
        # First write all layer instructions
        for i, config in zip(layer_indices, layers_to_process):
            params = config['params']

            f.write(f"NO_OP\n")
            f.write(f"INIT {i-1}\n")
            f.write(f"LDS 14 LAYER{i}\n")            
            write_movs_instructions(f, config['params'])
            # Write operation instruction
            op_assembly = convert_operation_to_assembly(params['operation_cfg'])
            if op_assembly:  # Only write if not MAIN_PRCS
                f.write(op_assembly + "\n")
            
            # Write post process instruction
            f.write(convert_post_process_to_assembly(params['post_process_cfg'], params['operation_cfg']) + "\n")
            
            # Write padding instruction
            f.write(convert_pad_to_assembly(params['pad_cfg']) + "\n")
            
            # Write quantization instructions
            f.write(convert_quant_to_assembly(params) + "\n")
            
            # Write AVLD instruction if enabled
            if avld:
                f.write("AVLD\n")
                f.write(f"JAL LYR{i}\n") #will return to this label after address calculation
                f.write(f"LYR{i}:\n") #LABEL FOR RETURNING
                f.write("LYREND\n")
                f.write("MOV WR_ST R26\n")
                f.write("BEQ R26 R1 WR_LABEL\n") #if R26 is 1 then jump to WR_LABEL(WRITE ADDRESS CALCULATION) 
                f.write("MOV IN1_ST R26\n")
                f.write("BEQ R26 R1 CH_LABEL\n")
                f.write("MOV WT_ST R26\n")
                f.write("BEQ R26 R1 CH_LABEL\n")
                f.write("MOV IN2_ST R26\n")
                f.write("BEQ R26 R1 CH_LABEL\n")
                f.write("MOV PM_ST R26\n")
                f.write("BEQ R26 R1 CH_LABEL\n")
            else:    
                f.write("LYREND\n")
        f.write(f"FINISH\n")    
         # Write address calculation instructions only once at the end if AVLD is enabled
        if avld:
           # f.write("\n# Address Calculation Instructions\n")
            f.write(read_address_cal_instructions() + "\n")       
        # Then write all layer parameters and addresses at the end
        for i, config in zip(layer_indices, layers_to_process):
            # Write layer label
            f.write(f"LAYER{i}:\n")
            
            # Write fixed value at the start this is for using in branch instructions so not to load 1 again and again
            f.write("HB 0x00000001\n")
            
            # Write addresses first
            write_addresses(f, i, config['params'])
            
            # Write layer parameters (height, width, channels,constants)
            write_layer_params(f, i, config['params'],avld)    


def main():
    # Import layer configurations
   # from layer_configs_yolov10 import layer_configs
    from layer_configs_yolov10n_slim import layer_configs
    
    # Generate assembly instructions and write to file
    output_file = 'assembly_instructions_yolov10_slim.txt'
    generate_assembly_instructions(layer_configs, output_file, (18,19), avld=True)  # Set avld=True to enable address calculation
    print(f"Assembly instructions have been written to {output_file}")

if __name__ == "__main__":
    main() 