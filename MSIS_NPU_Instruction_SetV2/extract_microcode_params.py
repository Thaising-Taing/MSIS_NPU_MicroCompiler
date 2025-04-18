import re

def extract_hex_value(hex_str):
    """Extract integer value from hex string"""
    return int(hex_str, 16)

def parse_microcode(file_path):
    """Parse microcode file and extract layer parameters"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into layers based on LYREND
    layers = content.split('ctrl_write(OPCODE["LYREND"])')
    
    layer_configs = []
    
    for layer in layers:
        if not layer.strip():
            continue
            
        # Extract layer parameters
        params = {}
        
        # Extract layer name/type from comments
        name_match = re.search(r'##########(\w+)', layer)
        layer_name = name_match.group(1) if name_match else f"Layer_{len(layer_configs) + 1}"
        
        # Extract operation type and parameters
        operation_match = re.search(r'mainop_write\(OPCODE\["(\w+)"\],\s*FUNC_PARAM\["(\w+)"\],\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', layer)
        if operation_match:
            params['operation_cfg'] = {
                'operation': operation_match.group(2),
                'kernel_size': int(operation_match.group(3)),
                'stride': int(operation_match.group(4)),
                'post_valid': int(operation_match.group(5)),
                'branch': int(operation_match.group(6)),
                'q_method': int(operation_match.group(7))
            }
        else:
            # Set default operation configuration if not found
            params['operation_cfg'] = {
                'operation': 'D2_CONV',
                'kernel_size': 0,
                'stride': 0,
                'post_valid': 0,
                'branch': 0,
                'q_method': 0
            }
        
        # Extract postop parameters
        postop_match = re.search(r'postop_write\(OPCODE\["(\w+)"\],\s*FUNC_PARAM\["(\w+)"\],\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(?:POST_PROC\["(\w+)"\]|False),\s*(?:POST_PROC\["(\w+)"\]|False),\s*(?:POST_PROC\["(\w+)"\]|False),\s*(?:POST_PROC\["(\w+)"\]|False)\)', layer)
        if postop_match:
            op_type = postop_match.group(1)
            func_param = postop_match.group(2)
            
            if op_type == "OPTYPE" and func_param == "MAIN_PRCS":
                # Update operation_cfg for MAIN_PRCS
                params['operation_cfg'] = {
                    'operation': 'MAIN_PRCS',
                    'kernel_size': 0,
                    'stride': 0,
                    'post_valid': 0,
                    'branch': int(postop_match.group(3)),  # num_branch
                    'q_method': int(postop_match.group(4))  # quant_method
                }
                
                # Store post-processing parameters with correct order
                params['post_process_cfg'] = {
                    'active_slope': int(postop_match.group(3)),  # activation_slope
                    'mp_stride': int(postop_match.group(4)),     # mp_stride
                    'prcs1': postop_match.group(7) if postop_match.group(7) else '0',  # First POST_PROC parameter
                    'prcs2': postop_match.group(8) if postop_match.group(8) else '0',  # Second POST_PROC parameter
                    'prcs3': postop_match.group(9) if postop_match.group(9) else '0',  # Third POST_PROC parameter
                    'prcs4': postop_match.group(10) if postop_match.group(10) else '0'   # Fourth POST_PROC parameter
                }
            else:
                # For POST_PRCS and other cases
                params['post_process_cfg'] = {
                    'active_slope': int(postop_match.group(3)),  # activation_slope
                    'mp_stride': int(postop_match.group(4)),     # mp_stride
                    'prcs1': postop_match.group(7) if postop_match.group(7) else '0',  # First POST_PROC parameter
                    'prcs2': postop_match.group(8) if postop_match.group(8) else '0',  # Second POST_PROC parameter
                    'prcs3': postop_match.group(9) if postop_match.group(9) else '0',  # Third POST_PROC parameter
                    'prcs4': postop_match.group(10) if postop_match.group(10) else '0'   # Fourth POST_PROC parameter
                }
        else:
            params['post_process_cfg'] = {
                'active_slope': 0, 'mp_stride': 0,
                'prcs1': '0', 'prcs2': '0', 'prcs3': '0', 'prcs4': '0'
            }
        
        # Extract channel parameters
        out_ch_match = re.search(r'setreg_write\(OPCODE\["SETREG"\],\s*OPERAND1\["OUT_CHANNEL"\],\s*(\d+),\s*(\d+)\)', layer)
        in_ch_match = re.search(r'setreg_write\(OPCODE\["SETREG"\],\s*OPERAND1\["IN_CHANNEL"\],\s*(\d+),\s*(\d+)\)', layer)
        if out_ch_match:
            params['tile_out_ch'] = {
                'total': int(out_ch_match.group(1)),
                'tile': int(out_ch_match.group(2))
            }
        else:
            params['tile_out_ch'] = {'total': 0, 'tile': 0}
            
        if in_ch_match:
            params['tile_in_ch'] = {
                'total': int(in_ch_match.group(1)),
                'tile': int(in_ch_match.group(2))
            }
        else:
            params['tile_in_ch'] = {'total': 0, 'tile': 0}
        
        # Extract width and height parameters
        width_match = re.search(r'setreg_write\(OPCODE\["SETREG"\],\s*OPERAND1\["IN_WIDTH"\],\s*(\d+),\s*(\d+)\)', layer)
        height_match = re.search(r'setreg_write\(OPCODE\["SETREG"\],\s*OPERAND1\["IN_HEIGHT"\],\s*(\d+),\s*(\d+)\)', layer)
        if width_match:
            params['tile_width'] = {
                'total': int(width_match.group(1)),
                'tile': int(width_match.group(2))
            }
        else:
            params['tile_width'] = {'total': 0, 'tile': 0}
            
        if height_match:
            params['tile_height'] = {
                'total': int(height_match.group(1)),
                'tile': int(height_match.group(2))
            }
        else:
            params['tile_height'] = {'total': 0, 'tile': 0}
        
        # Extract padding parameters
        pad_match = re.search(r'ovppad_write\(OPCODE\["SETREG"\],\s*OPERAND1\["OVERLAP_PAD"\],\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', layer)
        if pad_match:
            params['pad_cfg'] = {
                't_ovlp': int(pad_match.group(1)),
                'b_ovlp': int(pad_match.group(2)),
                'l_ovlp': int(pad_match.group(3)),
                'r_ovlp': int(pad_match.group(4)),
                'pad_type': int(pad_match.group(5)),
                't_pad': int(pad_match.group(6)),
                'b_pad': int(pad_match.group(7)),
                'l_pad': int(pad_match.group(8)),
                'r_pad': int(pad_match.group(9))
            }
        else:
            params['pad_cfg'] = {
                't_ovlp': 0, 'b_ovlp': 0, 'l_ovlp': 0, 'r_ovlp': 0,
                'pad_type': 0, 't_pad': 0, 'b_pad': 0, 'l_pad': 0, 'r_pad': 0
            }
        
        # Extract quantization parameters
        # First try to find QU_PARAM_QK
        qk_match = re.search(r'qparam_write\(OPCODE\["SETREG"\],\s*OPERAND1\[f?"QU_PARAM_QK"\],\s*(\d+),\s*(\d+)\)', layer)
        if qk_match:
            params['quant_param_qk'] = {
                'shift': int(qk_match.group(1)),
                'scale': int(qk_match.group(2))
            }
        
        # Then find regular quant params
        qparam_matches = re.finditer(r'qparam_write\(OPCODE\["SETREG"\],\s*OPERAND1\[f?"QU_PARAM(\d+)"\],\s*(\d+),\s*(\d+)\)', layer)
        for match in qparam_matches:
            param_num = match.group(1)
            shift = int(match.group(2))
            scale = int(match.group(3))
            params[f'quant_param{param_num}'] = {'shift': shift, 'scale': scale}
        
        # Set default quantization parameters if missing
        if 'quant_param1' not in params:
            params['quant_param1'] = {'shift': 0, 'scale': 0}
        if 'quant_param2' not in params:
            params['quant_param2'] = {'shift': 0, 'scale': 0}
        if 'quant_param3' not in params:
            params['quant_param3'] = {'shift': 0, 'scale': 0}
        if 'quant_param4' not in params:
            params['quant_param4'] = {'shift': 0, 'scale': 0}                        
        if 'quant_param_qk' not in params:
            params['quant_param_qk'] = {'shift': 0, 'scale': 0}
        #extract addresses
        offset_matches = re.finditer(r'offset_write\(OPCODE\["(\w+)"\],\s*0x([0-9A-Fa-f]+)\)', layer)
        for match in offset_matches:
            offset_type = match.group(1)
            hex_value = match.group(2)
            if offset_type == 'LD_WGT':
                params['weight_offset'] = f'0x{hex_value}'
            elif offset_type == 'LD_IN1':
                params['IN1_offset'] = f'0x{hex_value}'
            elif offset_type == 'LD_IN2':
                params['IN2_offset'] = f'0x{hex_value}'
            elif offset_type == 'LD_PARAM':
                params['param_offset'] = f'0x{hex_value}'
            elif offset_type == 'ST_OUT1':
                params['output1_offset'] = f'0x{hex_value}'
            elif offset_type == 'ST_OUT2':
                params['output2_offset'] = f'0x{hex_value}'
            elif offset_type == 'ST_OUT3':
                params['output3_offset'] = f'0x{hex_value}'
            elif offset_type == 'ST_OUT4':
                params['output4_offset'] = f'0x{hex_value}'
        
        # Set default C_CONCAT value (used in calculating constant params)
        params['C_CONCAT'] = 2
        
        layer_configs.append({
            'params': params,
            'operation': params['operation_cfg']['operation'],
          #  'output_file': f'layer{len(layer_configs) + 1}_params.txt'
        })
    
    return layer_configs

def generate_layer_configs_file(layer_configs, output_file):
    """Generate layer_configs_yolov10.py file with extracted parameters"""
    with open(output_file, 'w') as f:
        f.write("# Layer configurations for yolov10\n")
        f.write("layer_configs = [\n")
        
        for i, config in enumerate(layer_configs, 1):
            f.write(f"    # Layer {i}\n")
            f.write("    {\n")
            f.write("        'params': {\n")
            
        
            params = config['params']
            param_order = [
                'tile_height', 'tile_width', 'tile_in_ch', 'tile_out_ch',
                'pad_cfg', 'operation_cfg', 'post_process_cfg',
                'quant_param1', 'quant_param2','quant_param3', 'quant_param4', 'quant_param_qk',
                'C_CONCAT'
            ]
            
            for key in param_order:
                if key in params:
                    if isinstance(params[key], dict):
                        f.write(f"            '{key}': {params[key]},\n")
                    else:
                        f.write(f"            '{key}': {params[key]},\n")
            
            # Write address parameters
            address_order = [
                'weight_offset', 'IN1_offset', 'IN2_offset',
                'output1_offset', 'output2_offset', 'output3_offset', 'output4_offset',
                'param_offset'
            ]
            
            for key in address_order:
                if key in params:
                    f.write(f"            '{key}': {params[key]},\n")
            
            f.write("        },\n")
            f.write(f"        'operation': '{config['operation']}',\n")
         #   f.write(f"        'output_file': '{config['output_file']}'\n")
            f.write("    },\n")
        
        f.write("]\n")


def Extract_MicroParams(args):
    # Parse microcode file
    layer_configs = parse_microcode(f"{args.output_dir}/{args.model_name}/{args.model_name}_MicroScriptV1.py")  
    generate_layer_configs_file(layer_configs, f"{args.output_dir}/{args.model_name}/layer_configs_{args.model_name}_.py")