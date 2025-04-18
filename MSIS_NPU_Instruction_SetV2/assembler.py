
import re


# INPUT_FILE = "assembly_instructions_yolov10_slim.txt"
# OUTPUT_FILE = "output_microcode.txt"

# Define opcode mapping
OPCODES = {
    'ADD': '00000',    #ADD scr1 and scr2 regs and reult saved to dest reg e-g ADD R2 R3 R4
    'SUB': '00001',    #SUBtract scr1 and scr2 regs and reult saved to dest reg e-g SUB R2 R3 R4
    'MUL': '00010',    #Multiply scr1 and scr2 regs and reult saved to dest reg e-g MUL R2 R3 R4
    'SHL': '00011',    #Shift left the source reg by constant value and store to dest register
    'MOVS': '00100',   #move data from RF to Special regs e-g MOVS CII R3
    'LDS': '00101',    #Load data to RFs in burst 
    'AVLD': '00110',  
    'MOV': '00111',   #move data from  Special regs RF e-g MOV CII R3  
    'JMP': '01000',   #simple jump to address LABELLED e-g JMP LABEL2
    'BEQ': '01001',   #compare to registers value if they equal jump to LABEL e-g BEQ R1 R2 LABEL1
    'LD': '01010',    #move immediate value to RF e-g LD R1 100
    'INIT': '10000',
    'LYREND': '10001',
    'OP': '10010',
    'FINISH': '10011',
    'JAL': '10100',   #Jump to LAbel and Program counter value is saved in return_adrs_register e-g JAL LABEL3
    'JALR': '10101',  #Jump to address stored in return_register,(done flag is also asserted) e-g JALR 1
    'POST_PR':'10110',
    'SHR' :  '10111',  #Shift right the source reg by constant value and store to dest register
    'QUAN_CFG': '11000',
    'PAD_CFG': '11001',
    'NO_OP': '11010',
    'MAIN_PRCS': '11011'
}

register_indices = {
    'ADRS_OUT_W': '00000',
    'OC': '00001',
    'IC': '00010',
    'IW': '00011',
    'IH': '00100',
    'MOI_W': '00101',
    'WT_ADRS': '00110',
    'IN1_ADRS': '00111',
    'IN2_ADRS': '01000',
    'OUT_ADRS': '01001',
    'PARAM_ADRS': '01010',
    'MHI': '01011',
    'MII': '01100',
    'MOI': '01101',
    'MAI': '01110',
    'CII': '01111',
    'COI': '10000',
    'CWI': '10001',
    'CHI': '10010',
    'WT_ST': '10011',
    'IN1_ST': '10100',
    'IN2_ST': '10101',
    'WR_ST': '10110',
    'PM_ST': '10111',
    'BC': '11000',
    'IAR': '11001',
    'IAW': '11010',
    'RD_ST': '11011',
    'RH': '11100',
    'CHI_W': '11101',
    'BN_W': '11110',
    'DDR_RD': '11111'
}

# Define operations and flags for the OP instruction
OP_OPERATIONS = {
    '2D_CONV': '00001',  
    'DW_CONV': '00010',   
    'MATMUL': '00011',
    'MAIN_PRCS': '00100',
    'RESIZE': '00101',
    'EW_ADDER': '00110',
}

POST_PROCESS = {
    'ADDER': '001',  
    'MULT': '010',   
    'ACTIV': '011',
    'POOL': '100',
    'SOFTMAX': '101',
}


def assembly_to_microcode(assembly_code):
    label_map = {}
    microcode_lines = []
    errors = []  # List to store error messages
    warnings = []  # List to store warning messages
    instruction_count = 0  # Counter for actual instructions

    # First pass to record label locations and count instructions
    line_num = 0
    for line in assembly_code.strip().split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.startswith("//"):  # Skip comments
            continue
        if line.endswith(':'):  # Skip labels 
            label = line[:-1]
            if label in label_map:
                errors.append(f"Error at line {line_num + 1}: Duplicate label '{label}'")
            label_map[label] = line_num
        else:  # This is an instruction
            instruction_count += 1
            microcode_lines.append(line)
            line_num += 1

    microcode_output = []
    for line_num, line in enumerate(microcode_lines, 1):
        line = line.strip()
        if line.startswith("//"):
            microcode_output.append(line)
            continue
        if not line:
            continue

        parts = line.split()
        opcode = parts[0].upper()
        
        # Handle DB and HB commands first
        if opcode == 'DB' and len(parts) == 2:
            try:
                decimal_value = parts[1]
                binary_value = format(int(decimal_value), '032b')
                microcode_output.append(binary_value)
                continue
            except ValueError:
                continue  # Since DB and HB arent real opcode we can ignore the error

        if opcode == 'HB' and len(parts) == 2:
            try:
                hex_value = parts[1]
                if not hex_value.startswith('0x'):
                    hex_value = '0x' + hex_value
                binary_value = format(int(hex_value, 16), '032b')
                microcode_output.append(binary_value)
                continue
            except ValueError:
                continue  # Since DB and HB arent real opcode we can ignore the error
        
        # Check if opcode exists for other instructions
        if opcode not in OPCODES:
            errors.append(f"Error at line {line_num}: Invalid opcode '{opcode}'")
            continue

        opcode_bin = OPCODES[opcode]

        try:
            # Special handling for LDS instruction
            if opcode == 'LDS' and len(parts) == 3:
                src1 = parts[1][1:] if parts[1].startswith('R') else parts[1]
                label = parts[2]
                try:
                    src1_bin = format(int(src1), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register number '{src1}'")
                    continue
                
                if label in label_map:
                    constant_bin = format(label_map[label], '013b')
                else:
                    errors.append(f"Error at line {line_num}: Undefined label '{label}'")
                    continue
                microcode_line = f"{opcode_bin}{'00000'}{src1_bin}{'0000'}{constant_bin}"

            elif opcode == 'BEQ' and len(parts) == 4:
                src1, src2 = [part[1:] if part.startswith('R') else part for part in parts[1:3]]
                label = parts[3]
                try:
                    src1_bin = format(int(src1), '05b')
                    src2_bin = format(int(src2), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register numbers in BEQ instruction")
                    continue
                
                if label in label_map:
                    constant_bin = format(label_map[label], '013b')
                else:
                    errors.append(f"Error at line {line_num}: Undefined label '{label}'")
                    continue
                microcode_line = f"{opcode_bin}{src1_bin}{src2_bin}{'0000'}{constant_bin}"

            elif opcode == 'JMP' and len(parts) == 2:
                label = parts[1]
                if label in label_map:
                    constant_bin = format(label_map[label], '013b')
                else:
                    errors.append(f"Error at line {line_num}: Undefined label '{label}'")
                    continue
                microcode_line = f"{opcode_bin}{'00000000000000'}{constant_bin}"

            elif opcode == 'JAL' and len(parts) == 2:
                label = parts[1]
                if label in label_map:
                    constant_bin = format(label_map[label], '013b')
                else:
                    errors.append(f"Error at line {line_num}: Undefined label '{label}'")
                    continue
                microcode_line = f"{opcode_bin}{'00000'}{'00000'}{'0000'}{constant_bin}"

            elif opcode == 'JALR' and len(parts) == 2:
                try:
                    dest_reg = 1 << (int(parts[1]) - 1)
                    dest_reg_bin = format(dest_reg, '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register number '{parts[1]}'")
                    continue
                zero_pad = '0' * 22
                microcode_line = f"{opcode_bin}{dest_reg_bin}{zero_pad}"

            elif opcode in ['LYREND', 'FINISH', 'AVLD', 'NO_OP'] and len(parts) == 1:
                microcode_line = f"{opcode_bin}{'0'*27}"
            elif opcode == 'INIT' and len(parts) == 2:
                try:
                    layer_num = parts[1]  
                    layer_num_bin = format(int(layer_num), '011b')
                    zero_pad = '0'*16  
                    microcode_line = f"{opcode_bin}{zero_pad}{layer_num_bin}"
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid layer number '{layer_num}' in INIT instruction")
                    continue
                
            elif opcode == 'OP' and len(parts) >= 2:
                operation = parts[1]
                if operation not in OP_OPERATIONS:
                    errors.append(f"Error at line {line_num}: Invalid operation '{operation}'")
                    continue
                    
                kernel = parts[2] if len(parts) > 2 else '0'
                stride = parts[3] if len(parts) > 3 else '0'
                P_valid = parts[4] if len(parts) > 4 else '0'
                branch_ = parts[5] if len(parts) > 5 else '0'
                Q_method = parts[6] if len(parts) > 6 else '0'

                try:
                    kernel_bin = format(int(kernel), '03b')
                    stride_bin = format(int(stride), '03b')
                    branch_bin = format(int(branch_), '02b')
                    Q_method_bin = format(int(Q_method), '03b')
                    P_valid_bin = format(int(P_valid), '01b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid numeric parameters in OP instruction")
                    continue

                operation_bin = OP_OPERATIONS[operation]
                microcode_line = f"{opcode_bin}{operation_bin}{kernel_bin}{stride_bin}{P_valid_bin}{'000'}{branch_bin}{Q_method_bin}{'0000000'}"

            elif opcode == 'MAIN_PRCS' and len(parts) >= 5:
                branch_ = parts[1]
                Q_method = parts[2]
                active_slope = parts[3]
                MP_stride = parts[4]
                PRCS1 = parts[5]
                PRCS2 = parts[6]
                PRCS3 = parts[7]
                PRCS4 = parts[8] 
                try:
                    operation_bin = OP_OPERATIONS['MAIN_PRCS']
                    branch_bin = format(int(branch_), '02b')
                    Q_method_bin = format(int(Q_method), '03b')
                    active_slope_bin = format(int(active_slope), '03b')
                    MP_stride_bin = format(int(MP_stride), '01b') 
                    PRCS1_bin = POST_PROCESS.get(PRCS1, '000')
                    PRCS2_bin = POST_PROCESS.get(PRCS2, '000')
                    PRCS3_bin = POST_PROCESS.get(PRCS3, '000')
                    PRCS4_bin = POST_PROCESS.get(PRCS4, '000')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid numeric parameters in MAIN_PRCS instruction")
                    continue    
                microcode_line = f"{opcode_bin}{operation_bin}{branch_bin}{Q_method_bin}{'0'}{active_slope_bin}{MP_stride_bin}{PRCS1_bin}{PRCS2_bin}{PRCS3_bin}{PRCS4_bin}" 

            elif opcode == 'POST_PR' and len(parts) >= 4:
                SLOPE = parts[1]
                MP_STRIDE = parts[2]
                PRCS1 = parts[3] if len(parts) > 3 else '0'
                PRCS2 = parts[4] if len(parts) > 4 else '0'
                PRCS3 = parts[5] if len(parts) > 5 else '0'
                PRCS4 = parts[6] if len(parts) > 6 else '0'

                try:
                    slope_bin = format(int(SLOPE), '03b')
                    MP_STRIDE_bin = format(int(MP_STRIDE), '01b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid numeric parameters in POST_PR instruction")
                    continue

                # Check each post-processing operation individually
                if PRCS1 != '0' and PRCS1 not in POST_PROCESS:
                    errors.append(f"Error at line {line_num}: Invalid post-processing operation '{PRCS1}'")
                    continue
                if PRCS2 != '0' and PRCS2 not in POST_PROCESS:
                    errors.append(f"Error at line {line_num}: Invalid post-processing operation '{PRCS2}'")
                    continue
                if PRCS3 != '0' and PRCS3 not in POST_PROCESS:
                    errors.append(f"Error at line {line_num}: Invalid post-processing operation '{PRCS3}'")
                    continue
                if PRCS4 != '0' and PRCS4 not in POST_PROCESS:
                    errors.append(f"Error at line {line_num}: Invalid post-processing operation '{PRCS4}'")
                    continue

                # Use '000' for unspecified operations
                PRCS1_bin = POST_PROCESS.get(PRCS1, '000')
                PRCS2_bin = POST_PROCESS.get(PRCS2, '000')
                PRCS3_bin = POST_PROCESS.get(PRCS3, '000')
                PRCS4_bin = POST_PROCESS.get(PRCS4, '000')

                microcode_line = f"{opcode_bin}{'00000000000'}{slope_bin}{MP_STRIDE_bin}{PRCS1_bin}{PRCS2_bin}{PRCS3_bin}{PRCS4_bin}"

            elif opcode in ['ADD', 'SUB', 'MUL', 'DIV'] and len(parts) == 4:
                try:
                    dest_reg, src1, src2 = [part[1:] if part.startswith('R') else part for part in parts[1:4]]
                    dest_reg_bin = format(int(dest_reg), '05b')
                    src1_bin = format(int(src1), '05b')
                    src2_bin = format(int(src2), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register numbers in arithmetic instruction")
                    continue
                microcode_line = f"{opcode_bin}{dest_reg_bin}{src1_bin}{src2_bin}{'000000000000'}"

            elif opcode == 'LD' and len(parts) == 3:
                try:
                    dest = parts[1]
                    constant = parts[2]
                    dest_bin = format(int(dest[1:]), '05b') if dest.startswith('R') else '00000'
                    constant_bin = format(int(constant), '017b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register number or constant in LD instruction")
                    continue
                microcode_line = f"{opcode_bin}{'00000'}{dest_bin}{constant_bin}"

            elif opcode == 'SHR' and len(parts) == 4:
                try:
                    dest = parts[1]
                    src1 = parts[2]
                    src2 = parts[3]
                    dest_bin = format(int(dest[1:]), '05b') if dest.startswith('R') else '00000'
                    src1_bin = format(int(src1[1:]), '05b') if src1.startswith('R') else '00000'
                    src2_bin = format(int(src2), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register numbers or shift amount in SHR instruction")
                    continue
                microcode_line = f"{opcode_bin}{dest_bin}{src1_bin}{src2_bin}{'000000000000'}"

            elif opcode == 'SHL' and len(parts) == 4:
                try:
                    dest = parts[1]
                    src1 = parts[2]
                    src2 = parts[3]
                    dest_bin = format(int(dest[1:]), '05b') if dest.startswith('R') else '00000'
                    src1_bin = format(int(src1[1:]), '05b') if src1.startswith('R') else '00000'
                    src2_bin = format(int(src2), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register numbers or shift amount in SHL instruction")
                    continue
                microcode_line = f"{opcode_bin}{dest_bin}{src1_bin}{src2_bin}{'000000000000'}"

            elif opcode == 'MOVS' and len(parts) >= 3:
                reg_name = parts[1]
                if reg_name not in register_indices:
                    errors.append(f"Error at line {line_num}: Invalid special register '{reg_name}'")
                    continue
                index_bin = register_indices[reg_name]
                try:
                    src_reg = parts[2][1:] if parts[2].startswith('R') else parts[2]
                    out_adrs_brnch = parts[3] if len(parts) > 3 else '0'
                    out_adrs_brnch_bin = format(int(out_adrs_brnch), '05b')
                    src_reg_bin = format(int(src_reg), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register numbers in MOVS instruction")
                    continue
                microcode_line = f"{opcode_bin}{index_bin}{src_reg_bin}{out_adrs_brnch_bin}{'0'*12}"

            elif opcode == 'MOV' and len(parts) == 3:
                reg_name = parts[1]
                if reg_name not in register_indices:
                    errors.append(f"Error at line {line_num}: Invalid special register '{reg_name}'")
                    continue
                index_bin = register_indices[reg_name]
                try:
                    src_reg = parts[2][1:] if parts[2].startswith('R') else parts[2]
                    src_reg_bin = format(int(src_reg), '05b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid register number in MOV instruction")
                    continue
                microcode_line = f"{opcode_bin}{src_reg_bin}{index_bin}{'00000000000000000'}"

            elif opcode == 'QUAN_CFG' and len(parts) == 4:
                try:
                    quan_num = parts[1]
                    shift_value = parts[2]
                    scale_value = parts[3]
                    quan_num_bin = format(int(quan_num), '05b')
                    shift_value_bin = format(int(shift_value), '05b')
                    scale_value_bin = format(int(scale_value), '016b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid numeric parameters in QUAN_CFG instruction")
                    continue
                microcode_line = f"{opcode_bin}{quan_num_bin}{'0'}{shift_value_bin}{scale_value_bin}"

            elif opcode == 'PAD_CFG' and len(parts) == 10:
                try:
                    T_cfg = parts[1]
                    B_cfg = parts[2]
                    L_cfg = parts[3]
                    R_cfg = parts[4]
                    pad_type = parts[5]
                    T_pad = parts[6]
                    B_pad = parts[7]
                    L_pad = parts[8]
                    R_pad = parts[9]

                    T_cfg_bin = format(int(T_cfg), '02b')
                    B_cfg_bin = format(int(B_cfg), '02b')
                    L_cfg_bin = format(int(L_cfg), '02b')
                    R_cfg_bin = format(int(R_cfg), '02b')
                    pad_type_bin = format(int(pad_type), '01b')
                    T_pad_bin = format(int(T_pad), '03b')
                    B_pad_bin = format(int(B_pad), '03b')
                    L_pad_bin = format(int(L_pad), '03b')
                    R_pad_bin = format(int(R_pad), '03b')
                except ValueError:
                    errors.append(f"Error at line {line_num}: Invalid numeric parameters in PAD_CFG instruction")
                    continue
                microcode_line = f"{opcode_bin}{T_cfg_bin}{B_cfg_bin}{L_cfg_bin}{R_cfg_bin}{pad_type_bin}{'000000'}{T_pad_bin}{B_pad_bin}{L_pad_bin}{R_pad_bin}"

            else:
                errors.append(f"Error at line {line_num}: Invalid instruction format for opcode '{opcode}'")
                continue

            microcode_output.append(microcode_line)

        except Exception as e:
            errors.append(f"Error at line {line_num}: {str(e)}")
            continue

    if errors:
        return '\n'.join(errors), warnings

    # Count actual microcode lines (excluding comments)
    microcode_count = len([line for line in microcode_output if not line.startswith("//")])
    
    # Always add the count information
    warnings.append(f"Instruction Count: {instruction_count}")
    warnings.append(f"Microcode Lines Generated: {microcode_count}")
    
    if instruction_count != microcode_count:
        warnings.append(f"Warning: Number of instructions ({instruction_count}) does not match number of microcode lines generated ({microcode_count})")
    
    return '\n'.join(microcode_output), warnings


def process_assembly_file(input_file, output_file):
    """Process assembly file and write microcode to output file"""
    try:
        # Read assembly code from input file
        with open(input_file, 'r') as f:
            assembly_code = f.read()
        
        # Convert assembly to microcode
        microcode_output, warnings = assembly_to_microcode(assembly_code)
        
        # Check if there are any errors 
        if isinstance(microcode_output, str) and microcode_output.startswith("Error"):
            print("Errors found during assembly conversion:")
            print(microcode_output)
            print("\nConversion aborted due to errors.")
            return
        
        # Print any warnings
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(warning)
        
        # Write microcode to output file only if there are no errors
        with open(output_file, 'w') as f:
            f.write(microcode_output)
        
        print(f"\nSuccessfully converted assembly to microcode. Output written to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except Exception as e:
        print(f"Error: {str(e)}")


# if __name__ == "__main__":
#     process_assembly_file(INPUT_FILE, OUTPUT_FILE) 