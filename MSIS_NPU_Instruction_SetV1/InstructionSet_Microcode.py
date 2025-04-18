import os
from MicroCompiler import parse_args

# Define Output Directory
args = parse_args()
args.output_dir = args.output_dir + "/" + args.model_name
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
file_path = f"{args.output_dir}/{args.model_name}_MicrocodeV1.txt"
open(file_path, "w")

# Define Output Directory
script_path = f"{args.output_dir}/{args.model_name}_MicroScriptV1.py"
with open (script_path, mode="w") as script: 
    script.write("from MSIS_NPU_Instruction_SetV1.InstructionSet_Microcode import *\n")
        
def ctrl_write(opcode):
    data = (opcode << 28)
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')        
          
def setreg_write(opcode, operand1, operand2, operand3):
    data = (opcode << 28) + (operand1 << 22) + (operand2 << 11) + operand3
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')
        
def ovppad_write(opcode, operand1, t_ovp, b_ovp, l_ovp, r_ovp, pad_type, t_pad, b_pad, l_pad, r_pad):
    data = (opcode << 28) + (operand1 << 22) + (t_ovp << 20) + (b_ovp << 18) + (l_ovp << 16) + (r_ovp << 14) + (pad_type << 12) + (t_pad << 9) + (b_pad << 6) + (l_pad << 3) + (r_pad << 0)
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')   

def qparam_write(opcode, operand1, shift, scale):
    data = (opcode << 28) + (operand1 << 22) + (shift << 16) + scale
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')
 
def mainop_write(opcode, operand1, kernel_size, stride, post_valid, branch, q_method, int32_store):
    data = (opcode << 28) + (operand1 << 22) + (kernel_size << 19) + (stride << 16) + (post_valid << 15) + (branch << 13) + (q_method << 8) + (int32_store << 7)
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')   

def postop_write(opcode, operand1, active_slope, q_method, branch, mp_stride, prcs1, prcs2, prcs3, prcs4):
    data = (opcode << 28) + (operand1 << 22) + (active_slope << 19) + (q_method << 16) + (branch << 13) + (mp_stride << 12) + (prcs1 << 9) + (prcs2 << 6) + (prcs3 << 3) + (prcs4)
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')     

def offset_write(opcode, address):
    data = (opcode << 28) + (address>>4)
    # data = (bin(data)[2:]).zfill(32)
    data = (hex(data)[2:]).zfill(8)
    with open(file_path, "a+") as fptr:
        fptr.write(data + '\n')   


# Opcode parameters
OPCODE = {
    "INIT": 1,
    "SETREG": 2,
    "OPTYPE": 3,
    "LYREND": 5,
    "IRQ": 6,
    "FINISH": 7,
    "LD_WGT": 8,
    "LD_IN1": 9,
    "LD_IN2": 10,
    "LD_PARAM": 11,
    "ST_OUT1": 12,
    "ST_OUT2": 13,
    "ST_OUT3": 14,
    "ST_OUT4": 15,
}

# Operand1 parameters
OPERAND1 = {
    "CURRENT_LYR": 1,
    "OUT_CHANNEL": 2,
    "IN_CHANNEL": 3,
    "IN_WIDTH": 4,
    "IN_HEIGHT": 5,
    "OVERLAP_PAD": 6,
    "QU_PARAM1": 7,
    "QU_PARAM2": 8,
    "QU_PARAM3": 9,
    "QU_PARAM4": 10,
    "QU_PARAM_QK": 11,
}

# Function parameters
FUNC_PARAM = {
    "D2_CONV": 1,
    "DW_CONV": 2,
    "MATMUL": 3,
    "MAIN_PRCS": 4,
    "RESIZE": 5,
    "EWADDER": 6,
    "POST_PRCS": 7,
    "LINEAR": 8,
}

# Post process parameters
POST_PROC = {
    "ADD": 1,
    "MUL": 2,
    "ACTIVE": 3,
    "MAXPOOL": 4,
    "SOFTMAX": 5,
}

def Conv_MicroGen(conv_idx, model, Ops, hscale_idx, H_scale, 
                  Weight_Address, Bias_Address,
                  Input_Address, Output_Address,
                  Width, Height): 
    
    layer_name  = f"Conv{conv_idx}"
    
    if hasattr(model, layer_name ): 
        Conv = getattr(model, layer_name ) 
    
    ctrl_write(OPCODE["INIT"])
    setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
    for ch in [32, 16, 8, 4]: 
        if Conv.out_channels % ch == 0: 
            OUT_CH = ch
            break
    setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   Conv.out_channels,    OUT_CH)
    if Conv.groups != 1:
        setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
    else: 
        IN_CH = 4
        for ch in [32, 16, 8, 4]: 
            if Conv.in_channels % ch == 0: 
                IN_CH = ch
                break
        setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    max(4, Conv.in_channels),     IN_CH)  
    if Width % 20 == 0: 
        Width_Tile_Size = 20
    else: 
        Width_Tile_Size = Width
    if Height % 20 == 0: 
        Height_Tile_Size = 20
    else: 
        Height_Tile_Size = Height
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      Width,   Width_Tile_Size)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     Height,  Height_Tile_Size)
    
    if Conv.kernel_size[0] == 3 and Conv.stride[0] == 1:  
        ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
    elif Conv.kernel_size[0] == 3 and Conv.stride[0] == 2:  
        ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
    else: 
        ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)

    H_Shift = 14
    if not isinstance(hscale_idx, list): 
        hscale_idx = [hscale_idx]
    else: 
        hscale_idx = hscale_idx
    
    if len(hscale_idx) > 1: 
        for i, scale in enumerate(hscale_idx):
            qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM{i+1}"], H_Shift, H_scale[scale])
    else: 
        qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], H_Shift, H_scale[hscale_idx[0]])
        
    branch = len(hscale_idx) - 1

    if Ops.startswith("ConvActMax"):
        if Conv.groups != 1: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM['DW_CONV'],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        else: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,1,    POST_PROC["ADD"],POST_PROC["ACTIVE"],POST_PROC["MAXPOOL"],False)
    elif Ops.startswith("ConvAct"):
        if Conv.groups != 1: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM['DW_CONV'],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        else: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
    else: 
        if Conv.groups != 1: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM['DW_CONV'],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        else: 
            mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    Conv.kernel_size[0],Conv.stride[0],1, branch,7, 0)
        postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)

    offset_write(OPCODE["LD_WGT"],    Weight_Address)      
    offset_write(OPCODE["LD_IN1"],    Input_Address)       
    offset_write(OPCODE["LD_PARAM"],  Bias_Address)        
    if not isinstance(Output_Address, list): 
        Output_Address = [Output_Address]
    else: 
        Output_Address = Output_Address
    if len(Output_Address) > 1: 
        for i, out_addr in enumerate(Output_Address):
            offset_write(OPCODE[f"ST_OUT{i+1}"],   out_addr)
    else: 
        offset_write(OPCODE[f"ST_OUT1"],   Output_Address[0])
    ctrl_write(OPCODE["LYREND"])
    

def Conv_ScriptGen(conv_idx, model, Ops, hscale_idx, H_scale, 
                  Weight_Address, Bias_Address,
                  Input_Address, Output_Address,
                  Width, Height, Script_Path): 
    
    layer_name  = f"Conv{conv_idx}"
    
    if hasattr(model, layer_name ): 
        Conv = getattr(model, layer_name ) 
    
    with open(Script_Path, mode="a+") as script:
        script.write('ctrl_write(OPCODE["INIT"])\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)\n')
        for ch in [32, 16, 8, 4]: 
            if Conv.out_channels % ch == 0: 
                OUT_CH = ch
                break
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   {Conv.out_channels},    {OUT_CH})\n')
        if Conv.groups != 1:
            script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)\n')
        else: 
            IN_CH = 4
            for ch in [32, 16, 8, 4]: 
                if Conv.in_channels % ch == 0: 
                    IN_CH = ch
                    break
            script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    {max(4, Conv.in_channels)},     {IN_CH})\n')  
        if Width % 20 == 0: 
            Width_Tile_Size = 20
        else: 
            Width_Tile_Size = Width
        if Height % 20 == 0: 
            Height_Tile_Size = 20
        else: 
            Height_Tile_Size = Height
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      {Width},   {Width_Tile_Size})\n')
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     {Height},  {Height_Tile_Size})\n')
        
        if Conv.kernel_size[0] == 3 and Conv.stride[0] == 1:  
            script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)\n')
        elif Conv.kernel_size[0] == 3 and Conv.stride[0] == 2:  
            script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)\n')
        else: 
            script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)\n')

        H_Shift = 14
        if not isinstance(hscale_idx, list): 
            hscale_idx = [hscale_idx]
        else: 
            hscale_idx = hscale_idx
        
        if len(hscale_idx) > 1: 
            for i, scale in enumerate(hscale_idx):
                script.write(f'qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM{i+1}"], {H_Shift}, {H_scale[scale]})\n')
        else: 
            script.write(f'qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], {H_Shift}, {H_scale[hscale_idx[0]]})\n')
            
        branch = len(hscale_idx) - 1

        if Ops.startswith("ConvActMax"):
            if Conv.groups != 1: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            else: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            script.write(f'postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,1,    POST_PROC["ADD"],POST_PROC["ACTIVE"],POST_PROC["MAXPOOL"],False)\n')
        elif Ops.startswith("ConvAct"):
            if Conv.groups != 1: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            else: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            script.write('postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)\n')
        else: 
            if Conv.groups != 1: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            else: 
                script.write(f'mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    {Conv.kernel_size[0]},{Conv.stride[0]},1, {branch},7, 0)\n')
            script.write('postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)\n')

        script.write(f'offset_write(OPCODE["LD_WGT"],    0x{hex(Weight_Address)[2:].zfill(8)})\n')
        script.write(f'offset_write(OPCODE["LD_IN1"],    0x{hex(Input_Address)[2:].zfill(8)})\n')
        script.write(f'offset_write(OPCODE["LD_PARAM"],  0x{hex(Bias_Address)[2:].zfill(8)})\n')          
        if not isinstance(Output_Address, list): 
            Output_Address = [Output_Address]
        else: 
            Output_Address = Output_Address
        if len(Output_Address) > 1: 
            for i, out_addr in enumerate(Output_Address):
                script.write(f'offset_write(OPCODE["ST_OUT{i+1}"],   0x{hex(out_addr)[2:].zfill(8)})\n')
        else: 
            script.write(f'offset_write(OPCODE["ST_OUT1"],   0x{hex(Output_Address[0])[2:].zfill(8)})\n')
        script.write('ctrl_write(OPCODE["LYREND"])\n')


def EWAdder_MicroGen(args, layer, Input_Address, Output_Address, Width, Height): 
    ctrl_write(OPCODE["INIT"])
    setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
    if args.model_name == "YOLOv10n_Slim":
        if layer == "layer1":
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,   16)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
        else: 
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
    elif args.model_name == "YOLOv10n": 
        if layer == "layer2":
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,   16)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
        elif layer == "layer4":
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   32,   32)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
        elif layer == "layer6":
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   64,   32)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
        else:
            setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
            setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
    if Width % 20 == 0: 
        Width_Tile_Size = 20
    else: 
        Width_Tile_Size = Width
    if Height % 20 == 0: 
        Height_Tile_Size = 20
    else: 
        Height_Tile_Size = Height
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      Width,   Width_Tile_Size)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     Height,  Height_Tile_Size)
    ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)

    mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
    offset_write(OPCODE["LD_IN1"],    Input_Address[0])     
    offset_write(OPCODE["LD_IN2"],    Input_Address[1])     
    offset_write(OPCODE["ST_OUT1"],   Output_Address)       
    ctrl_write(OPCODE["LYREND"])
    

def EWAdder_ScriptGen(args, layer, Input_Address, Output_Address, Width, Height, Script_Path): 
    with open(Script_Path, mode="a+") as script:
        script.write('ctrl_write(OPCODE["INIT"])\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)\n')
        if args.model_name == "YOLOv10n_Slim":
            if layer == "layer1":
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,   16)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
            else: 
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
        elif args.model_name == "YOLOv10n": 
            if layer == "layer2":
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,   16)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
            elif layer == "layer4":
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   32,   32)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
            elif layer == "layer6":
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   64,   32)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
            else: 
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)\n')
                script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)\n')
        if Width % 20 == 0: 
            Width_Tile_Size = 20
        else: 
            Width_Tile_Size = Width
        if Height % 20 == 0: 
            Height_Tile_Size = 20
        else: 
            Height_Tile_Size = Height
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      {Width},   {Width_Tile_Size})\n')
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     {Height},  {Height_Tile_Size})\n')
        script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)\n')
        script.write('mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)\n')
        script.write(f'offset_write(OPCODE["LD_IN1"],    0x{hex(Input_Address[0])[2:].zfill(8)})\n')   
        script.write(f'offset_write(OPCODE["LD_IN2"],    0x{hex(Input_Address[1])[2:].zfill(8)})\n')     
        script.write(f'offset_write(OPCODE["ST_OUT1"],   0x{hex(Output_Address)[2:].zfill(8)})\n')      
        script.write('ctrl_write(OPCODE["LYREND"])\n')


def MaxPool_MicroGen(Input_Address, Output_Address, Width, Height):
    ctrl_write(OPCODE["INIT"])
    setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
    setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
    if Width % 20 == 0: 
        Width_Tile_Size = 20
    else: 
        Width_Tile_Size = Width
    if Height % 20 == 0: 
        Height_Tile_Size = 20
    else: 
        Height_Tile_Size = Height
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      Width,   Width_Tile_Size)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     Height,  Height_Tile_Size)
    ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,1,0,1, 1, 0,1,0,1)

    postop_write(OPCODE["OPTYPE"], FUNC_PARAM["MAIN_PRCS"], 0,0,0,0,    POST_PROC["MAXPOOL"],False,False,False)
    offset_write(OPCODE["LD_IN1"],    Input_Address)         
    offset_write(OPCODE["ST_OUT1"],   Output_Address)        
    ctrl_write(OPCODE["LYREND"])
    

def MaxPool_ScriptGen(Input_Address, Output_Address, Width, Height, Script_Path):
    with open (Script_Path, mode="a+") as script:
        script.write('ctrl_write(OPCODE["INIT"])\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)\n')
        if Width % 20 == 0: 
            Width_Tile_Size = 20
        else: 
            Width_Tile_Size = Width
        if Height % 20 == 0: 
            Height_Tile_Size = 20
        else: 
            Height_Tile_Size = Height
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      {Width},   {Width_Tile_Size})\n')
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     {Height},  {Height_Tile_Size})\n')
        script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,1,0,1, 1, 0,1,0,1)\n')

        script.write('postop_write(OPCODE["OPTYPE"], FUNC_PARAM["MAIN_PRCS"], 0,0,0,0,    POST_PROC["MAXPOOL"],False,False,False)\n')
        script.write(f'offset_write(OPCODE["LD_IN1"],    0x{hex(Input_Address)[2:].zfill(8)})\n')       
        script.write(f'offset_write(OPCODE["ST_OUT1"],   0x{hex(Output_Address)[2:].zfill(8)})\n')       
        script.write('ctrl_write(OPCODE["LYREND"])\n')
    

def Attention_MicroGen(H_scale, hscale_idx, Input_Address, Output_Address, Width, Height):
    ctrl_write(OPCODE["INIT"])
    setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,      0)
    setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,     16)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    32,     32)
    if Width % 20 == 0: 
        Width_Tile_Size = 20
    else: 
        Width_Tile_Size = Width
    if Height % 20 == 0: 
        Height_Tile_Size = 20
    else: 
        Height_Tile_Size = Height
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      Width,   Width_Tile_Size)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     Height,  Height_Tile_Size)

    H_Shift = 14
    if not isinstance(hscale_idx, list): 
        hscale_idx = [hscale_idx]
    else: 
        hscale_idx = hscale_idx
    qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM_QK"], H_Shift+2, H_scale[hscale_idx[0]])
    qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM1"], H_Shift, H_scale[hscale_idx[1]])

    mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["MATMUL"],    0,0,0, 0,7, 0) 
    offset_write(OPCODE["LD_WGT"],  Input_Address[0]) # Query
    offset_write(OPCODE["LD_IN1"],  Input_Address[1]) # Key
    offset_write(OPCODE["LD_IN2"],  Input_Address[2]) # Value
    offset_write(OPCODE["ST_OUT1"], Output_Address) 
    ctrl_write(OPCODE["LYREND"])
    

def Attention_ScriptGen(H_scale, hscale_idx, Input_Address, Output_Address, Width, Height, Script_Path):
    with open(Script_Path, mode="a+") as script: 
        script.write('ctrl_write(OPCODE["INIT"])\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,      0)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,     16)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    32,     32)\n')
        if Width % 20 == 0: 
            Width_Tile_Size = 20
        else: 
            Width_Tile_Size = Width
        if Height % 20 == 0: 
            Height_Tile_Size = 20
        else: 
            Height_Tile_Size = Height
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      {Width},   {Width_Tile_Size})\n')
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     {Height},  {Height_Tile_Size})\n')

        H_Shift = 14
        
        if not isinstance(hscale_idx, list): 
            hscale_idx = [hscale_idx]
        else: 
            hscale_idx = hscale_idx
        script.write(f'qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM_QK"], {H_Shift+2}, {H_scale[hscale_idx[0]]})\n')
        script.write(f'qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM1"], {H_Shift}, {H_scale[hscale_idx[1]]})\n')

        script.write('mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["MATMUL"],    0,0,0, 0,7, 0)\n')
        script.write(f'offset_write(OPCODE["LD_WGT"],  0x{hex(Input_Address[0])[2:].zfill(8)})\n') # Query
        script.write(f'offset_write(OPCODE["LD_IN1"],  0x{hex(Input_Address[1])[2:].zfill(8)})\n') # Key
        script.write(f'offset_write(OPCODE["LD_IN2"],  0x{hex(Input_Address[2])[2:].zfill(8)})\n') # Value
        script.write(f'offset_write(OPCODE["ST_OUT1"], 0x{hex(Output_Address)[2:].zfill(8)})\n')
        script.write('ctrl_write(OPCODE["LYREND"])\n')


def Upsample_MicroGen(Input_Address, Output_Address, Width, Height):
    ctrl_write(OPCODE["INIT"])
    setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
    setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
    if Width % 20 == 0: 
        Width_Tile_Size = 20
    else: 
        Width_Tile_Size = Width
    if Height % 20 == 0: 
        Height_Tile_Size = 20
    else: 
        Height_Tile_Size = Height
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      Width,   Width_Tile_Size)
    setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     Height,  Height_Tile_Size)
    ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)

    mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["RESIZE"],       2,1,0, 0,0, 0)
    offset_write(OPCODE["LD_IN1"],    Input_Address)     
    offset_write(OPCODE["ST_OUT1"],   Output_Address)    
    ctrl_write(OPCODE["LYREND"])
    

def Upsample_ScriptGen(Input_Address, Output_Address, Width, Height, Script_Path):
    with open(Script_Path, "a+") as script:
        script.write('ctrl_write(OPCODE["INIT"])\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)\n')
        script.write('setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)\n')
        if Width % 20 == 0: 
            Width_Tile_Size = 20
        else: 
            Width_Tile_Size = Width
        if Height % 20 == 0: 
            Height_Tile_Size = 20
        else: 
            Height_Tile_Size = Height
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      {Width},   {Width_Tile_Size})\n')
        script.write(f'setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     {Height},  {Height_Tile_Size})\n')
        script.write('ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)\n')

        script.write('mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["RESIZE"],       2,1,0, 0,0, 0)\n')
        script.write(f'offset_write(OPCODE["LD_IN1"],    0x{hex(Input_Address)[2:].zfill(8)})\n')   
        script.write(f'offset_write(OPCODE["ST_OUT1"],   0x{hex(Output_Address)[2:].zfill(8)})\n')  
        script.write('ctrl_write(OPCODE["LYREND"])\n')