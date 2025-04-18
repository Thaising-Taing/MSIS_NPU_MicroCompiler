###########################################
#                                         #
#       Designed By: Thaising Taing       #
#               MSIS Lab                  #
#                                         #
###########################################

import re
import argparse
from tabulate import tabulate
from termcolor import colored
# ---------------- MSIS NPU Instruction SetV1 --------------------
from MSIS_NPU_Instruction_SetV1.InstructionSet_Microcode import *
# ---------------- MSIS NPU Instruction SetV2 --------------------
from MSIS_NPU_Instruction_SetV2.extract_microcode_params import Extract_MicroParams
# ------------------ Activation Addresses ------------------------
from Model.YOLOv10n_Slim_Address import CONV21_OUTPUT, YOLOv10n_Slim_Address_Map
from Model.YOLOv10n_Address import CONV52_OUTPUT, YOLOv10n_Address_Map
# ---------------- Architecture Script --------------------------
from Model.Architecture import YOLOv10n_Slim_Architecture, YOLOv10n_Architecture
# -------------------- Inference & Models -----------------------
from Model.SSBQ_YOLOv10n_Slim_LeakyReLU_TestVector import Inference_YOLOv10n_Slim, YOLOv10n_Slim, YOLOv10n_Detector
from Model.SSBQ_YOLOv10n_LeakyReLU_TestVector import Inference_YOLOv10n, YOLOv10n, YOLOv10n_Detector

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Simple-Compiler')
    parser.add_argument('--model_name', dest='model_name',
                        default='YOLOv10n_Slim', type=str,
                        help=["YOLOv10n_Slim, YOLOv10n"])
    parser.add_argument('--image_path', dest='image_path',
                        default="images/bus.jpg", type=str,
                        help=["Inference Image"])
    parser.add_argument('--output_dir', dest='output_dir',
                        default='Microcode', type=str)
    parser.add_argument('--DEBUG', dest='DEBUG',
                        default=False, type=bool,
                        help="Print Out Layers Name")
    parser.add_argument('--DEBUG_WRITE', dest='DEBUG_WRITE',
                        default=False, type=bool,
                        help="Write Layers Name into Microcode File")

    args = parser.parse_args()
    args.save_path = f"Parameters_zone/{args.model_name}/Detection/Result.jpg"
    if args.model_name == "YOLOv10n_Slim": 
        args.model_architecture = YOLOv10n_Slim_Architecture
    elif args.model_name == "YOLOv10n": 
        args.model_architecture = YOLOv10n_Architecture
    return args


def Cal_WB_Address(Weight_Size, Bias_Size): 
    Weight_Size = [weight * 4 for weight in Weight_Size]
    Bias_Size = [bias * 4 for bias in Bias_Size]
    Weight_Address = []; Bias_Address = []
    weight_address = DRAM_BASE["WEIGHT_BASE"]
    bias_address = DRAM_BASE["PARAM_BASE"]
    for weight, bias in zip(Weight_Size, Bias_Size): 
        Weight_Address.append(weight_address)
        Bias_Address.append(bias_address)
        weight_address += weight
        bias_address += bias
    
    return Weight_Size, Bias_Size, Weight_Address, Bias_Address


def Cal_Activation_Address(args): 
    if args.model_name == "YOLOv10n_Slim": 
        address = YOLOv10n_Slim_Address_Map
    elif args.model_name == "YOLOv10n":
        address = YOLOv10n_Address_Map
    arch = args.model_architecture
    for layer_name in arch:
        arch[layer_name]["Input_Address"] = address[layer_name]["Input_Address"]
        arch[layer_name]["Output_Address"] = address[layer_name]["Output_Address"]
    
    return arch


def Arch_MicroGen(args, arch, Weight_Size, Bias_Size, 
                  Weight_Address, Bias_Address, H_scale): 
    for layer, details in arch.items():
        ops = details["op"]
        hscales = details["hscale"]
        input_address = details["Input_Address"]
        output_address = details["Output_Address"]
        width = details["Width"]
        height = details["Height"]
        if args.DEBUG: print(f"\nLayer: {layer}")
        for idx, op in enumerate(ops): 
            hscale_idx = hscales[idx] if idx < len(hscales) else None
            in_addr = input_address[idx] if idx < len(input_address) else None
            out_addr = output_address[idx] if idx < len(output_address) else None
            w = width[idx] if idx < len(width) else None
            h = height[idx] if idx < len(height) else None
            if op.startswith(("ConvActMax", "ConvAct", "Conv")):
                conv_idx = re.search(r'\d+', op)
                conv_idx = int(conv_idx.group()) if conv_idx else "N/A"
                if args.DEBUG_WRITE: 
                    if op == "ConvActMax0": 
                        with open(file_path, "a+") as fptr:
                            fptr.write("# ConvActMax0\n")
                    else: 
                        with open(file_path, "a+") as fptr:
                            fptr.write(f"# {op}\n")
                if args.model_name == "YOLOv10n_Slim": 
                    if op == "Conv15_Head1": 
                        Weight_Addr = Weight_Address[conv_idx] + Weight_Size[conv_idx] // 2
                        Bias_Addr = Bias_Address[conv_idx] + Bias_Size[conv_idx] // 2
                    else: 
                        Weight_Addr = Weight_Address[conv_idx]
                        Bias_Addr = Bias_Address[conv_idx]
                if args.model_name == "YOLOv10n": 
                    if op == "Conv33_Head1": 
                        Weight_Addr = Weight_Address[conv_idx] + Weight_Size[conv_idx] // 2
                        Bias_Addr = Bias_Address[conv_idx] + Bias_Size[conv_idx] // 2
                    else: 
                        Weight_Addr = Weight_Address[conv_idx]
                        Bias_Addr = Bias_Address[conv_idx]
                Conv_MicroGen(conv_idx, model, op, hscale_idx, H_scale, 
                              Weight_Addr, Bias_Addr,
                              in_addr, out_addr, w, h)
                if args.DEBUG: print(f"{idx}, {op}")
            elif op.startswith("Upsample"):
                if args.DEBUG_WRITE: 
                    with open(file_path, "a+") as fptr:
                        fptr.write(f"# {op}\n")
                Upsample_MicroGen(in_addr, out_addr, w, h)
                if args.DEBUG: print(f"{idx}, {op}")
            elif op.startswith("MaxPool"):
                if args.DEBUG_WRITE: 
                    with open(file_path, "a+") as fptr:
                        fptr.write(f"# {op}\n")
                MaxPool_MicroGen(in_addr, out_addr, w, h)
                if args.DEBUG: print(f"{idx}, {op}")
            elif op.startswith("EWAdder"):
                if args.DEBUG_WRITE: 
                    with open(file_path, "a+") as fptr:
                        fptr.write(f"# {op}\n")
                EWAdder_MicroGen(args, layer, in_addr, out_addr, w, h)
                if args.DEBUG: print(f"{idx}, {op}")
            elif op.startswith("AttnHead"):
                if args.DEBUG_WRITE: 
                    with open(file_path, "a+") as fptr:
                        fptr.write(f"# {op}\n")
                Attention_MicroGen(H_scale, hscale_idx, in_addr, out_addr, w, h)
                if args.DEBUG: print(f"{idx}, {op}")
    ctrl_write(OPCODE["FINISH"])
    

def Arch_ScriptGen(args, arch, Weight_Size, Bias_Size, 
                   Weight_Address, Bias_Address, H_scale): 
    for layer, details in arch.items():
        ops = details["op"]
        hscales = details["hscale"]
        input_address = details["Input_Address"]
        output_address = details["Output_Address"]
        width = details["Width"]
        height = details["Height"]
        with open(script_path, "a+") as f:
            f.write("\n########################\n")
            f.write(f"#        {layer}        #\n")
            f.write("########################\n")
        for idx, op in enumerate(ops): 
            hscale_idx = hscales[idx] if idx < len(hscales) else None
            in_addr = input_address[idx] if idx < len(input_address) else None
            out_addr = output_address[idx] if idx < len(output_address) else None
            w = width[idx] if idx < len(width) else None
            h = height[idx] if idx < len(height) else None
            if op.startswith(("ConvActMax", "ConvAct", "Conv")):
                conv_idx = re.search(r'\d+', op)
                conv_idx = int(conv_idx.group()) if conv_idx else "N/A"
                if op == "ConvActMax0": 
                    with open(script_path, "a+") as fptr:
                        fptr.write("# ConvActMax0\n")
                else: 
                    with open(script_path, "a+") as fptr:
                        fptr.write(f"# {op}\n")
                if args.model_name == "YOLOv10n_Slim": 
                    if op == "Conv15_Head1": 
                        Weight_Addr = Weight_Address[conv_idx] + Weight_Size[conv_idx] // 2
                        Bias_Addr = Bias_Address[conv_idx] + Bias_Size[conv_idx] // 2
                    else: 
                        Weight_Addr = Weight_Address[conv_idx]
                        Bias_Addr = Bias_Address[conv_idx]
                if args.model_name == "YOLOv10n": 
                    if op == "Conv33_Head1": 
                        Weight_Addr = Weight_Address[conv_idx] + Weight_Size[conv_idx] // 2
                        Bias_Addr = Bias_Address[conv_idx] + Bias_Size[conv_idx] // 2
                    else: 
                        Weight_Addr = Weight_Address[conv_idx]
                        Bias_Addr = Bias_Address[conv_idx]
                Conv_ScriptGen(conv_idx, model, op, hscale_idx, H_scale, 
                              Weight_Addr, Bias_Addr,
                              in_addr, out_addr, w, h, script_path)
            elif op.startswith("Upsample"):
                with open(script_path, "a+") as fptr:
                    fptr.write(f"# {op}\n")
                Upsample_ScriptGen(in_addr, out_addr, w, h, script_path)
            elif op.startswith("MaxPool"):
                with open(script_path, "a+") as fptr:
                    fptr.write(f"# {op}\n")
                MaxPool_ScriptGen(in_addr, out_addr, w, h, script_path)
            elif op.startswith("EWAdder"):
                with open(script_path, "a+") as fptr:
                    fptr.write(f"# {op}\n")
                EWAdder_ScriptGen(args, layer, in_addr, out_addr, w, h, script_path)
            elif op.startswith("AttnHead"): 
                with open(script_path, "a+") as fptr:
                    fptr.write(f"# {op}\n")
                Attention_ScriptGen(H_scale, hscale_idx, in_addr, out_addr, w, h, script_path)
            elif op.startswith("Concat"):
                with open(script_path, "a+") as fptr:
                    fptr.write(f"# Concatenation\n")
    with open(script_path, mode="a+") as script: 
        script.write('ctrl_write(OPCODE["FINISH"])')
    

def MicroV2Convertor(args): 
    Extract_MicroParams(args)

if __name__ == "__main__":
    # Define Parse input arguments
    args = parse_args()
    
    # Load Quantized Model 
    print(colored(f"--> 🔹 Loading Quantized {args.model_name} Model & Architecture ...", color="cyan"))
    if args.model_name == "YOLOv10n_Slim":
        YOLOv10_Detector = YOLOv10n_Detector(ch=(256, 128, 256))
        model = YOLOv10n_Slim(YOLOv10_Detector) 
    elif args.model_name == "YOLOv10n":
        YOLOv10_Detector = YOLOv10n_Detector(ch=(64, 128, 256))
        model = YOLOv10n(YOLOv10_Detector) 
    model.eval().cuda()

    # Quantization Inference
    print(colored("--> 🔹 Extracting Quantized Model Parameters ...", color="cyan"))
    print(colored("--> 🔹 Visualizing Quantized Model Inference Result ...", color="magenta"))
    if args.model_name == "YOLOv10n_Slim":
        Weight_Size, Bias_Size, H_scale = Inference_YOLOv10n_Slim(args, model)
    if args.model_name == "YOLOv10n":
        Weight_Size, Bias_Size, H_scale = Inference_YOLOv10n(args, model)

    # Calculating Weight and Bias Address, New Weight_Size and Bias_Size
    print(colored("--> 🔹 Calculating Weight and Bias Address ...", color="magenta"))
    Weight_Size, Bias_Size, Weight_Address, Bias_Address = Cal_WB_Address(Weight_Size, Bias_Size)
    
    # Calculating Activation Address
    print(colored("--> 🔹 Calculating Activation Address ...", color="magenta"))
    Architecture = Cal_Activation_Address(args)
    
    # Extracting the Model Architecture Operations
    print(colored("--> 🔹 Extracting Model Operations ...", color="red"))
    
    # Generating MicroScriptV1
    print(colored("--> 🔹 Generating MicroScriptV1 ...", color="yellow"))
    Arch_ScriptGen(args, Architecture, Weight_Size, Bias_Size, Weight_Address, Bias_Address, H_scale)
    
    # Generating MicrocodeV1
    print(colored("--> 🔹 Generating MicrocodeV1 ...", color="yellow"))
    Arch_MicroGen(args, Architecture, Weight_Size, Bias_Size, Weight_Address, Bias_Address, H_scale)
    
    # Generating MicrocodeV2
    print(colored("--> 🔹 Generating MicrocodeV2 ...", color="yellow"))
    MicroV2Convertor(args)
    print(colored("--> 🔹 Process is Done!\n", color="green"))
    
    # DRAM
    Headers = ["Categories", "Address"]
    if args.model_name == "YOLOv10n_Slim":
        DRAM_INFO = [["Weight Start", f"0x{hex(Weight_Address[0])[2:].zfill(8)}"], 
                    ["Weight End", f"0x{hex(Weight_Address[-1] + (Weight_Size[-1]))[2:].zfill(8)}"],
                    ["Params Start", f"0x{hex(Bias_Address[0])[2:].zfill(8)}"],
                    ["Params End", f"0x{hex(Bias_Address[-1] + (Bias_Size[-1]))[2:].zfill(8)}"],
                    ["Data Start", hex(DRAM_BASE["INPUT_BASE"])],
                    ["Data End", f"0x{hex(CONV21_OUTPUT + (19200 * 4))[2:].zfill(8)}"]]
    if args.model_name == "YOLOv10n":
        DRAM_INFO = [["Weight Start", f"0x{hex(Weight_Address[0])[2:].zfill(8)}"], 
                    ["Weight End", f"0x{hex(Weight_Address[-1] + (Weight_Size[-1]))[2:].zfill(8)}"],
                    ["Params Start", f"0x{hex(Bias_Address[0])[2:].zfill(8)}"],
                    ["Params End", f"0x{hex(Bias_Address[-1] + (Bias_Size[-1]))[2:].zfill(8)}"],
                    ["Data Start", hex(DRAM_BASE["INPUT_BASE"])],
                    ["Data End", f"0x{hex(CONV52_OUTPUT + (19200 * 4))[2:].zfill(8)}"]]
    print(tabulate(DRAM_INFO, Headers, tablefmt="grid"))
    print()
    