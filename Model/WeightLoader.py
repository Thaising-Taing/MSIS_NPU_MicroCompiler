import torch 
import numpy as np
from tqdm import tqdm

def Load_Params_SSBQ(args):
    Integer_Params_Name_List = []
    FP_Params_Name_List = []
    Weight_Array = []
    Bias_Array = []
    # --------------- Load Integer for Quantized Model -------------
    Integer_Params_List = f"Parameters_zone/{args.model_name}/Parameter_Name/{args.model_name}_Quantized_Parameters.txt"
    with open(Integer_Params_List, mode="r") as f:
        Params_Name = f.readlines()
        for name in Params_Name:
            Integer_Params_Name_List.append(name.replace('\n', ''))     
    Path_in_W = f'Parameters_zone/{args.model_name}/Integer_parameters/'
    Path_in_B = f'Parameters_zone/{args.model_name}/Floating_parameters/'
    for name in Integer_Params_Name_List:
        if ".weight" in name:
            Weight = torch.from_numpy(np.load(Path_in_W + str(name) + '.npy'))
            Weight_Array.append(Weight)
        elif ".bias" in name: 
            Bias = torch.from_numpy(np.load(Path_in_B + str(name)+ '.npy'))
            Bias_Array.append(Bias)
    
    # --------------- Load FP for Detector Model -------------
    Integer_Params_List = f"Parameters_zone/{args.model_name}/Parameter_Name/{args.model_name}_Detector_Parameters.txt"
    with open(Integer_Params_List, mode="r") as f:
        Params_Name = f.readlines()
        for name in Params_Name:
            FP_Params_Name_List.append(name.replace('\n', ''))     
    Path_in = f'Parameters_zone/{args.model_name}/Floating_parameters/'
    for name in FP_Params_Name_List:
        if ".weight" in name:
            Weight = torch.from_numpy(np.load(Path_in + str(name) + '.npy'))
            Weight_Array.append(Weight)
        elif ".bias" in name: 
            Bias = torch.from_numpy(np.load(Path_in + str(name)+ '.npy'))
            Bias_Array.append(Bias)
            
    return Weight_Array, Bias_Array

def WeightLoader_SSBQ(args, model, Zpw, Scale_fmap): 
    weight_idx = 0
    bias_idx = 0
    Weight_Array, Bias_Array = Load_Params_SSBQ(args)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if name.startswith("detector"):
                    # Copied the Conv Weight
                    module.weight.copy_(Weight_Array[weight_idx])
                else: 
                    # Copied the Conv Weight
                    module.weight.copy_((Weight_Array[weight_idx] - Zpw[weight_idx]).cuda().clone().detach())
                    
                weight_idx += 1
                
                # Copied Conv Bias
                if module.bias is not None:
                    if name.startswith("detector"):
                        module.bias.copy_(Bias_Array[bias_idx])
                    else: 
                        module.bias.copy_(torch.round(torch.mul(Bias_Array[bias_idx], Scale_fmap[bias_idx])).view(-1).cuda().clone().detach())
                    
                    bias_idx += 1