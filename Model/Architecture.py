# Abbreviation: 
# ConvActMax: Conv/DWConv + Act + MaxPool2x2S2
# ConvAct: Conv/DWConv + Act
# Conv: Conv/DWConv
# ConvAct_Branch: Conv/DWConv + Act + Existed Branch
# MaxPool: Replicated + MaxPool2x2S1
# EWAdder: Element-Wise Adder
# Attn_Head: Attention
# Concat: Concatenation
# Upsample: Upsampling

import re

# YOLOv10n
YOLOv10n_Architecture = {
    "layer0": {
        "op": ["ConvAct0"], 
        "hscale": [0],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [640], 
        "Width": [480]
    },
    "layer1": {
        "op": ["ConvAct1"], 
        "hscale": [1],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [320], 
        "Width": [240]
    },
    "layer2": {
        "op": ["ConvAct2", "ConvAct4", "ConvAct5", "EWAdder0", "Concat0", "ConvAct3"], 
        "hscale": [[2, 5], 3, 4, None, None, 8],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [160, 160, 160, 160, None, 160], 
        "Width": [120, 120, 120, 120, None, 120]
    },
    "layer3": {
        "op": ["ConvAct6"], 
        "hscale": [9],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [160], 
        "Width": [120]
    },
    "layer4": {
        "op": ["ConvAct7", "ConvAct9", "ConvAct10", "EWAdder0", "ConvAct11", "ConvAct12", "EWAdder1", "EWAdder2", "Concat0", "ConvAct8"], 
        "hscale": [[10, 13, 16], 11, [12, 15], None, 14, 17, None, None, None, [20, 68]],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [80, 80, 80, 80, 80, 80, 80, 80, 80, 80], 
        "Width": [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
    },
    "layer5": {
        "op": ["ConvAct13", "Conv14"], 
        "hscale": [21, 22],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [80, 80], 
        "Width": [60, 60]
    },
    "layer6": {
        "op": ["ConvAct15", "ConvAct17", "ConvAct18", "EWAdder0", "ConvAct19", "ConvAct20", "EWAdder1", "EWAdder2", "Concat0", "ConvAct16"], 
        "hscale": [[23, 26, 29], 24, [25, 28], None, 27, 30, None, None, None, [33, 61]],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40, 40, 40, 40, 40, 40, 40, 40, 40, 40], 
        "Width": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    },
    "layer7": {
        "op": ["ConvAct21", "Conv22"], 
        "hscale": [34, 35],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40, 40], 
        "Width": [30, 30]
    },
    "layer8": {
        "op": ["ConvAct23", "ConvAct25", "ConvAct26", "EWAdder0", "Concat0", "ConvAct24"], 
        "hscale": [[36, 39], 37, 38, None, None, 42],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [20, 20, 20, 20, None, 20], 
        "Width": [15, 15, 15, 15, None, 15]
    },
    "layer9": {
        "op": ["ConvAct27", "MaxPool1", "MaxPool2", "MaxPool3", "Concat0", "ConvAct28"], 
        "hscale": [43, None, None, None, None, 44],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [20, 20, 20, 20, None, 20], 
        "Width": [15, 15, 15, 15, None, 15]
    },
    "layer10": {
        "op": ["ConvAct29", "Conv31", "AttnHead0", "AttnHead1", "Conv33_Head0", "Conv33_Head1", "EWAdder0", "Conv32", 
               "EWAdder1", "ConvAct34", "Conv35", "EWAdder2", "EWAdder3", "Concat0", "ConvAct30"], 
        "hscale": [[45, 54, 56, 56], [46, 49], [48, 51], [48, 51], 52, 52, None, [53, 57],
                   None, 55, 58, None, None, None, [60, 85]],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, None, 20],
        "Width": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, None, 15] 
    },
    "layer11": {
        "op": ["Upsample0"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [20], 
        "Width": [15]
    },
    "layer12": {
        "op": ["Concat0"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [None], 
        "Width": [None]
    },
    "layer13": {
        "op": ["ConvAct36", "ConvAct38", "ConvAct39", "Concat0", "ConvAct37"], 
        "hscale": [[62, 65], 63, 66, None, [67, 76]],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40, 40, 40, None, 40], 
        "Width": [30, 30, 30, None, 30]
    },
    "layer14": {
        "op": ["Upsample1"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40], 
        "Width": [30]
    },
    "layer15": {
        "op": ["Concat0"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [None], 
        "Width": [None]
    },
    "layer16": {
        "op": ["ConvAct40", "ConvAct42", "ConvAct43", "Concat0", "ConvAct41"], 
        "hscale": [[69, 72], 70, 73, None, 74],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [80, 80, 80, None, 80], 
        "Width": [60, 60, 60, None, 60]
    },
    "layer17": {
        "op": ["ConvAct44"], 
        "hscale": [75],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [80], 
        "Width": [60]
    },
    "layer18": {
        "op": ["Concat0"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [None], 
        "Width": [None]
    },
    "layer19": {
        "op": ["ConvAct45", "ConvAct47", "ConvAct48", "Concat0", "ConvAct46"], 
        "hscale": [[77, 80], 78, 81, None, 82],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40, 40, 40, None, 40], 
        "Width": [30, 30, 30, None, 30]
    },
    "layer20": {
        "op": ["ConvAct49", "Conv50"], 
        "hscale": [83, 84],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [40, 40], 
        "Width": [30, 30]
    },
    "layer21": {
        "op": ["Concat0"], 
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [None], 
        "Width": [None]
    },
    "layer22": {
        "op": ["ConvAct51", "ConvAct53", "ConvAct54", "ConvAct55", "ConvAct56", "ConvAct57", "EWAdder0", "Concat0", "ConvAct52"], 
        "hscale": [[86, 91], 87, 88, 89, 90, 92, None, None, 95],
        "Input_Address": [],
        "Output_Address": [],
        "Height": [20, 20, 20, 20, 20, 20, 20, None, 20],
        "Width": [15, 15, 15, 15, 15, 15, 15, None, 15]
    },
}


# YOLOv10n_Slim
YOLOv10n_Slim_Architecture = {
    "layer0": {
        "op": ["ConvActMax0"], 
        "hscale": [0],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [480], 
        "Height": [640]
    },
    "layer1": {
        "op":["ConvAct1", "ConvAct3", "ConvAct4", "EWAdder0", "Concat0", "ConvAct2"], 
        "hscale": [[1, 4], 2, 3, None, None, 7],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [120, 120, 120, 120, None, 120], 
        "Height": [160, 160, 160, 160, None, 160]
    },
    "layer2": {
        "op": ["ConvAct5"],
        "hscale": [8],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [120], 
        "Height": [160]
    },
    "layer3": {
        "op": ["ConvAct6", "Conv7"],
        "hscale": [9, [10, 30]],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [60, 60], 
        "Height": [80, 80]
    },
    "layer4": {
        "op": ["ConvAct8"],
        "hscale": [11],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [30], 
        "Height": [40]
    },
    "layer5": {
        "op": ["ConvAct9", "MaxPool1", "MaxPool2", "MaxPool3", "Concat0", "ConvAct10"],
        "hscale": [12, None, None, None, None, 14],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [15, 15, 15, 15, None, 15], 
        "Height": [20, 20, 20, 20, None, 20]
    },
    "layer6": {
        "op": ["ConvAct11", "Conv13", "AttnHead0", "AttnHead1", "Conv15_Head0", "Conv15_Head1", "EWAdder0", 
               "Conv14", "EWAdder1", "ConvAct16", "Conv17", "EWAdder2", "EWAdder3", 
               "Concat0", "ConvAct12"],
        "hscale": [[15, 24, 26, 29], [16, 19], [18, 21], [18, 21], 22, 22, None,
                   [23, 27], None, 25, 28, None, None, None, 33],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, None, 15], 
        "Height": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, None, 20]
    },
    "layer7": {
        "op": ["Upsample0"],
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [40], 
        "Height": [30]
    },
    "layer8": {
        "op": ["ConvAct18"],
        "hscale": [31],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [60], 
        "Height": [80]
    },
    "layer9": {
        "op": ["ConvActMax19"],
        "hscale": [32],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [60], 
        "Height": [80]
    },
    "layer10": {
        "op": ["Concat0"],
        "hscale": [None],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [None], 
        "Height": [None]
    } ,
    "layer11": {
        "op": ["ConvAct20", "ConvAct22", "ConvAct23", "ConvAct24", "ConvAct25", 
                "ConvAct26", "EWAdder0", "Concat0", "ConvAct21"],
        "hscale": [[34, 39], 35, 36, 37, 38,
                   40, None, None, 43],
        "Input_Address": [],
        "Output_Address": [],
        "Width": [15, 15, 15, 15, 15, 15, 15, None, 15], 
        "Height": [20, 20, 20, 20, 20, 20, 20, None, 20]
    },
}


if __name__ == "__main__": 
    # Process each layer
    for layer, details in YOLOv10n_Slim_Architecture.items():
        ops = details["op"]
        hscales = details["hscale"]
        input_address = details["Input_Address"]
        output_address = details["Output_Address"]
        width = details["Width"]
        height = details["Height"]
        
        print(f"\nlayer: {layer}")
        
        for idx, op in enumerate(ops): 
            hscale = hscales[idx] if idx < len(hscales) else None
            in_addr = input_address[idx] if idx < len(input_address) else None
            out_addr = output_address[idx] if idx < len(output_address) else None
            w = width[idx] if idx < len(width) else None
            h = height[idx] if idx < len(height) else None
            # print(f"Ops: {op}, H_scale: {hscale_value}")
            if op.startswith(("ConvActMax", "ConvAct", "Conv")):
                conv_idx = re.search(r'\d+', op)
                conv_idx = int(conv_idx.group()) if conv_idx else "N/A"
                print(f"Ops: {op} --> index: {conv_idx}, H_scale: {hscale}, in_addr: 0x{str(in_addr)[2:].zfill(8)}, out_addr: 0x{str(out_addr)[2:].zfill(8)}, width: {w}, height: {h}")