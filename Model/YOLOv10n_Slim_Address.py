from MSIS_NPU_Instruction_SetV1.DRAM_Config import DRAM_BASE

# DEBUG - Boolean
DEBUG = False
DEBUG_WRITE = False
DEBUG_ADDR = False
DEBUG_COMPILER = True

# Size of Reordered Weight
Weight_Size = [144, 192, 576, 576, 576, 4608, 3072, 288, 73728, 12288, 49152, 24576, 24576, 
               12288, 6144, 288, 12288, 12288, 73728, 73728, 36864, 36864, 288, 12288, 
               576, 12288, 288]

# Each Line has 4-data
Weight_Size = [weight * 4 for weight in Weight_Size]
if DEBUG: print(f"Weight_Size: {Weight_Size}")
if DEBUG: print(f"Weight_Size Length: {len(Weight_Size)}\n")

# Size of Reordered Bias
Bias_Size = [8, 16, 16, 8, 8, 32, 64, 64, 128, 64, 128, 128, 128, 128, 64, 
             64, 128, 64, 128, 64, 128, 128, 64, 128, 128, 64, 64]

# Each Line has 4-data
Bias_Size = [bias * 4 for bias in Bias_Size]
if DEBUG: print(f"Bias_Size: {Bias_Size}")
if DEBUG: print(f"Bias_Size Length: {len(Bias_Size)}\n")

# Calculate Weight & Bias Address
Weight_Address = []; Bias_Address = []
weight_address = DRAM_BASE["WEIGHT_BASE"]
bias_address = DRAM_BASE["PARAM_BASE"]
for weight, bias in zip(Weight_Size, Bias_Size): 
    Weight_Address.append(weight_address)
    Bias_Address.append(bias_address)
    weight_address += weight
    bias_address += bias

if DEBUG_ADDR:  
    print(f"Weight_Address: {', '.join(['0x' + hex(weight)[2:].zfill(8) for weight in Weight_Address])}")
    print(f"Weight_Address Length: {len(Weight_Address)}\n")
    print(f"Bias_Address: {', '.join(['0x' + hex(bias)[2:].zfill(8) for bias in Bias_Address])}")
    print(f"Bias_Address Length: {len(Bias_Address)}\n")

# Activation Address
YOLOv10n_Slim_activation_address = {}
for i in range(12):
    YOLOv10n_Slim_activation_address[f"layer{i}"] = {}
    
# Function to append the operation if it doesn't exist
def Activation_Address(layer, operation_name, input_value, output_value):
    # Initialize the operation if not already present
    if operation_name not in YOLOv10n_Slim_activation_address[layer]:
        YOLOv10n_Slim_activation_address[layer][operation_name] = {
            "Input_Address": [],
            "Output_Address": []
        }
    # Append the input and output to the operation
    YOLOv10n_Slim_activation_address[layer][operation_name]["Input_Address"].extend(input_value)
    YOLOv10n_Slim_activation_address[layer][operation_name]["Output_Address"].extend(output_value)

##################################
#             Layer0             #
##################################
## ConvActMax0 
CONV0_INPUT  = DRAM_BASE["INPUT_BASE"]
CONV0_OUTPUT = CONV0_INPUT + (307200 * 4)
Activation_Address("layer0", "ConvActMax0", [CONV0_INPUT], [CONV0_OUTPUT])

if DEBUG_ADDR:
    print(f"layer0 --> ConvActMax0:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[0])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[0] + (144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV0_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV0_OUTPUT)[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[0])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[0] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV0_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer1             #
##################################
## ConvAct1 -> Branch 
CONV1_INPUT   = CONV0_OUTPUT
CONV1_OUTPUT0 = CONV0_OUTPUT + (76800 * 4)
CONV1_OUTPUT1 = CONV1_OUTPUT0 + (76800 * 2 * 4) # After CONV1_OUTPUT0(Split0 + Split1)

if DEBUG_ADDR:
    print(f"Layer1 --> ConvAct1:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[1])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[1] + (192 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV1_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV1_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[1])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[1] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV1_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV1_OUTPUT0 + (76800 * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV1_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV1_OUTPUT1 + (76800 * 2 * 4))[2:].zfill(8)}\n")

# For EWAdder --> Concat: 
EWADDER0_OUTPUT = CONV1_OUTPUT1 + (76800 * 2 * 4) # After CONV1_OUTPUT1(Split0 + Split1)
Activation_Address("layer1", "ConvAct1", [CONV1_INPUT], [CONV1_OUTPUT0, CONV1_OUTPUT1])

## ConvAct3
CONV3_INPUT  = CONV1_OUTPUT0 + (76800 * 4) # After CONV1_OUTPUT0(Split0) --> start CONV1_OUTPUT0(Split1)
CONV3_OUTPUT = EWADDER0_OUTPUT + (76800 * 4) # After EWADDER0_OUTPUT
Activation_Address("layer1", "ConvAct3", [CONV3_INPUT], [CONV3_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer1 --> ConvAct3:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[3])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[3] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV3_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV3_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[3])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[3] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV3_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

## ConvAct4
CONV4_INPUT  = CONV3_OUTPUT
CONV4_OUTPUT = CONV4_INPUT + (76800 * 4)
Activation_Address("layer1", "ConvAct4", [CONV4_INPUT], [CONV4_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer1 --> ConvAct4:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[4])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[4] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV4_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV4_INPUT+ (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[4])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[4] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV4_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV4_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

## EWAdder0
EWADDER0_INPUT1 = CONV4_OUTPUT
EWADDER0_INPUT2 = CONV1_OUTPUT1 + (76800 * 4) # After CONV1_OUTPUT1(Split0) --> start CONV1_OUTPUT1(Split1)
Activation_Address("layer1", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer1 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

## Concat0
# --> Conv1-Split0 + Conv1-Split1 + EWAdder0-Output
# -- > CONV1_OUTPUT1
# Output --> (1, 48, 160, 120)
Activation_Address("layer1", "Concat0", [None], [None])

## ConvAct2
CONV2_INPUT  = CONV1_OUTPUT1                    # Concat (Conv1-Split0 + Conv1-Split1 + EWAdder0-Output)   
CONV2_OUTPUT = CONV4_OUTPUT + (76800 * 4)
Activation_Address("layer1", "ConvAct2", [CONV2_INPUT], [CONV2_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer1 --> ConvAct2:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[2])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[2] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV2_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV2_INPUT + (230400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[2])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[2] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV2_OUTPUT + (153600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer2             #
##################################
## ConvAct5
CONV5_INPUT  = CONV2_OUTPUT
CONV5_OUTPUT = CONV2_OUTPUT + (153600 * 4)
Activation_Address("layer2", "ConvAct5", [CONV5_INPUT], [CONV5_OUTPUT])
if DEBUG_ADDR:
    print(f"layer2 --> ConvAct5:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[5])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[5] + (4608 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV5_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV5_INPUT + (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[5])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[5] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV5_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV5_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer3             #
##################################
## ConvAct6
CONV6_INPUT  = CONV5_OUTPUT
CONV6_OUTPUT = CONV6_INPUT + (76800 * 4)
Activation_Address("layer3", "ConvAct6", [CONV6_INPUT], [CONV6_OUTPUT])
if DEBUG_ADDR:
    print(f"layer3 --> ConvAct6:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[6])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[6] + (3072 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV6_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV6_INPUT+ (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[6])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[6] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV6_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV6_OUTPUT+ (153600 * 4))[2:].zfill(8)}\n")

## DepthWise-Conv7 - Branch
CONV7_INPUT  = CONV6_OUTPUT
CONV7_OUTPUT1 = CONV7_INPUT + (153600 * 4) # After CONV6_OUTPUT
CONV7_OUTPUT2 = CONV7_OUTPUT1 + (38400 * 4) # After CONV7_OUTPUT1
Activation_Address("layer3", "Conv7", [CONV7_INPUT], [CONV7_OUTPUT1, CONV7_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer3 --> Conv7:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[7])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[7] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV7_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV7_INPUT+ (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[7])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[7] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV7_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV7_OUTPUT1+ (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV7_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV7_OUTPUT2+ (38400 * 4))[2:].zfill(8)}\n")

##################################
#             Layer4             #
##################################
## ConvAct8
CONV8_INPUT  = CONV7_OUTPUT1
CONV8_OUTPUT = CONV7_OUTPUT2 + (38400 * 4)
Activation_Address("layer4", "ConvAct8", [CONV8_INPUT], [CONV8_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct8:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[8])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[8] + (73728 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV8_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV8_INPUT+ (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[8])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[8] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV8_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV8_OUTPUT+ (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer5             #
##################################
## ConvAct9 
CONV9_INPUT  = CONV8_OUTPUT
CONV9_OUTPUT = CONV9_INPUT + (19200 * 4)
Activation_Address("layer5", "ConvAct9", [CONV9_INPUT], [CONV9_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> ConvAct9:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[9])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[9] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV9_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV9_INPUT+ (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[9])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[9] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV9_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV9_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## MaxPool1
MAXPOOL1_INPUT  = CONV9_OUTPUT
MAXPOOL1_OUTPUT = MAXPOOL1_INPUT + (9600 * 4)
Activation_Address("layer5", "MaxPool1", [MAXPOOL1_INPUT], [MAXPOOL1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> MaxPool1:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL1_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL1_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL1_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## MaxPool2
MAXPOOL2_INPUT  = MAXPOOL1_OUTPUT
MAXPOOL2_OUTPUT = MAXPOOL2_INPUT + (9600 * 4)
Activation_Address("layer5", "MaxPool2", [MAXPOOL2_INPUT], [MAXPOOL2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> MaxPool2:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL2_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL2_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL2_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## MaxPool3
MAXPOOL3_INPUT  = MAXPOOL2_OUTPUT
MAXPOOL3_OUTPUT = MAXPOOL3_INPUT + (9600 * 4)
Activation_Address("layer5", "MaxPool3", [MAXPOOL3_INPUT], [MAXPOOL3_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> MaxPool3:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL3_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL3_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL3_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## Concat0
# Conv9-Output + MaxPool1-Out + MaxPool2-Out + MaxPool3-Out
Activation_Address("layer5", "Concat0", [None], [None])

## ConvAct10
CONV10_INPUT  = CONV9_OUTPUT
CONV10_OUTPUT = MAXPOOL3_OUTPUT + (9600 * 4)
Activation_Address("layer5", "ConvAct10", [CONV10_INPUT], [CONV10_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> ConvAct10:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[10])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[10] + (49152 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV10_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV10_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[10])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[10] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV10_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV10_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer6             #
##################################
## ConvAct11 - Branch
CONV11_INPUT   = CONV10_OUTPUT
CONV11_OUTPUT1 = CONV11_INPUT + (19200 * 4)      # --> Conv13
CONV11_OUTPUT2 = CONV11_OUTPUT1 + (9600 * 2 * 4) # After CONV11_OUTPUT1(Split0 + Split1) --> EWAdder1
CONV11_OUTPUT3 = CONV11_OUTPUT2 + (9600 * 2 * 4) # After CONV11_OUTPUT2(Split0 + Split1) --> EWAdder2 (EWAdderi0)
CONV11_OUTPUT4 = CONV11_OUTPUT3 + (9600 * 2 * 4) # After CONV11_OUTPUT3(Split0 + Split1) --> Split0 --> Concat

# Concat:
EWADDER3_OUTPUT = CONV11_OUTPUT4 + (9600 * 4) # After CONV11_OUTPUT4(Split0) overwrite CONV11_OUTPUT4(Split1)
Activation_Address("layer6", "ConvAct11", [CONV11_INPUT], [CONV11_OUTPUT1, CONV11_OUTPUT2, CONV11_OUTPUT3, CONV11_OUTPUT4])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct11:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[11])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[11] + (24576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV11_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV11_INPUT+ (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[11])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[11] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV11_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV11_OUTPUT1+ (9600 * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV11_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV11_OUTPUT2+ (9600 * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT3: START: 0x{hex(CONV11_OUTPUT3)[2:].zfill(8)}\t END: 0x{hex(CONV11_OUTPUT3+ (9600 * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT4: START: 0x{hex(CONV11_OUTPUT4)[2:].zfill(8)}\t END: 0x{hex(CONV11_OUTPUT4+ (9600 * 2 * 4))[2:].zfill(8)}\n")

## ConvAct13 - Branch
CONV13_INPUT   = CONV11_OUTPUT1 + (9600 * 4)                        # After CONV11_OUTPUT1(Split0) --> CONV11_OUTPUT1(Split1)
CONV13_OUTPUT1 = EWADDER3_OUTPUT + (9600 * 4)                       # After EWADDER3_OUTPUT --> q,k,v (brach0)v
CONV13_OUTPUT2 = CONV13_OUTPUT1 + ((2400 + 2400 + 4800) * 2 * 4)    # After CONV13_OUTPUT1 (2-Heads) --> q,k,v (branch1) --> Conv15
Activation_Address("layer6", "ConvAct13", [CONV13_INPUT], [CONV13_OUTPUT1, CONV13_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct13:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[13])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[13] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV13_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV13_INPUT+ (9600 * 4) )[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[13])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[13] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV13_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV13_OUTPUT1+ ((2400 + 2400 + 4800) * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV13_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV13_OUTPUT2+ ((2400 + 2400 + 4800) * 2 * 4))[2:].zfill(8)}\n")

# Attention: MatMul0 & MatMul1
# Matrix1 (32x300 x 300x32)
##########Attn_Head0############################
MATMUL0_INPUT1 = CONV13_OUTPUT1                                       # Query-Head0
MATMUL0_INPUT2 = MATMUL0_INPUT1 + (2400 * 4)                          # Key-Head0
MATMUL0_INPUT3 = MATMUL0_INPUT2 + (2400 * 4)                          # Value-Head0
MATMUL0_OUTPUT = CONV13_OUTPUT2 + ((2400 + 2400 + 4800) * 2 * 4)      # After CONV13_OUTPUT2 (2-Heads) q,k,v (branch1)
Activation_Address("layer6", "AttnHead0", [MATMUL0_INPUT1, MATMUL0_INPUT2, MATMUL0_INPUT3], [MATMUL0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> AttnHead0:") 
    print(f"\tLD_WGT: START: 0x{hex(MATMUL0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT1+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(MATMUL0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT2+ (2400 * 4) )[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(MATMUL0_INPUT3)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT3+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MATMUL0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_OUTPUT+ (4800 * 4))[2:].zfill(8)}\n")

##########Attn_Head1############################
MATMUL1_INPUT1 = MATMUL0_INPUT3 + (4800 * 4)   # Query-Head1
MATMUL1_INPUT2 = MATMUL1_INPUT1 + (2400 * 4)   # Key-Head1
MATMUL1_INPUT3 = MATMUL1_INPUT2 + (2400 * 4)   # Value-Head1
MATMUL1_OUTPUT = MATMUL0_OUTPUT + (4800 * 4)  # After MATMUL0_OUTPUT

# Concat: MATMUL0_OUTPUT + MATMUL1_OUTPUT
Activation_Address("layer6", "AttnHead1", [MATMUL1_INPUT1, MATMUL1_INPUT2, MATMUL1_INPUT3], [MATMUL1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> AttnHead1:") 
    print(f"\tLD_WGT: START: 0x{hex(MATMUL1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT1+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(MATMUL1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT2+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(MATMUL1_INPUT3)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT3+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MATMUL1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_OUTPUT+ (4800 * 4))[2:].zfill(8)}\n")

## DepthWise-Conv15 -- Head0
CONV15_INPUT  = CONV13_OUTPUT2 + ((2400 + 2400) * 4) # v-Head0 -- Branch2
CONV15_OUTPUT_Head0 = MATMUL1_OUTPUT + (4800 * 4)
Activation_Address("layer6", "Conv15_Head0", [CONV15_INPUT], [CONV15_OUTPUT_Head0])
if DEBUG_ADDR:
    print(f"layer6 --> Conv15_Head0:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[15])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[15]+ ((288 // 2) * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV15_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV15_INPUT+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[15])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[15]+ ((64 // 2) * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV15_OUTPUT_Head0)[2:].zfill(8)}\t END: 0x{hex(CONV15_OUTPUT_Head0 + ((9600 // 2) * 4))[2:].zfill(8)}\n")
    
## DepthWise-Conv15 -- Head1
CONV15_INPUT  = CONV13_OUTPUT2 + ((2400 + 2400 + 4800) + (2400 + 2400)) * 4 # v-Head1 -- Branch2
CONV15_OUTPUT_Head1 = CONV15_OUTPUT_Head0 + (9600 // 2) * 4 # (1, 64, 20, 15) -- Head1
Activation_Address("layer6", "Conv15_Head1", [CONV15_INPUT], [CONV15_OUTPUT_Head1])
if DEBUG_ADDR:
    print(f"layer6 --> Conv15_Head1:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[15]+ ((288 // 2) * 4))[2:].zfill(8)}\t END: 0x{hex(Weight_Address[15] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV15_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV15_INPUT+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[15]+ ((64 // 2) * 4))[2:].zfill(8)}\t END: 0x{hex(Bias_Address[15] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV15_OUTPUT_Head1)[2:].zfill(8)}\t END: 0x{hex(CONV15_OUTPUT_Head1+((9600 // 2) * 4))[2:].zfill(8)}\n")
    
# Additional Concat CONV15_OUTPUT (Head0+1)

## EWAdder0
EWADDER0_INPUT1 = MATMUL0_OUTPUT                   # CONV15_OUTPUT_Head0 + CONV15_OUTPUT_Head1
EWADDER0_INPUT2 = CONV15_OUTPUT_Head0              # MATMUL0_OUTPUT + MATMUL1_OUTPUT
EWADDER0_OUTPUT = CONV15_OUTPUT_Head0 + (9600 * 4) # (1, 128, 20, 15) -- After Head0 + Head1
Activation_Address("layer6", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## Conv14 - Branch
CONV14_INPUT  = EWADDER0_OUTPUT
CONV14_OUTPUT1 = CONV14_INPUT + (9600 * 4)
CONV14_OUTPUT2 = CONV14_OUTPUT1 + (9600 * 4) # After CONV14_OUTPUT1
Activation_Address("layer6", "Conv14", [CONV14_INPUT], [CONV14_OUTPUT1, CONV14_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer6 --> Conv14:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[14])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[14] + (6144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV14_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV14_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[14])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[14] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV14_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV14_OUTPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV14_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV14_OUTPUT2+ (9600 * 4))[2:].zfill(8)}\n")

## EWAdder1
EWADDER1_INPUT1 = CONV11_OUTPUT2 + (9600 * 4) # CONV11_OUTPUT2 (Split1)
EWADDER1_INPUT2 = CONV14_OUTPUT1
EWADDER1_OUTPUT = CONV14_OUTPUT2 + (9600 * 4) # After CONV14_OUTPUT2
Activation_Address("layer6", "EWAdder1", [EWADDER1_INPUT1, EWADDER1_INPUT2], [EWADDER1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder1:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT2+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## ConvAct16
CONV16_INPUT  = EWADDER1_OUTPUT
CONV16_OUTPUT = CONV16_INPUT + (9600 * 4)
Activation_Address("layer6", "ConvAct16", [CONV16_INPUT], [CONV16_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct16:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[16])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[16] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV16_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV16_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[16])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[16] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV16_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV16_OUTPUT+ (19200 * 4))[2:].zfill(8)}\n")

## Conv17
CONV17_INPUT  = CONV16_OUTPUT
CONV17_OUTPUT = CONV17_INPUT + (19200 * 4)
Activation_Address("layer6", "Conv17", [CONV17_INPUT], [CONV17_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> Conv17:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[17])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[17] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV17_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV17_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[17])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[17] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV17_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV17_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

## EWAdder2 (EWAdderi0)
EWADDER2_INPUT1 = CONV11_OUTPUT3 + (9600 * 4) # CONV11_OUTPUT3 (Split1)
EWADDER2_INPUT2 = CONV14_OUTPUT2
EWADDER2_OUTPUT = CONV17_OUTPUT + (9600 * 4)
Activation_Address("layer6", "EWAdder2", [EWADDER2_INPUT1, EWADDER2_INPUT2], [EWADDER2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder2:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER2_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT1 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER2_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## EWAdder3
EWADDER3_INPUT1 = EWADDER2_OUTPUT
EWADDER3_INPUT2 = CONV17_OUTPUT
Activation_Address("layer6", "EWAdder3", [EWADDER3_INPUT1, EWADDER3_INPUT2], [EWADDER3_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder3:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER3_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER3_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_INPUT2+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

# Concatenation
# Conv11-out + EWAdder3 
Activation_Address("layer6", "Concat0", [None], [None])

## ConvAct12
CONV12_INPUT  = CONV11_OUTPUT4                      # Concat-output (Conv11-split0 + EWAdder3)
CONV12_OUTPUT = EWADDER2_OUTPUT + (9600 * 4)

# Concat
CONV19_OUTPUT = CONV12_OUTPUT + (19200 * 4)
Activation_Address("layer6", "ConvAct12", [CONV12_INPUT], [CONV12_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct12:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[12])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[12] + (24576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV12_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV12_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[12])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[12] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV12_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV12_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer7             #
##################################
## Upsample (Resize) microcode
UPSAMPLE_INPUT  = CONV7_OUTPUT2
UPSAMPLE_OUTPUT = CONV19_OUTPUT + (9600 * 4)
Activation_Address("layer7", "Upsample0", [UPSAMPLE_INPUT], [UPSAMPLE_OUTPUT])
if DEBUG_ADDR:
    print(f"layer7 --> Upsample0:") 
    print(f"\tLD_IN1: START: 0x{hex(UPSAMPLE_INPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE_INPUT+(38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(UPSAMPLE_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE_OUTPUT+ (153600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer8             #
##################################
## ConvAct18
CONV18_INPUT  = UPSAMPLE_OUTPUT
CONV18_OUTPUT = CONV18_INPUT + (153600 * 4)
Activation_Address("layer8", "ConvAct18", [CONV18_INPUT], [CONV18_OUTPUT])
if DEBUG_ADDR:
    print(f"layer8 --> ConvAct18:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[18])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[18] + (73728 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV18_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV18_INPUT)[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[18])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[18] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV18_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV18_OUTPUT)[2:].zfill(8)}\n")

##################################
#             Layer9             #
##################################
## ConvActMax19
CONV19_INPUT  = CONV18_OUTPUT
Activation_Address("layer9", "ConvActMax19", [CONV19_INPUT], [CONV19_OUTPUT])
if DEBUG_ADDR:
    print(f"layer9 --> ConvActMax19:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[19])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[19] + (73728 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV19_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV19_INPUT+ (307200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[19])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[19] + (64 *4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV19_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV19_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer10            #
##################################
# Concatenation
# Conv12-out + Conv19-out
Activation_Address("layer10", "Concat0", [None], [None])

##################################
#             Layer11            #
##################################
## Conv20 - Branch
CONV20_INPUT1  = CONV12_OUTPUT
CONV20_OUTPUT1 = CONV18_OUTPUT + (307200 * 4)    # After CONV18_OUTPUT
CONV20_OUTPUT2 = CONV20_OUTPUT1 + (9600 * 2 * 4) # After CONV20_OUTPUT1 (Split0 + Split1)

# Concat
EWADDER0_OUTPUT = CONV20_OUTPUT2 + (9600 * 2 * 4) # After CONV20_OUTPUT2 (Split0 + Split1)
Activation_Address("layer11", "ConvAct20", [CONV20_INPUT1], [CONV20_OUTPUT1, CONV20_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer11 --> Conv20:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[20])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[20] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV20_INPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV20_INPUT1 + (28800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[20])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[20] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV20_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV20_OUTPUT1+ (9600 * 2 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV20_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV20_OUTPUT2+ (9600 * 2 * 4))[2:].zfill(8)}\n")

## DepthWise-Conv22
CONV22_INPUT  = CONV20_OUTPUT1 + (9600 * 4)
CONV22_OUTPUT = EWADDER0_OUTPUT + (9600 * 4) # After EWADDER0_OUTPUT
Activation_Address("layer11", "ConvAct22", [CONV22_INPUT], [CONV22_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> Conv22:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[22])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[22] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV22_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV22_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[22])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[22] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV22_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV22_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

## ConvAct23
CONV23_INPUT  = CONV22_OUTPUT
CONV23_OUTPUT = CONV23_INPUT + (9600 * 4) 
Activation_Address("layer11", "ConvAct23", [CONV23_INPUT], [CONV23_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> ConvAct23:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[23])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[23] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV23_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV23_INPUT+ (9600 * 4) )[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[23])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[23] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV23_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV23_OUTPUT+ (19200 * 4) )[2:].zfill(8)}\n")

## DepthWise-ConvAct24
CONV24_INPUT  = CONV23_OUTPUT
CONV24_OUTPUT = CONV24_INPUT + (19200 * 4)  
Activation_Address("layer11", "ConvAct24", [CONV24_INPUT], [CONV24_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> ConvAct24:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[24])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[24] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV24_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV24_INPUT+ (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[24])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[24] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV24_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV24_OUTPUT+ (19200 * 4))[2:].zfill(8)}\n")

## ConvAct25
CONV25_INPUT  = CONV24_OUTPUT
CONV25_OUTPUT = CONV25_INPUT + (19200 * 4)  
Activation_Address("layer11", "ConvAct25", [CONV25_INPUT], [CONV25_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> ConvAct25:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[25])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[25] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV25_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV25_INPUT+ (19200 * 4) )[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[25])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[25] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV25_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV25_OUTPUT+ (9600 * 4) )[2:].zfill(8)}\n")

## DepthWise-Conv26
CONV26_INPUT  = CONV25_OUTPUT
CONV26_OUTPUT = CONV26_INPUT + (9600 * 4) 
Activation_Address("layer11", "ConvAct26", [CONV26_INPUT], [CONV26_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> Conv26:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[26])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[26] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV26_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV26_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[26])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[26] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV26_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV26_OUTPUT+ (9600 * 4) )[2:].zfill(8)}\n")

## EWAdder0
EWADDER0_INPUT1 = CONV20_OUTPUT2 + (9600 * 4)  # CONV20_OUTPUT2 (Split1) 
EWADDER0_INPUT2 = CONV26_OUTPUT
Activation_Address("layer11", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer11 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

## Concat0
# Conv20-split0 + Conv20-split1 + EWAdder0-out
Activation_Address("layer11", "Concat0", [None], [None])

## ConvAct21
CONV21_INPUT  = CONV20_OUTPUT2
CONV21_OUTPUT = CONV26_OUTPUT + (9600 * 4)   
Activation_Address("layer11", "ConvAct21", [CONV21_INPUT], [CONV21_OUTPUT])

if DEBUG_ADDR:
    print(f"layer11 --> ConvAct21:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[21])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[21] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV21_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV21_INPUT + (28800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[21])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[21] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV21_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV21_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

OUTPUT = CONV21_OUTPUT

####################################
if DEBUG_COMPILER: 
    YOLOv10n_Slim_Address_Map = {}
    for layer, ops in YOLOv10n_Slim_activation_address.items():
        input_list = []
        output_list = []
        for op, details in ops.items():
            input = details["Input_Address"]
            output = details["Output_Address"]
            if len(input) != 1: 
                input_list.append(input)
            else: 
                input_list.extend(input)
            if len(output) != 1:
                output_list.append(output)
            else: 
                output_list.extend(output)
        YOLOv10n_Slim_Address_Map[layer] = {
            "Input_Address":input_list,
            "Output_Address":output_list 
        }
        # print(f"\nLayer: {layer}, Input_Address: {input_list}")
        # print(f"Layer: {layer}, Output_Address: {output_list}")
    # print(YOLOv10n_Slim_Address_Map)