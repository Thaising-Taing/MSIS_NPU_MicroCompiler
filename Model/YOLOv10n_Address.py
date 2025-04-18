# DEBUG - Boolean
DEBUG = False
DEBUG_WRITE = False
DEBUG_ADDR = False
DEBUG_COMPILER = True

# DRAM Base Address
WEIGHT_BASE = 0x00000000
PARAM_BASE  = 0x10000000
INPUT_BASE  = 0x01000000

# Size of Reordered Weight
Weight_Size = [144, 1152, 384, 576, 576, 576, 4608, 1536, 3072, 2304, 2304, 2304, 2304, 3072, 288, 
               6144, 12288, 9216, 9216, 9216, 9216, 12288, 576, 24576, 36864, 36864, 36864, 12288, 
               49152, 24576, 24576, 12288, 6144, 288, 12288, 12288, 18432, 9216, 9216, 9216, 4608, 
               2304, 2304, 2304, 9216, 9216, 9216, 9216, 9216, 6144, 288, 36864, 36864, 288, 12288, 
               576, 12288, 288]

# Each Line has 4-data
Weight_Size = [weight * 4 for weight in Weight_Size]

# Size of Reordered Bias
Bias_Size = [8, 16, 16, 16, 8, 8, 32, 32, 32, 16, 16, 16, 16, 64, 64, 64, 64, 32, 32, 32, 32, 128, 128, 
             128, 128, 64, 64, 64, 128, 128, 128, 128, 64, 64, 128, 64, 64, 64, 32, 32, 32, 32, 16, 16, 
             32, 64, 64, 32, 32, 64, 64, 128, 128, 64, 128, 128, 64, 64]

# Each Line has 4-data
Bias_Size = [bias * 4 for bias in Bias_Size]

# Calculate Weight & Bias Address
Weight_Address = []; Bias_Address = []
weight_address = WEIGHT_BASE
bias_address = PARAM_BASE
for weight, bias in zip(Weight_Size, Bias_Size): 
    Weight_Address.append(weight_address)
    Bias_Address.append(bias_address)
    weight_address += weight
    bias_address += bias

# Activation Address
YOLOv10n_activation_address = {}
for i in range(23):
    YOLOv10n_activation_address[f"layer{i}"] = {}
    
# Function to append the operation if it doesn't exist
def Activation_Address(layer, operation_name, input_value, output_value):
    # Initialize the operation if not already present
    if operation_name not in YOLOv10n_activation_address[layer]:
        YOLOv10n_activation_address[layer][operation_name] = {
            "Input_Address": [],
            "Output_Address": []
        }
    # Append the input and output to the operation
    YOLOv10n_activation_address[layer][operation_name]["Input_Address"].extend(input_value)
    YOLOv10n_activation_address[layer][operation_name]["Output_Address"].extend(output_value)

##################################
#             Layer0             #
##################################
# ConvAct0
CONV0_INPUT  = INPUT_BASE
CONV0_OUTPUT = CONV0_INPUT + (307200 * 4)
Activation_Address("layer0", "ConvAct0", [CONV0_INPUT], [CONV0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer0 --> ConvAct0:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[0])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[0] + (144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV0_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV0_OUTPUT)[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[0])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[0] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV0_OUTPUT + (307200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer1             #
##################################
# ConvAct1
CONV1_INPUT  = CONV0_OUTPUT
CONV1_OUTPUT = CONV1_INPUT + (307200 * 4) # After CONV0_OUTPUT
Activation_Address("layer1", "ConvAct1", [CONV1_INPUT], [CONV1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer1 --> ConvAct1:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[1])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[1] + (1152 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV1_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV1_OUTPUT)[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[1])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[1] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV1_OUTPUT + (153600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer2             #
##################################
# "ConvAct2 -- Branch", 
CONV2_INPUT   = CONV1_OUTPUT
CONV2_OUTPUT0 = CONV2_INPUT + (153600 * 4)      # After CONV1_OUTPUT
CONV2_OUTPUT1 = CONV2_OUTPUT0 + (76800 * 4 * 2) # After CONV2_OUTPUT0 (Split0 + Split1) 
# For Concat--layer2
EWADDER0_OUTPUT = CONV2_OUTPUT1 + (76800 * 4 * 2) # After CONV2_OUTPUT1 (Split0 + Split1) 
Activation_Address("layer2", "ConvAct2", [CONV2_INPUT], [CONV2_OUTPUT0, CONV2_OUTPUT1])

if DEBUG_ADDR:
    print(f"layer2 --> ConvAct2:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[2])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[2] + (384 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV2_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV2_INPUT+ (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[2])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[2] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV2_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV2_OUTPUT0+ (76800 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV2_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV2_OUTPUT1+ (76800 * 4 * 2))[2:].zfill(8)}\n")

# "ConvAct4", 
CONV4_INPUT  = CONV2_OUTPUT0 + (76800 * 4) # After CONV2_OUTPUT0 (Split0) --> Split1
CONV4_OUTPUT = EWADDER0_OUTPUT + (76800 * 4) # After EWADDER0_OUTPUT
Activation_Address("layer2", "ConvAct4", [CONV4_INPUT], [CONV4_OUTPUT])
if DEBUG_ADDR:
    print(f"layer2 --> ConvAct4:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[4])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[4] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV4_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV4_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[4])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[4] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV4_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV4_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

# "ConvAct5",
CONV5_INPUT  = CONV4_OUTPUT
CONV5_OUTPUT = CONV5_INPUT +  (76800 * 4) # After CONV4_OUTPUT
Activation_Address("layer2", "ConvAct5", [CONV5_INPUT], [CONV5_OUTPUT])
if DEBUG_ADDR:
    print(f"layer2 --> ConvAct5:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[5])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[5] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV5_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV5_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[5])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[5] + (8 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV5_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV5_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

# "EWAdder0",
EWADDER0_INPUT1 = CONV5_OUTPUT
EWADDER0_INPUT2 = CONV2_OUTPUT1 + (76800 * 4) # After CONV2_OUTPUT1(Split0) --> Split1
Activation_Address("layer2", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer2 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

# "Concat0", 
# CONV2_OUTPUT1 + EWADDER0_OUTPUT
Activation_Address("layer2", "Concat0", [None], [None])

# "ConvAct3
CONV3_INPUT  = CONV2_OUTPUT1          # Concat (Split0 + Split1 + EWADDER0_OUTPUT)
CONV3_OUTPUT = CONV5_OUTPUT + (76800 * 4) # After CONV5_OUTPUT
Activation_Address("layer2", "ConvAct3", [CONV3_INPUT], [CONV3_OUTPUT])
if DEBUG_ADDR:
    print(f"layer2 --> ConvAct3:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[3])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[3] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV3_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV3_INPUT + (230400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[3])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[3] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV3_OUTPUT + (153600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer3             #
##################################
# "ConvAct6",
CONV6_INPUT  = CONV3_OUTPUT
CONV6_OUTPUT = CONV6_INPUT + (153600 * 4) # After CONV3_OUTPUT
Activation_Address("layer3", "ConvAct6", [CONV6_INPUT], [CONV6_OUTPUT])
if DEBUG_ADDR:
    print(f"layer3 --> ConvAct6:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[6])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[6] + (4608 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV6_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV6_INPUT + (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[6])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[6] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV6_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV6_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer4             #
##################################
# "ConvAct7 -- Branch2", 
CONV7_INPUT   = CONV6_OUTPUT
CONV7_OUTPUT0 = CONV7_INPUT + (76800 * 4) # After CONV6_OUTPUT
CONV7_OUTPUT1 = CONV7_OUTPUT0 + (38400 * 4 * 2) # After CONV7_OUTPUT0 (Split0 + Split1)
CONV7_OUTPUT2 = CONV7_OUTPUT1 + (38400 * 4 * 2) # After CONV7_OUTPUT1 (Split0 + Split1)
Activation_Address("layer4", "ConvAct7", [CONV7_INPUT], [CONV7_OUTPUT0, CONV7_OUTPUT1, CONV7_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct7:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[7])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[7] + (1536 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV7_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV7_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[7])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[7] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV7_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV7_OUTPUT0 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV7_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV7_OUTPUT1 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT3: START 0x{hex(CONV7_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV7_OUTPUT2 + (76800 * 4))[2:].zfill(8)}\n")

# For Concat--layer4
EWADDER1_OUTPUT = CONV7_OUTPUT2 + (38400 * 4 * 2) # After CONV7_OUTPUT2 (Split0 + Split1) 
EWADDER2_OUTPUT = EWADDER1_OUTPUT +  (38400 * 4) # After EWADDER1_OUTPUT

# "ConvAct9", 
CONV9_INPUT  = CONV7_OUTPUT0 + (38400 * 4) # After CONV7_OUTPUT0 (Split0) --> Split1
CONV9_OUTPUT = EWADDER2_OUTPUT + (38400 * 4) # After EWADDER2_OUTPUT
Activation_Address("layer4", "ConvAct9", [CONV9_INPUT], [CONV9_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct9:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[9])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[9] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV9_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV9_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[9])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[9] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV9_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV9_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "ConvAct10 -- Branch", 
CONV10_INPUT  = CONV9_OUTPUT
CONV10_OUTPUT0 = CONV10_INPUT + (38400 * 4) # After CONV9_OUTPUT
CONV10_OUTPUT1 = CONV10_OUTPUT0 + (38400 * 4) # After CONV10_OUTPUT0
Activation_Address("layer4", "ConvAct10", [CONV10_INPUT], [CONV10_OUTPUT0, CONV10_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct10:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[10])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[10] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV10_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV10_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[10])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[10] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV10_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV10_OUTPUT0 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV10_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV10_OUTPUT1 + (38400 * 4))[2:].zfill(8)}\n")

# "EWAdder0",
EWADDER0_INPUT1 = CONV10_OUTPUT0
EWADDER0_INPUT2 = CONV7_OUTPUT1 + (38400 * 4) # After CONV7_OUTPUT1(Split0) --> Split1
EWADDER0_OUTPUT = CONV10_OUTPUT1 + (38400 * 4) # After CONV10_OUTPUT1
Activation_Address("layer4", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "ConvAct11",
CONV11_INPUT  = EWADDER0_OUTPUT
CONV11_OUTPUT = CONV11_INPUT + (38400 * 4) # After EWADDER0_OUTPUT
Activation_Address("layer4", "ConvAct11", [CONV11_INPUT], [CONV11_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct11:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[11])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[11] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV11_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV11_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[11])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[11] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV11_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV11_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "ConvAct12",
CONV12_INPUT  = CONV11_OUTPUT
CONV12_OUTPUT = CONV12_INPUT + (38400 * 4) # After CONV11_OUTPUT
Activation_Address("layer4", "ConvAct12", [CONV12_INPUT], [CONV12_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct12:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[12])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[12] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV12_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV12_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[12])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[12] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV12_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV12_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "EWAdder1", 
EWADDER1_INPUT1 = CONV10_OUTPUT1
EWADDER1_INPUT2 = CONV7_OUTPUT2 + (38400 * 4) # After CONV7_OUTPUT2(Split0) --> Split1
Activation_Address("layer4", "EWAdder1", [EWADDER1_INPUT1, EWADDER1_INPUT2], [EWADDER1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> EWAdder1:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT1 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT2 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "EWAdder2",
EWADDER2_INPUT1 = EWADDER1_OUTPUT
EWADDER2_INPUT2 = CONV12_OUTPUT 
Activation_Address("layer4", "EWAdder2", [EWADDER2_INPUT1, EWADDER2_INPUT2], [EWADDER2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer4 --> EWAdder2:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER2_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT1 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER2_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT2 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

# "Concat0", 
# CONV7_OUTPUT2 + EWADDER1_OUTPUT + EWADDER2_OUTPUT
Activation_Address("layer4", "Concat0", [None], [None])

# "ConvAct8 -- Branch"
CONV8_INPUT   = CONV7_OUTPUT2          # Concat (Split0 + Split1 + EWADDER1_OUTPUT + EWADDER2_OUTPUT)
CONV8_OUTPUT0 = CONV12_OUTPUT + (38400 * 4) # After CONV12_OUTPUT
# For Concat--layer15
UPSAMPLE1_OUTPUT = CONV8_OUTPUT0 + (76800 * 4) # After CONV8_OUTPUT0
CONV8_OUTPUT1 = UPSAMPLE1_OUTPUT + (153600 * 4) # After UPSAMPLE1_OUTPUT
Activation_Address("layer4", "ConvAct8", [CONV8_INPUT], [CONV8_OUTPUT0, CONV8_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer4 --> ConvAct8:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[8])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[8] + (3072 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV8_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV8_INPUT + (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[8])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[8] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV8_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV8_OUTPUT0 + (76800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV8_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV8_OUTPUT1 + (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer5             #
##################################
# "ConvAct13", 
CONV13_INPUT  = CONV8_OUTPUT0
CONV13_OUTPUT = CONV8_OUTPUT1 + (76800 * 4) # After CONV8_OUTPUT1
Activation_Address("layer5", "ConvAct13", [CONV13_INPUT], [CONV13_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> ConvAct13:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[13])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[13] + (3072 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV13_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV13_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[13])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[13] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV13_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV13_OUTPUT + (153600 * 4))[2:].zfill(8)}\n")

# "Conv14"
CONV14_INPUT  = CONV13_OUTPUT
CONV14_OUTPUT = CONV14_INPUT + (153600 * 4) # After CONV13_OUTPUT
Activation_Address("layer5", "Conv14", [CONV14_INPUT], [CONV14_OUTPUT])
if DEBUG_ADDR:
    print(f"layer5 --> Conv14:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[14])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[14] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV14_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV14_INPUT + (153600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[14])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[14] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV14_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV14_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

##################################
#             Layer6             #
##################################
# "ConvAct15 -- Branch2",
CONV15_INPUT   = CONV14_OUTPUT
CONV15_OUTPUT0 = CONV15_INPUT + (38400 * 4) # After CONV14_OUTPUT
CONV15_OUTPUT1 = CONV15_OUTPUT0 + (19200 * 4 * 2) # After CONV15_OUTPUT0 (Split0 + Split1)
CONV15_OUTPUT2 = CONV15_OUTPUT1 + (19200 * 4 * 2) # After CONV15_OUTPUT1 (Split0 + Split1) 
Activation_Address("layer6", "ConvAct15", [CONV15_INPUT], [CONV15_OUTPUT0, CONV15_OUTPUT1, CONV15_OUTPUT2])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct15:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[15])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[15] + (6144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV15_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV15_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[15])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[15] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV15_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV15_OUTPUT0 + (19200 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV15_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV15_OUTPUT1 + (19200 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT3: START 0x{hex(CONV15_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV15_OUTPUT2 + (19200 * 4 * 2))[2:].zfill(8)}\n")

# For Concat--layer6
EWADDER1_OUTPUT = CONV15_OUTPUT2 + (19200 * 4 * 2) # After CONV15_OUTPUT2 (Split0 + Split1) 
EWADDER2_OUTPUT = EWADDER1_OUTPUT + (19200 * 4) # After EWADDER1_OUTPUT

# "ConvAct17",
CONV17_INPUT  = CONV15_OUTPUT0 + (19200 * 4) # After CONV15_OUTPUT0 (Split0) --> Split1
CONV17_OUTPUT = EWADDER2_OUTPUT + (19200 * 4) # After EWADDER2_OUTPUT  
Activation_Address("layer6", "ConvAct17", [CONV17_INPUT], [CONV17_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct17:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[17])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[17] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV17_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV17_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[17])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[17] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV17_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV17_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct18 -- Branch",
CONV18_INPUT  = CONV17_OUTPUT
CONV18_OUTPUT0 = CONV18_INPUT + (19200 * 4) # After CONV17_OUTPUT  
CONV18_OUTPUT1 = CONV18_OUTPUT0 + (19200 * 4) # After CONV18_OUTPUT0   
Activation_Address("layer6", "ConvAct18", [CONV18_INPUT], [CONV18_OUTPUT0, CONV18_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct18:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[18])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[18] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV18_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV18_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[18])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[18] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV18_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV18_OUTPUT0 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV18_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV18_OUTPUT1 + (19200 * 4))[2:].zfill(8)}\n")

# "EWAdder0",
EWADDER0_INPUT1 = CONV18_OUTPUT0
EWADDER0_INPUT2 = CONV15_OUTPUT1 + (19200 * 4) # After CONV15_OUTPUT1(Split0) --> Split1
EWADDER0_OUTPUT = CONV18_OUTPUT1 + (19200 * 4) # After CONV18_OUTPUT1  
Activation_Address("layer6", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct19",
CONV19_INPUT  = EWADDER0_OUTPUT
CONV19_OUTPUT = EWADDER0_OUTPUT + (19200 * 4) # After EWADDER0_OUTPUT 
Activation_Address("layer6", "ConvAct19", [CONV19_INPUT], [CONV19_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct19:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[19])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[19] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV19_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV19_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[19])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[19] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV19_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV19_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct20",
CONV20_INPUT  = CONV19_OUTPUT
CONV20_OUTPUT = CONV19_OUTPUT + (19200 * 4) # After CONV19_OUTPUT    
Activation_Address("layer6", "ConvAct20", [CONV20_INPUT], [CONV20_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct20:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[20])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[20] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV20_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV20_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[20])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[20] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV20_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV20_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "EWAdder1",
EWADDER1_INPUT1 = CONV18_OUTPUT1
EWADDER1_INPUT2 = CONV15_OUTPUT2 + (19200 * 4) # After CONV15_OUTPUT2(Split0) --> Split1 
Activation_Address("layer6", "EWAdder1", [EWADDER1_INPUT1, EWADDER1_INPUT2], [EWADDER1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder1:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT1 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT2 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "EWAdder2",
EWADDER2_INPUT1 = EWADDER1_OUTPUT
EWADDER2_INPUT2 = CONV20_OUTPUT
Activation_Address("layer6", "EWAdder2", [EWADDER2_INPUT1, EWADDER2_INPUT2], [EWADDER2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer6 --> EWAdder2:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER2_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT1 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER2_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT2 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "Concat0", 
# CONV15_OUTPUT2 + EWADDER1_OUTPUT + EWADDER2_OUTPUT
Activation_Address("layer6", "Concat0", [None], [None])

# "ConvAct16 -- Branch"
CONV16_INPUT  = CONV15_OUTPUT2          # Concat (Split0 + Split1 + EWADDER1_OUTPUT + EWADDER2_OUTPUT)
CONV16_OUTPUT0 = CONV20_OUTPUT + (19200 * 4) # After CONV20_OUTPUT  
# For Concat--layer12
UPSAMPLE0_OUTPUT = CONV16_OUTPUT0 + (38400 * 4) # After CONV16_OUTPUT0
CONV16_OUTPUT1 = UPSAMPLE0_OUTPUT + (76800 * 4) # After UPSAMPLE0_OUTPUT
Activation_Address("layer6", "ConvAct16", [CONV16_INPUT], [CONV16_OUTPUT0, CONV16_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer6 --> ConvAct16:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[16])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[16] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV16_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV16_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[16])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[16] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV16_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV16_OUTPUT0 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV16_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV16_OUTPUT1 + (38400 * 4))[2:].zfill(8)}\n")

##################################
#             Layer7             #
##################################
# "ConvAct21",
CONV21_INPUT  = CONV16_OUTPUT0
CONV21_OUTPUT = CONV16_OUTPUT1 + (38400 * 4) # After CONV16_OUTPUT1
Activation_Address("layer7", "ConvAct21", [CONV21_INPUT], [CONV21_OUTPUT])
if DEBUG_ADDR:
    print(f"layer7 --> ConvAct21:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[21])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[21] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV21_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV21_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[21])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[21] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV21_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV21_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

# "Conv22"
CONV22_INPUT  = CONV21_OUTPUT
CONV22_OUTPUT = CONV22_INPUT + (76800 * 4) # After CONV21_OUTPUT
Activation_Address("layer7", "Conv22", [CONV22_INPUT], [CONV22_OUTPUT])
if DEBUG_ADDR:
    print(f"layer7 --> Conv22:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[22])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[22] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV22_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV22_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[22])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[22] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV22_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV22_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer8             #
##################################
# "ConvAct23 -- Branch",
CONV23_INPUT   = CONV22_OUTPUT
CONV23_OUTPUT0 = CONV22_OUTPUT + (19200 * 4) # After CONV22_OUTPUT
CONV23_OUTPUT1 = CONV23_OUTPUT0 + (9600 * 4 * 2) # After CONV23_OUTPUT0 (Split0 + Split1)
# For Concat--layer8
EWADDER0_OUTPUT = CONV23_OUTPUT1 + (9600 * 4 * 2) # After CONV23_OUTPUT1 (Split0 + Split1)
Activation_Address("layer8", "ConvAct23", [CONV23_INPUT], [CONV23_OUTPUT0, CONV23_OUTPUT1])
if DEBUG_ADDR:
    print(f"Layer8 --> ConvAct23:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[23])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[23] + (24576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV23_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV23_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[23])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[23] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV23_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV23_OUTPUT0 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV23_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV23_OUTPUT1 + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct25",
CONV25_INPUT  = CONV23_OUTPUT0 + (9600 * 4) # After CONV23_OUTPUT0 (Split0) --> Split1
CONV25_OUTPUT = EWADDER0_OUTPUT + (9600 * 4) # After EWADDER0_OUTPUT   
Activation_Address("layer8", "ConvAct25", [CONV25_INPUT], [CONV25_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer8 --> ConvAct25:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[25])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[25] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV25_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV25_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[25])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[25] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV25_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV25_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

# "ConvAct26",
CONV26_INPUT  = CONV25_OUTPUT
CONV26_OUTPUT = CONV26_INPUT +  (9600 * 4) # After CONV25_OUTPUT
Activation_Address("layer8", "ConvAct26", [CONV26_INPUT], [CONV26_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer8 --> ConvAct26:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[26])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[26] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV26_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV26_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[26])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[26] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV26_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV26_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

# "EWAdder0",
EWADDER0_INPUT1 = CONV23_OUTPUT1 + (9600 * 4) # After CONV23_OUTPUT1(Split0) --> Split1
EWADDER0_INPUT2 = CONV26_OUTPUT  
Activation_Address("layer8", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])
if DEBUG_ADDR:
    print(f"layer8 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT1 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

# "Concat0", 
# CONV23_OUTPUT1 + EWADDER0_OUTPUT
Activation_Address("layer8", "Concat0", [None], [None])

# "ConvAct24"
CONV24_INPUT  = CONV23_OUTPUT1         # Concat (Split0 + Split1 + EWADDER0_OUTPUT)
CONV24_OUTPUT = CONV26_OUTPUT + (9600 * 4) # After CONV26_OUTPUT
Activation_Address("layer8", "ConvAct24", [CONV24_INPUT], [CONV24_OUTPUT])
if DEBUG_ADDR:
    print(f"Layer8 --> ConvAct24:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[24])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[24] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV24_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV24_INPUT + (28800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[24])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[24] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV24_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV24_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer9             #
##################################
# "ConvAct27",
CONV27_INPUT  = CONV24_OUTPUT 
CONV27_OUTPUT = CONV24_OUTPUT + (19200 * 4) # After CONV24_OUTPUT   
Activation_Address("layer9", "ConvAct27", [CONV27_INPUT], [CONV27_OUTPUT])
if DEBUG_ADDR:
    print(f"layer9 --> ConvAct27:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[27])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[27] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV27_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV27_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[27])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[27] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV27_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV27_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

# "MaxPool1", 
MAXPOOL1_INPUT  = CONV27_OUTPUT
MAXPOOL1_OUTPUT = MAXPOOL1_INPUT + (9600 * 4) # After CONV27_OUTPUT
Activation_Address("layer9", "MaxPool1", [MAXPOOL1_INPUT], [MAXPOOL1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer9 --> MaxPool1:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL1_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL1_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL1_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

# "MaxPool2", 
MAXPOOL2_INPUT  = MAXPOOL1_OUTPUT 
MAXPOOL2_OUTPUT = MAXPOOL2_INPUT + (9600 * 4) # After MAXPOOL1_OUTPUT 
Activation_Address("layer9", "MaxPool2", [MAXPOOL2_INPUT], [MAXPOOL2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer9 --> MaxPool2:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL2_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL2_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL2_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n") 

# "MaxPool3",
MAXPOOL3_INPUT  = MAXPOOL2_OUTPUT
MAXPOOL3_OUTPUT = MAXPOOL3_INPUT + (9600 * 4) # After MAXPOOL2_OUTPUT   
Activation_Address("layer9", "MaxPool3", [MAXPOOL3_INPUT], [MAXPOOL3_OUTPUT])
if DEBUG_ADDR:
    print(f"layer9 --> MaxPool3:") 
    print(f"\tLD_IN1: START: 0x{hex(MAXPOOL3_INPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL3_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MAXPOOL3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MAXPOOL3_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n") 

# "Concat0",
# CONV27_OUTPUT + MAXPOOL1_OUTPUT + MAXPOOL2_OUTPUT + MAXPOOL3_OUTPUT 
Activation_Address("layer9", "Concat0", [None], [None])

# "ConvAct28"
CONV28_INPUT  = CONV27_OUTPUT
CONV28_OUTPUT = MAXPOOL3_OUTPUT + (9600 * 4) # After MAXPOOL3_OUTPUT 
Activation_Address("layer9", "ConvAct28", [CONV28_INPUT], [CONV28_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer9 --> ConvAct28:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[28])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[28] + (49152 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV28_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV28_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[28])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[28] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV28_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV28_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer10            #
##################################
# "ConvAct29 -- Branch2",
CONV29_INPUT   = CONV28_OUTPUT
CONV29_OUTPUT0 = CONV29_INPUT + (19200 * 4) # After CONV28_OUTPUT
CONV29_OUTPUT1 = CONV29_OUTPUT0 + (9600 * 4 * 2) # After CONV29_OUTPUT0 (Split0 + Split1)
CONV29_OUTPUT2 = CONV29_OUTPUT1 + (9600 * 4 * 2) # After CONV29_OUTPUT1 (Split0 + Split1) 
CONV29_OUTPUT3 = CONV29_OUTPUT2 + (9600 * 4 * 2) # After CONV29_OUTPUT2 (Split0 + Split1) 
# For Concat--layer10
EWADDER3_OUTPUT = CONV29_OUTPUT3 + (9600 * 4) # After CONV29_OUTPUT3(Split0) overwrite CONV11_OUTPUT4(Split1)
Activation_Address("layer10", "ConvAct29", [CONV29_INPUT], [CONV29_OUTPUT0, CONV29_OUTPUT1, CONV29_OUTPUT2, CONV29_OUTPUT3]) 
if DEBUG_ADDR:
    print(f"layer10 --> ConvAct29:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[29])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[29] + (24576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV29_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV29_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[29])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[29] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV29_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV29_OUTPUT0 + (9600 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV29_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV29_OUTPUT1 + (9600 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT3: START 0x{hex(CONV29_OUTPUT2)[2:].zfill(8)}\t END: 0x{hex(CONV29_OUTPUT2 + (9600 * 4 * 2))[2:].zfill(8)}")
    print(f"\tST_OUT4: START 0x{hex(CONV29_OUTPUT3)[2:].zfill(8)}\t END: 0x{hex(CONV29_OUTPUT3 + (9600 * 4 * 2))[2:].zfill(8)}\n")

# "Conv31 -- Branch",
CONV31_INPUT   = CONV29_OUTPUT0 + (9600 * 4)                        # After CONV29_OUTPUT0 (Split0) --> Split1
CONV31_OUTPUT0 = EWADDER3_OUTPUT + (9600 * 4)                       # After EWADDER3_OUTPUT 
CONV31_OUTPUT1 = CONV31_OUTPUT0 + ((2400 + 2400 + 4800) * 4 * 2)    # After CONV31_OUTPUT0 (2-Heads) --> q,k,v (branch1)
Activation_Address("layer10", "Conv31", [CONV31_INPUT], [CONV31_OUTPUT0, CONV31_OUTPUT1]) 
if DEBUG_ADDR:
    print(f"layer10 --> Conv31:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[31])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[31] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV31_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV31_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[31])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[31] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV31_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV31_OUTPUT0 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV31_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV31_OUTPUT1 + (19200 * 4))[2:].zfill(8)}\n")

# "AttnHead0",
MATMUL0_INPUT1 = CONV29_OUTPUT0                                   # Query-Head0
MATMUL0_INPUT2 = MATMUL0_INPUT1 + (2400 * 4)                          # Key-Head0
MATMUL0_INPUT3 = MATMUL0_INPUT2 + (2400 * 4)                          # Value-Head0
MATMUL0_OUTPUT = CONV29_OUTPUT1 + ((2400 + 2400 + 4800) * 4 * 2)      # After CONV13_OUTPUT2 (2-Heads) q,k,v (branch1)
Activation_Address("layer10", "AttnHead0", [MATMUL0_INPUT1, MATMUL0_INPUT2, MATMUL0_INPUT3], [MATMUL0_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer10 --> AttnHead0:") 
    print(f"\tLD_WGT: START: 0x{hex(MATMUL0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT1+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(MATMUL0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT2+ (2400 * 4) )[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(MATMUL0_INPUT3)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_INPUT3+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MATMUL0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MATMUL0_OUTPUT+ (4800 * 4))[2:].zfill(8)}\n")

# "AttnHead1",
MATMUL1_INPUT1 = MATMUL0_INPUT3 + (4800 * 4)   # Query-Head1
MATMUL1_INPUT2 = MATMUL1_INPUT1 + (2400 * 4)   # Key-Head1
MATMUL1_INPUT3 = MATMUL1_INPUT2 + (2400 * 4)   # Value-Head1
MATMUL1_OUTPUT = MATMUL0_OUTPUT + (4800 * 4)  # After MATMUL0_OUTPUT
Activation_Address("layer10", "AttnHead1", [MATMUL1_INPUT1, MATMUL1_INPUT2, MATMUL1_INPUT3], [MATMUL1_OUTPUT]) 

# Concat: MATMUL0_OUTPUT + MATMUL1_OUTPUT
if DEBUG_ADDR:
    print(f"layer10 --> AttnHead1:") 
    print(f"\tLD_WGT: START: 0x{hex(MATMUL1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT1+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(MATMUL1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT2+ (2400 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(MATMUL1_INPUT3)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_INPUT3+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(MATMUL1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(MATMUL1_OUTPUT+ (4800 * 4))[2:].zfill(8)}\n")

# "Conv33-Head0",
CONV33_INPUT  = CONV31_OUTPUT1 + ((2400 + 2400) * 4) # v-Head0 -- Branch2
CONV33_OUTPUT_Head0 = MATMUL1_OUTPUT + (4800 * 4) # After MATMUL1_OUTPUT  
Activation_Address("layer10", "Conv33_Head0", [CONV33_INPUT], [CONV33_OUTPUT_Head0])  
if DEBUG_ADDR:
    print(f"layer10 --> Conv33_Head0:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[33])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[33]+ ((288 // 2) * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV33_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV33_INPUT+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[33])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[33]+ ((64 // 2) * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV33_OUTPUT_Head0)[2:].zfill(8)}\t END: 0x{hex(CONV33_OUTPUT_Head0 + ((9600 // 2) * 4))[2:].zfill(8)}\n")

# "Conv33-Head1",
CONV33_INPUT  = CONV31_OUTPUT1 + ((2400 + 2400 + 4800) + (2400 + 2400)) * 4 # v-Head1 -- Branch2
CONV33_OUTPUT_Head1 = CONV33_OUTPUT_Head0 + (9600 // 2) * 4 # (1, 64, 20, 15) -- Head1
Activation_Address("layer10", "Conv33_Head1", [CONV33_INPUT], [CONV33_OUTPUT_Head1])  
if DEBUG_ADDR:
    print(f"layer10 --> Conv33_Head1:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[33]+ ((288 // 2) * 4))[2:].zfill(8)}\t END: 0x{hex(Weight_Address[33] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV33_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV33_INPUT+ (4800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[33]+ ((64 // 2) * 4))[2:].zfill(8)}\t END: 0x{hex(Bias_Address[33] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV33_OUTPUT_Head1)[2:].zfill(8)}\t END: 0x{hex(CONV33_OUTPUT_Head1+((9600 // 2) * 4))[2:].zfill(8)}\n")

# "EWAdder0",
EWADDER0_INPUT1 = CONV33_OUTPUT_Head0 # CONV33_OUTPUT_Head0 + CONV33_OUTPUT_Head1
EWADDER0_INPUT2 = MATMUL0_OUTPUT
EWADDER0_OUTPUT = CONV33_OUTPUT_Head0 + (9600 * 4) # After CONV33_OUTPUT_Head0 + CONV33_OUTPUT_Head1 
Activation_Address("layer10", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer10 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")

# "Conv32 -- Branch",
CONV32_INPUT   = EWADDER0_OUTPUT
CONV32_OUTPUT0 = CONV32_INPUT + (9600 * 4) # After EWADDER0_OUTPUT
CONV32_OUTPUT1 = CONV32_OUTPUT0 + (9600 * 4) # After CONV32_OUTPUT0   
Activation_Address("layer10", "Conv32", [CONV32_INPUT], [CONV32_OUTPUT0, CONV32_OUTPUT1])  
if DEBUG_ADDR:
    print(f"layer10 --> Conv32:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[32])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[32] + (6144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV32_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV32_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[32])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[32] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV32_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV32_OUTPUT0+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV32_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV32_OUTPUT1+ (9600 * 4))[2:].zfill(8)}\n")

# "EWAdder1",
EWADDER1_INPUT1 = CONV29_OUTPUT1 + (9600 * 4) # After CONV29_OUTPUT1(Split0) --> Split1
EWADDER1_INPUT2 = CONV32_OUTPUT0
EWADDER1_OUTPUT = CONV32_OUTPUT1 + (9600 * 4) # After CONV32_OUTPUT1 
Activation_Address("layer10", "EWAdder1", [EWADDER1_INPUT1, EWADDER1_INPUT2], [EWADDER1_OUTPUT])   
if DEBUG_ADDR:
    print(f"layer10 --> EWAdder1:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER1_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER1_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_INPUT2+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER1_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n") 

# "ConvAct34",
CONV34_INPUT  = EWADDER1_OUTPUT
CONV34_OUTPUT = CONV34_INPUT + (9600 * 4) # After EWADDER1_OUTPUT 
Activation_Address("layer10", "ConvAct34", [CONV34_INPUT], [CONV34_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer10 --> ConvAct34:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[34])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[34] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV34_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV34_INPUT+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[34])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[34] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV34_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV34_OUTPUT+ (19200 * 4))[2:].zfill(8)}\n")  

# "Conv35",
CONV35_INPUT  = CONV34_OUTPUT
CONV35_OUTPUT = CONV35_INPUT + (19200 * 4) # After CONV34_OUTPUT
Activation_Address("layer10", "Conv35", [CONV35_INPUT], [CONV35_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer10 --> Conv35:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[35])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[35] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV35_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV35_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[35])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[35] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV35_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV35_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")   

# "EWAdder2",
EWADDER2_INPUT1 = CONV29_OUTPUT2 + (9600 * 4) # After CONV29_OUTPUT2(Split0) --> Split1
EWADDER2_INPUT2 = CONV32_OUTPUT1
EWADDER2_OUTPUT = CONV35_OUTPUT + (9600 * 4) # After CONV35_OUTPUT  
Activation_Address("layer10", "EWAdder2", [EWADDER2_INPUT1, EWADDER2_INPUT2], [EWADDER2_OUTPUT])
if DEBUG_ADDR:
    print(f"layer10 --> EWAdder2:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER2_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT1 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER2_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER2_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER2_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")   

# "EWAdder3",
EWADDER3_INPUT1 = EWADDER2_OUTPUT
EWADDER3_INPUT2 = CONV35_OUTPUT
Activation_Address("layer10", "EWAdder3", [EWADDER3_INPUT1, EWADDER3_INPUT2], [EWADDER3_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer10 --> EWAdder3:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER3_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_INPUT1+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER3_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_INPUT2+ (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER3_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER3_OUTPUT+ (9600 * 4))[2:].zfill(8)}\n")   

# "Concat0", 
# CONV29_OUTPUT3 (SPLIT0) + EWADDER3_OUTPUT
Activation_Address("layer10", "Concat0", [None], [None])

# "ConvAct30 -- Branch"
CONV30_INPUT  = CONV29_OUTPUT3         # Concat (Split0 + EWADDER3_OUTPUT)
CONV30_OUTPUT0 = EWADDER2_OUTPUT + (9600 * 4) # After EWADDER2_OUTPUT
# For Concat--layer21
CONV50_OUTPUT = CONV30_OUTPUT0 + (19200 * 4) # After CONV30_OUTPUT0
CONV30_OUTPUT1 = CONV50_OUTPUT + (9600 * 4) # After CONV50_OUTPUT
Activation_Address("layer10", "ConvAct30", [CONV30_INPUT], [CONV30_OUTPUT0, CONV30_OUTPUT1]) 
if DEBUG_ADDR:
    print(f"layer10 --> ConvAct30:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[30])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[30] + (24576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV30_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV30_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[30])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[30] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(CONV30_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV30_OUTPUT0 + (19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START: 0x{hex(CONV30_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV30_OUTPUT1 + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer11            #
##################################
# "Upsample0"
UPSAMPLE0_INPUT  = CONV30_OUTPUT0
Activation_Address("layer11", "Upsample0", [UPSAMPLE0_INPUT], [UPSAMPLE0_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer11 --> Upsample0:") 
    print(f"\tLD_IN1: START: 0x{hex(UPSAMPLE0_INPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE0_INPUT+(19200 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(UPSAMPLE0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE0_OUTPUT+ (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer12            #
##################################
# "Concat0"
# UPSAMPLE0_OUTPUT + CONV16_OUTPUT1 
Activation_Address("layer12", "Concat0", [None], [None])

##################################
#             Layer13            #
##################################
# "ConvAct36 -- Branch",
CONV36_INPUT   = UPSAMPLE0_OUTPUT
CONV36_OUTPUT0 = CONV30_OUTPUT1 + (19200 * 4) # After CONV30_OUTPUT1
CONV36_OUTPUT1 = CONV36_OUTPUT0  + (19200 * 4 * 2) # After CONV36_OUTPUT0 (Split0 + Split1)
# For Concat--layer13
CONV39_OUTPUT = CONV36_OUTPUT1 + (19200 * 4 * 2) # After CONV36_OUTPUT1 (Split0 + Split1) 
Activation_Address("layer13", "ConvAct36", [CONV36_INPUT], [CONV36_OUTPUT0, CONV36_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer13 --> ConvAct36:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[13])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[13] + (18432 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV36_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV36_INPUT + (115200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[13])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[13] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV36_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV36_OUTPUT0 + (19200 * 4 * 2))[2:].zfill(8)}\n")
    print(f"\tST_OUT2: START 0x{hex(CONV36_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV36_OUTPUT1 + (19200 * 4 * 2))[2:].zfill(8)}\n")

# "ConvAct38",
CONV38_INPUT  = CONV36_OUTPUT0 + (19200 * 4) # After CONV36_OUTPUT0 (Split0) --> Split1
CONV38_OUTPUT = CONV39_OUTPUT + (19200 * 4) # After CONV39_OUTPUT
Activation_Address("layer13", "ConvAct38", [CONV38_INPUT], [CONV38_OUTPUT])
if DEBUG_ADDR:
    print(f"layer13 --> ConvAct38:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[38])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[38] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV38_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV38_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[38])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[38] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV38_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV38_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct39", 
CONV39_INPUT  = CONV38_OUTPUT 
Activation_Address("layer13", "ConvAct39", [CONV39_INPUT], [CONV39_OUTPUT])
if DEBUG_ADDR:
    print(f"layer13 --> ConvAct39:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[39])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[39] + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV39_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV39_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[39])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[39] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV39_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV39_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "Concat0", 
# CONV36_OUTPUT1 + CONV39_OUTPUT
Activation_Address("layer13", "Concat0", [None], [None])

# "ConvAct37 -- Branch"
CONV37_INPUT   = CONV36_OUTPUT1          # Concat (Split0 + Split1 + CONV39_OUTPUT)
CONV37_OUTPUT0 = CONV38_OUTPUT + (19200 * 4) # After CONV38_OUTPUT 
CONV37_OUTPUT1 = CONV37_OUTPUT0 + (38400 * 4) # After CONV37_OUTPUT0 
# For Concat--layer13
CONV44_OUTPUT = CONV37_OUTPUT1 + (38400 * 4) # After CONV37_OUTPUT1
Activation_Address("layer13", "ConvAct37", [CONV37_INPUT], [CONV37_OUTPUT0, CONV37_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer13 --> ConvAct37:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[37])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[37] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV37_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV37_INPUT + (57600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[37])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[37] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV37_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV37_OUTPUT0 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV37_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV37_OUTPUT1 + (38400 * 4))[2:].zfill(8)}\n")

##################################
#             Layer14            #
##################################
# "Upsample1"
UPSAMPLE1_INPUT  = CONV44_OUTPUT
Activation_Address("layer14", "Upsample1", [UPSAMPLE1_INPUT], [UPSAMPLE1_OUTPUT])
if DEBUG_ADDR:
    print(f"layer14 --> Upsample1:") 
    print(f"\tLD_IN1: START: 0x{hex(UPSAMPLE1_INPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE1_INPUT+(38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(UPSAMPLE1_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(UPSAMPLE1_OUTPUT+ (153600 * 4))[2:].zfill(8)}\n")

##################################
#             Layer15            #
##################################
# "Concat0"
# UPSAMPLE1_OUTPUT + CONV8_OUTPUT1
Activation_Address("layer15", "Concat0", [None], [None])

##################################
#             Layer16            #
##################################
# "ConvAct40 -- Branch",
CONV40_INPUT   = UPSAMPLE1_OUTPUT
CONV40_OUTPUT0 = CONV44_OUTPUT + (19200 * 4) # After CONV44_OUTPUT 
CONV40_OUTPUT1 = CONV40_OUTPUT0 + (38400 * 4 * 2) # After CONV40_OUTPUT0 (Split0 + Split1)
# For Concat--layer16
CONV43_OUTPUT = CONV40_OUTPUT1 + (38400 * 4 * 2) # After CONV40_OUTPUT1 (Split0 + Split1) 
Activation_Address("layer16", "ConvAct40", [CONV40_INPUT], [CONV40_OUTPUT0, CONV40_OUTPUT1]) 
if DEBUG_ADDR:
    print(f"layer16 --> ConvAct40:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[40])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[40] + (4608 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV40_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV40_INPUT + (230400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[40])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[40] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV40_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV40_OUTPUT0 + (76800 * 4))[2:].zfill(8)}")  
    print(f"\tST_OUT1: START 0x{hex(CONV40_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV40_OUTPUT1 + (76800 * 4))[2:].zfill(8)}\n")  

# "ConvAct42",
CONV42_INPUT  = CONV40_OUTPUT0 + (38400 * 4) # After CONV40_OUTPUT0 (Split0) --> Split1
CONV42_OUTPUT = CONV43_OUTPUT + (38400 * 4) # After CONV43_OUTPUT 
Activation_Address("layer16", "ConvAct42", [CONV42_INPUT], [CONV42_OUTPUT])
if DEBUG_ADDR:
    print(f"layer16 --> ConvAct42:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[42])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[42] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV42_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV42_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[42])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[42] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV42_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV42_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")     

# "ConvAct43",
CONV43_INPUT  = CONV42_OUTPUT
Activation_Address("layer16", "ConvAct43", [CONV43_INPUT], [CONV43_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer16 --> ConvAct43:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[43])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[43] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV43_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV43_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[43])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[43] + (16 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV43_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV43_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")     

# "Concat0", 
# CONV40_OUTPUT1 + CONV43_OUTPUT
Activation_Address("layer16", "Concat0", [None], [None])

# "ConvAct41"
CONV41_INPUT  = CONV40_OUTPUT1          # Concat (Split0 + Split1 + CONV43_OUTPUT)
CONV41_OUTPUT = CONV42_OUTPUT + (38400 * 4) # After CONV42_OUTPUT  
Activation_Address("layer16", "ConvAct41", [CONV41_INPUT], [CONV41_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer16 --> ConvAct41:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[41])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[41] + (2304 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV41_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV41_INPUT + (115200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[41])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[41] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV41_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV41_OUTPUT + (76800 * 4))[2:].zfill(8)}\n")

##################################
#             Layer17            #
##################################
# "ConvAct44"
CONV44_INPUT  = CONV41_OUTPUT 
Activation_Address("layer17", "ConvAct44", [CONV44_INPUT], [CONV44_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer17 --> ConvAct44:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[44])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[44] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV44_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV44_INPUT + (76800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[44])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[44] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV44_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV44_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

##################################
#             Layer18            #
##################################
# "Concat0"
# CONV37_OUTPUT1 + CONV44_OUTPUT
Activation_Address("layer18", "Concat0", [None], [None])

##################################
#             Layer19            #
##################################
# "ConvAct45 -- Branch",
CONV45_INPUT   = CONV37_OUTPUT1
CONV45_OUTPUT0 = CONV41_OUTPUT + (76800 * 4) # After CONV41_OUTPUT
CONV45_OUTPUT1 = CONV45_OUTPUT0 + (19200 * 4 * 2) # After CONV45_OUTPUT0 (Split0 + Split1)
# For Concat--layer19
CONV48_OUTPUT = CONV45_OUTPUT1 + (19200 * 4 * 2) # After CONV45_OUTPUT1 (Split0 + Split1)   
Activation_Address("layer19", "ConvAct45", [CONV45_INPUT], [CONV45_OUTPUT0, CONV45_OUTPUT1])  
if DEBUG_ADDR:
    print(f"layer19 --> ConvAct45:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[45])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[45] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV45_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV45_INPUT + (57600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[45])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[45] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV45_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV45_OUTPUT0 + (38400 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT2: START 0x{hex(CONV45_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV45_OUTPUT1 + (38400 * 4))[2:].zfill(8)}\n")

# "ConvAct47",
CONV47_INPUT  = CONV45_OUTPUT0 + (19200 * 4) # After CONV45_OUTPUT0 (Split0) --> Split1
CONV47_OUTPUT = CONV48_OUTPUT + (19200 * 4) # After CONV48_OUTPUT
Activation_Address("layer19", "ConvAct47", [CONV47_INPUT], [CONV47_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer19 --> ConvAct47:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[47])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[47] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV47_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV47_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[47])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[47] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV47_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV47_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct48",
CONV48_INPUT  = CONV47_OUTPUT  
Activation_Address("layer19", "ConvAct48", [CONV48_INPUT], [CONV48_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer19 --> ConvAct48:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[48])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[48] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV48_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV48_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[48])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[48] + (32 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV48_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV48_OUTPUT + (19200 * 4))[2:].zfill(8)}\n")

# "Concat0",
# CONV45_OUTPUT1 + CONV48_OUTPUT 
Activation_Address("layer19", "Concat0", [None], [None])

# "ConvAct46"
CONV46_INPUT  = CONV45_OUTPUT1          # Concat (Split0 + Split1 + CONV48_OUTPUT)
CONV46_OUTPUT = CONV47_OUTPUT + (19200 * 4) # After CONV47_OUTPUT
Activation_Address("layer19", "ConvAct46", [CONV46_INPUT], [CONV46_OUTPUT])  
if DEBUG_ADDR:
    print(f"layer19 --> ConvAct46:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[46])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[46] + (9216 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV46_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV46_INPUT + (57600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[46])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[46] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV46_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV46_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")

##################################
#             Layer20            #
##################################
# "ConvAct49",
CONV49_INPUT  = CONV46_OUTPUT
CONV49_OUTPUT = CONV49_INPUT + (38400 * 4) # After CONV46_OUTPUT
Activation_Address("layer20", "ConvAct49", [CONV49_INPUT], [CONV49_OUTPUT])
if DEBUG_ADDR:
    print(f"layer20 --> ConvAct49:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[49])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[49] + (6144 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV49_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV49_INPUT + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[49])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[49] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV49_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV49_OUTPUT + (38400 * 4))[2:].zfill(8)}\n")  

# "Conv50"
CONV50_INPUT  = CONV49_OUTPUT
Activation_Address("layer20", "Conv50", [CONV50_INPUT], [CONV50_OUTPUT])
if DEBUG_ADDR:
    print(f"layer20 --> Conv50:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[50])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[50] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV50_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV50_INPUT  + (38400 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[50])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[50] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV50_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV50_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")  

##################################
#             Layer21            #
##################################
# "Concat0"
# CONV50_OUTPUT + CONV30_OUTPUT1
Activation_Address("layer21", "Concat0", [None], [None])

##################################
#             Layer22            #
##################################
# "ConvAct51 -- Branch", 
CONV51_INPUT   = CONV50_OUTPUT
CONV51_OUTPUT0 = CONV49_OUTPUT + (38400 * 4) # After CONV49_OUTPUT
CONV51_OUTPUT1 = CONV51_OUTPUT0 + (9600 * 4 * 2) # After CONV51_OUTPUT0 (Split0 + Split1)
# For Concat--layer22
EWADDER0_OUTPUT = CONV51_OUTPUT1 + (9600 * 4 * 2) # After CONV51_OUTPUT1 (Split0 + Split1) 
Activation_Address("layer22", "ConvAct51", [CONV51_INPUT], [CONV51_OUTPUT0, CONV51_OUTPUT1])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct51:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[51])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[51] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV51_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV51_INPUT + (28800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[51])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[51] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV51_OUTPUT0)[2:].zfill(8)}\t END: 0x{hex(CONV51_OUTPUT0 + (19200 * 4))[2:].zfill(8)}")  
    print(f"\tST_OUT2: START 0x{hex(CONV51_OUTPUT1)[2:].zfill(8)}\t END: 0x{hex(CONV51_OUTPUT1 + (19200 * 4))[2:].zfill(8)}\n")

# "ConvAct53", 
CONV53_INPUT  = CONV51_OUTPUT0 + (9600 * 4) # After CONV51_OUTPUT0 (Split0) --> Split1
CONV53_OUTPUT = EWADDER0_OUTPUT + (9600 * 4) # After EWADDER0_OUTPUT 
Activation_Address("layer22", "ConvAct53", [CONV53_INPUT], [CONV53_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct53:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[53])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[53] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV53_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV53_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[53])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[53] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV53_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV53_OUTPUT + (9600 * 4))[2:].zfill(8)}\n") 

# "ConvAct54",
CONV54_INPUT  = CONV53_OUTPUT
CONV54_OUTPUT = CONV54_INPUT + (9600 * 4) # After CONV53_OUTPUT
Activation_Address("layer22", "ConvAct54", [CONV54_INPUT], [CONV54_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct54:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[54])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[54] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV54_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV54_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[54])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[54] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV54_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV54_OUTPUT + (19200 * 4))[2:].zfill(8)}\n") 

# "ConvAct55",
CONV55_INPUT  = CONV54_OUTPUT
CONV55_OUTPUT = CONV55_INPUT + (19200 * 4) # After CONV54_OUTPUT
Activation_Address("layer22", "ConvAct55", [CONV55_INPUT], [CONV55_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct55:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[55])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[55] + (576 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV55_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV55_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[55])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[55] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV55_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV55_OUTPUT + (19200 * 4))[2:].zfill(8)}\n") 

# "ConvAct56",
CONV56_INPUT  = CONV55_OUTPUT
CONV56_OUTPUT = CONV56_INPUT + (19200 * 4) # After CONV55_OUTPUT
Activation_Address("layer22", "ConvAct56", [CONV56_INPUT], [CONV56_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct56:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[56])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[56] + (12288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV56_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV56_INPUT + (19200 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[56])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[56] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV56_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV56_OUTPUT + (9600 * 4))[2:].zfill(8)}\n") 

# "ConvAct57",
CONV57_INPUT  = CONV56_OUTPUT
CONV57_OUTPUT = CONV57_INPUT + (9600 * 4) # After CONV56_OUTPUT
Activation_Address("layer22", "ConvAct57", [CONV57_INPUT], [CONV57_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct57:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[57])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[57] + (288 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV57_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV57_INPUT + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[57])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[57] + (64 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV57_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV57_OUTPUT + (9600 * 4))[2:].zfill(8)}\n") 

# "EWAdder0",
EWADDER0_INPUT1 = CONV51_OUTPUT1 + (9600 * 4) # After CONV51_OUTPUT1(Split0) --> Split1
EWADDER0_INPUT2 = CONV57_OUTPUT 
Activation_Address("layer22", "EWAdder0", [EWADDER0_INPUT1, EWADDER0_INPUT2], [EWADDER0_OUTPUT]) 
if DEBUG_ADDR:
    print(f"layer22 --> EWAdder0:") 
    print(f"\tLD_IN1: START: 0x{hex(EWADDER0_INPUT1)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT1 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN2: START: 0x{hex(EWADDER0_INPUT2)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_INPUT2 + (9600 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START: 0x{hex(EWADDER0_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(EWADDER0_OUTPUT + (9600 * 4))[2:].zfill(8)}\n")

# "Concat0",
# CONV51_OUTPUT1 + EWADDER0_OUTPUT 
Activation_Address("layer22", "Concat0", [None], [None])

# "ConvAct52"
CONV52_INPUT  = CONV51_OUTPUT1          # Concat (Split0 + Split1 + EWADDER0_OUTPUT)
CONV52_OUTPUT = CONV57_OUTPUT + (9600 * 4) # After CONV57_OUTPUT
Activation_Address("layer22", "ConvAct52", [CONV52_INPUT], [CONV52_OUTPUT])
if DEBUG_ADDR:
    print(f"layer22 --> ConvAct52:") 
    print(f"\tLD_WGT: START: 0x{hex(Weight_Address[52])[2:].zfill(8)}\t END: 0x{hex(Weight_Address[52] + (36864 * 4))[2:].zfill(8)}")
    print(f"\tLD_IN1: START: 0x{hex(CONV52_INPUT)[2:].zfill(8)}\t END: 0x{hex(CONV52_INPUT + (28800 * 4))[2:].zfill(8)}")
    print(f"\tLD_PARAM: START: 0x{hex(Bias_Address[52])[2:].zfill(8)}\t END: 0x{hex(Bias_Address[52] + (128 * 4))[2:].zfill(8)}")
    print(f"\tST_OUT1: START 0x{hex(CONV52_OUTPUT)[2:].zfill(8)}\t END: 0x{hex(CONV52_OUTPUT + (19200 * 4))[2:].zfill(8)}\n") 

####################################
if DEBUG_COMPILER: 
    YOLOv10n_Address_Map = {}
    for layer, ops in YOLOv10n_activation_address.items():
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
        YOLOv10n_Address_Map[layer] = {
            "Input_Address":input_list,
            "Output_Address":output_list 
        }
        # print(f"\nLayer: {layer}, Input_Address: {input_list}")
        # print(f"Layer: {layer}, Output_Address: {output_list}")