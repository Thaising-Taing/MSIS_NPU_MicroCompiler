from MSIS_NPU_Instruction_SetV1.InstructionSet_Microcode import *

########################
#        layer0        #
########################
# ConvActMax0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,    16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      480,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     640,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 77)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,2,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,1,    POST_PROC["ADD"],POST_PROC["ACTIVE"],POST_PROC["MAXPOOL"],False)
offset_write(OPCODE["LD_WGT"],    0x00000000)
offset_write(OPCODE["LD_IN1"],    0x01000000)
offset_write(OPCODE["LD_PARAM"],  0x10000000)
offset_write(OPCODE["ST_OUT1"],   0x0112c000)
ctrl_write(OPCODE["LYREND"])

########################
#        layer1        #
########################
# ConvAct1
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   32,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    16,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 1260)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 425)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 1,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00000240)
offset_write(OPCODE["LD_IN1"],    0x0112c000)
offset_write(OPCODE["LD_PARAM"],  0x10000020)
offset_write(OPCODE["ST_OUT1"],   0x01177000)
offset_write(OPCODE["ST_OUT2"],   0x0120d000)
ctrl_write(OPCODE["LYREND"])
# ConvAct3
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,    16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    16,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 198)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00000e40)
offset_write(OPCODE["LD_IN1"],    0x011c2000)
offset_write(OPCODE["LD_PARAM"],  0x100000a0)
offset_write(OPCODE["ST_OUT1"],   0x012ee000)
ctrl_write(OPCODE["LYREND"])
# ConvAct4
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,    16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    16,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 187)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00001740)
offset_write(OPCODE["LD_IN1"],    0x012ee000)
offset_write(OPCODE["LD_PARAM"],  0x100000c0)
offset_write(OPCODE["ST_OUT1"],   0x01339000)
ctrl_write(OPCODE["LYREND"])
# EWAdder0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,   16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x01339000)
offset_write(OPCODE["LD_IN2"],    0x01258000)
offset_write(OPCODE["ST_OUT1"],   0x012a3000)
ctrl_write(OPCODE["LYREND"])
# Concatenation
# ConvAct2
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   32,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    48,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 385)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00000540)
offset_write(OPCODE["LD_IN1"],    0x0120d000)
offset_write(OPCODE["LD_PARAM"],  0x10000060)
offset_write(OPCODE["ST_OUT1"],   0x01384000)
ctrl_write(OPCODE["LYREND"])

########################
#        layer2        #
########################
# ConvAct5
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   64,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    32,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      120,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     160,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 212)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,2,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00002040)
offset_write(OPCODE["LD_IN1"],    0x01384000)
offset_write(OPCODE["LD_PARAM"],  0x100000e0)
offset_write(OPCODE["ST_OUT1"],   0x0141a000)
ctrl_write(OPCODE["LYREND"])

########################
#        layer3        #
########################
# ConvAct6
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    64,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      60,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     80,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 264)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00006840)
offset_write(OPCODE["LD_IN1"],    0x0141a000)
offset_write(OPCODE["LD_PARAM"],  0x10000160)
offset_write(OPCODE["ST_OUT1"],   0x01465000)
ctrl_write(OPCODE["LYREND"])
# Conv7
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      60,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     80,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 425)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 425)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,2,1, 1,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x00009840)
offset_write(OPCODE["LD_IN1"],    0x01465000)
offset_write(OPCODE["LD_PARAM"],  0x10000260)
offset_write(OPCODE["ST_OUT1"],   0x014fb000)
offset_write(OPCODE["ST_OUT2"],   0x01520800)
ctrl_write(OPCODE["LYREND"])

########################
#        layer4        #
########################
# ConvAct8
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      30,   30)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     40,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 129)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,2,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00009cc0)
offset_write(OPCODE["LD_IN1"],    0x014fb000)
offset_write(OPCODE["LD_PARAM"],  0x10000360)
offset_write(OPCODE["ST_OUT1"],   0x01546000)
ctrl_write(OPCODE["LYREND"])

########################
#        layer5        #
########################
# ConvAct9
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 262)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00051cc0)
offset_write(OPCODE["LD_IN1"],    0x01546000)
offset_write(OPCODE["LD_PARAM"],  0x10000560)
offset_write(OPCODE["ST_OUT1"],   0x01558c00)
ctrl_write(OPCODE["LYREND"])
# MaxPool1
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,1,0,1, 1, 0,1,0,1)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["MAIN_PRCS"], 0,0,0,0,    POST_PROC["MAXPOOL"],False,False,False)
offset_write(OPCODE["LD_IN1"],    0x01558c00)
offset_write(OPCODE["ST_OUT1"],   0x01562200)
ctrl_write(OPCODE["LYREND"])
# MaxPool2
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,1,0,1, 1, 0,1,0,1)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["MAIN_PRCS"], 0,0,0,0,    POST_PROC["MAXPOOL"],False,False,False)
offset_write(OPCODE["LD_IN1"],    0x01562200)
offset_write(OPCODE["ST_OUT1"],   0x0156b800)
ctrl_write(OPCODE["LYREND"])
# MaxPool3
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,1,0,1, 1, 0,1,0,1)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["MAIN_PRCS"], 0,0,0,0,    POST_PROC["MAXPOOL"],False,False,False)
offset_write(OPCODE["LD_IN1"],    0x0156b800)
offset_write(OPCODE["ST_OUT1"],   0x01574e00)
ctrl_write(OPCODE["LYREND"])
# Concatenation
# ConvAct10
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    512,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 132)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x0005dcc0)
offset_write(OPCODE["LD_IN1"],    0x01558c00)
offset_write(OPCODE["LD_PARAM"],  0x10000660)
offset_write(OPCODE["ST_OUT1"],   0x0157e400)
ctrl_write(OPCODE["LYREND"])

########################
#        layer6        #
########################
# ConvAct11
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 303)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 275)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM3"], 14, 287)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM4"], 14, 287)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 3,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x0008dcc0)
offset_write(OPCODE["LD_IN1"],    0x0157e400)
offset_write(OPCODE["LD_PARAM"],  0x10000860)
offset_write(OPCODE["ST_OUT1"],   0x01591000)
offset_write(OPCODE["ST_OUT2"],   0x015a3c00)
offset_write(OPCODE["ST_OUT3"],   0x015b6800)
offset_write(OPCODE["ST_OUT4"],   0x015c9400)
ctrl_write(OPCODE["LYREND"])
# Conv13
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 10)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 211)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 1,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x000bdcc0)
offset_write(OPCODE["LD_IN1"],    0x0159a600)
offset_write(OPCODE["LD_PARAM"],  0x10000c60)
offset_write(OPCODE["ST_OUT1"],   0x015dc000)
offset_write(OPCODE["ST_OUT2"],   0x015eec00)
ctrl_write(OPCODE["LYREND"])
# AttnHead0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,      0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    32,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM_QK"], 16, 2730)
qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM1"], 14, 2730)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["MATMUL"],    0,0,0, 0,7, 0)
offset_write(OPCODE["LD_WGT"],  0x015dc000)
offset_write(OPCODE["LD_IN1"],  0x015de580)
offset_write(OPCODE["LD_IN2"],  0x015e0b00)
offset_write(OPCODE["ST_OUT1"], 0x01601800)
ctrl_write(OPCODE["LYREND"])
# AttnHead1
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,      0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   16,     16)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    32,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM_QK"], 16, 2730)
qparam_write(OPCODE["SETREG"], OPERAND1["QU_PARAM1"], 14, 2730)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["MATMUL"],    0,0,0, 0,7, 0)
offset_write(OPCODE["LD_WGT"],  0x015e5600)
offset_write(OPCODE["LD_IN1"],  0x015e7b80)
offset_write(OPCODE["LD_IN2"],  0x015ea100)
offset_write(OPCODE["ST_OUT1"], 0x01606300)
ctrl_write(OPCODE["LYREND"])
# Conv15_Head0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 65)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x000cfcc0)
offset_write(OPCODE["LD_IN1"],    0x015f3700)
offset_write(OPCODE["LD_PARAM"],  0x10000f60)
offset_write(OPCODE["ST_OUT1"],   0x0160ae00)
ctrl_write(OPCODE["LYREND"])
# Conv15_Head1
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 65)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x000cff00)
offset_write(OPCODE["LD_IN1"],    0x015fcd00)
offset_write(OPCODE["LD_PARAM"],  0x10000fe0)
offset_write(OPCODE["ST_OUT1"],   0x0160f900)
ctrl_write(OPCODE["LYREND"])
# EWAdder0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,1, 0)
offset_write(OPCODE["LD_IN1"],    0x01601800)
offset_write(OPCODE["LD_IN2"],    0x0160ae00)
offset_write(OPCODE["ST_OUT1"],   0x01614400)
ctrl_write(OPCODE["LYREND"])
# Conv14
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 84)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 88)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 1,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x000c9cc0)
offset_write(OPCODE["LD_IN1"],    0x01614400)
offset_write(OPCODE["LD_PARAM"],  0x10000e60)
offset_write(OPCODE["ST_OUT1"],   0x0161da00)
offset_write(OPCODE["ST_OUT2"],   0x01627000)
ctrl_write(OPCODE["LYREND"])
# EWAdder1
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x015ad200)
offset_write(OPCODE["LD_IN2"],    0x0161da00)
offset_write(OPCODE["ST_OUT1"],   0x01630600)
ctrl_write(OPCODE["LYREND"])
# ConvAct16
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 186)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x000d0140)
offset_write(OPCODE["LD_IN1"],    0x01630600)
offset_write(OPCODE["LD_PARAM"],  0x10001060)
offset_write(OPCODE["ST_OUT1"],   0x01639c00)
ctrl_write(OPCODE["LYREND"])
# Conv17
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 39)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 0,0,0,0,    POST_PROC["ADD"],False,False,False)
offset_write(OPCODE["LD_WGT"],    0x000dc140)
offset_write(OPCODE["LD_IN1"],    0x01639c00)
offset_write(OPCODE["LD_PARAM"],  0x10001260)
offset_write(OPCODE["ST_OUT1"],   0x0164c800)
ctrl_write(OPCODE["LYREND"])
# EWAdder2
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x015bfe00)
offset_write(OPCODE["LD_IN2"],    0x01627000)
offset_write(OPCODE["ST_OUT1"],   0x01655e00)
ctrl_write(OPCODE["LYREND"])
# EWAdder3
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x01655e00)
offset_write(OPCODE["LD_IN2"],    0x0164c800)
offset_write(OPCODE["ST_OUT1"],   0x015d2a00)
ctrl_write(OPCODE["LYREND"])
# Concatenation
# ConvAct12
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 107)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x000a5cc0)
offset_write(OPCODE["LD_IN1"],    0x015c9400)
offset_write(OPCODE["LD_PARAM"],  0x10000a60)
offset_write(OPCODE["ST_OUT1"],   0x0165f400)
ctrl_write(OPCODE["LYREND"])

########################
#        layer7        #
########################
# Upsample0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,   32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      40,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     30,  30)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["RESIZE"],       2,1,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x01520800)
offset_write(OPCODE["ST_OUT1"],   0x0167b600)
ctrl_write(OPCODE["LYREND"])

########################
#        layer8        #
########################
# ConvAct18
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      60,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     80,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 99)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x000e8140)
offset_write(OPCODE["LD_IN1"],    0x0167b600)
offset_write(OPCODE["LD_PARAM"],  0x10001360)
offset_write(OPCODE["ST_OUT1"],   0x01711600)
ctrl_write(OPCODE["LYREND"])

########################
#        layer9        #
########################
# ConvActMax19
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      60,   20)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     80,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,0,1,0, 0, 1,0,1,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 106)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    3,2,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,1,    POST_PROC["ADD"],POST_PROC["ACTIVE"],POST_PROC["MAXPOOL"],False)
offset_write(OPCODE["LD_WGT"],    0x00130140)
offset_write(OPCODE["LD_IN1"],    0x01711600)
offset_write(OPCODE["LD_PARAM"],  0x10001560)
offset_write(OPCODE["ST_OUT1"],   0x01672000)
ctrl_write(OPCODE["LYREND"])

########################
#        layer10        #
########################
# Concatenation

########################
#        layer11        #
########################
# ConvAct20
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    384,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 682)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM2"], 14, 182)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 1,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x00178140)
offset_write(OPCODE["LD_IN1"],    0x0165f400)
offset_write(OPCODE["LD_PARAM"],  0x10001660)
offset_write(OPCODE["ST_OUT1"],   0x0183d600)
offset_write(OPCODE["ST_OUT2"],   0x01850200)
ctrl_write(OPCODE["LYREND"])
# ConvAct22
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 546)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x001c0140)
offset_write(OPCODE["LD_IN1"],    0x01846c00)
offset_write(OPCODE["LD_PARAM"],  0x10001a60)
offset_write(OPCODE["ST_OUT1"],   0x0186c400)
ctrl_write(OPCODE["LYREND"])
# ConvAct23
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    128,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 442)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x001c05c0)
offset_write(OPCODE["LD_IN1"],    0x0186c400)
offset_write(OPCODE["LD_PARAM"],  0x10001b60)
offset_write(OPCODE["ST_OUT1"],   0x01875a00)
ctrl_write(OPCODE["LYREND"])
# ConvAct24
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 520)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x001cc5c0)
offset_write(OPCODE["LD_IN1"],    0x01875a00)
offset_write(OPCODE["LD_PARAM"],  0x10001d60)
offset_write(OPCODE["ST_OUT1"],   0x01888600)
ctrl_write(OPCODE["LYREND"])
# ConvAct25
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    256,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 315)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x001ccec0)
offset_write(OPCODE["LD_IN1"],    0x01888600)
offset_write(OPCODE["LD_PARAM"],  0x10001f60)
offset_write(OPCODE["ST_OUT1"],   0x0189b200)
ctrl_write(OPCODE["LYREND"])
# ConvAct26
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,     4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   1,1,1,1, 0, 1,1,1,1)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 385)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["DW_CONV"],    3,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x001d8ec0)
offset_write(OPCODE["LD_IN1"],    0x0189b200)
offset_write(OPCODE["LD_PARAM"],  0x10002060)
offset_write(OPCODE["ST_OUT1"],   0x018a4800)
ctrl_write(OPCODE["LYREND"])
# EWAdder0
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   128,  32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    4,    4)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["EWADDER"],       0,0,0, 0,0, 0)
offset_write(OPCODE["LD_IN1"],    0x01859800)
offset_write(OPCODE["LD_IN2"],    0x018a4800)
offset_write(OPCODE["ST_OUT1"],   0x01862e00)
ctrl_write(OPCODE["LYREND"])
# Concatenation
# ConvAct21
ctrl_write(OPCODE["INIT"])
setreg_write(OPCODE["SETREG"], OPERAND1["CURRENT_LYR"],   0,     0)
setreg_write(OPCODE["SETREG"], OPERAND1["OUT_CHANNEL"],   256,    32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_CHANNEL"],    384,     32)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_WIDTH"],      15,   15)
setreg_write(OPCODE["SETREG"], OPERAND1["IN_HEIGHT"],     20,  20)
ovppad_write(OPCODE["SETREG"], OPERAND1["OVERLAP_PAD"],   0,0,0,0, 0, 0,0,0,0)
qparam_write(OPCODE["SETREG"], OPERAND1[f"QU_PARAM1"], 14, 327)
mainop_write(OPCODE["OPTYPE"], FUNC_PARAM["D2_CONV"],    1,1,1, 0,7, 0)
postop_write(OPCODE["OPTYPE"], FUNC_PARAM["POST_PRCS"], 3,0,0,0,    POST_PROC["ADD"],POST_PROC["ACTIVE"],False,False)
offset_write(OPCODE["LD_WGT"],    0x0019c140)
offset_write(OPCODE["LD_IN1"],    0x01850200)
offset_write(OPCODE["LD_PARAM"],  0x10001860)
offset_write(OPCODE["ST_OUT1"],   0x018ade00)
ctrl_write(OPCODE["LYREND"])
ctrl_write(OPCODE["FINISH"])