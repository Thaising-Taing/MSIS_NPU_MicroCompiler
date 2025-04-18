###########################################
#                                         #
#     Simplified By: Thaising Taing       #
#        Credit: Ultralytic               #
#                                         #
###########################################

import torch 
import torch.nn as nn
import cv2 
import copy
import math
import os
import numpy as np
from pathlib import Path
from ultralytics.utils.checks import check_version
from ultralytics.utils import ops, LOGGER
from ultralytics.engine.results import Results
from ultralytics.data.augment import LetterBox
from Model.WeightLoader import WeightLoader_SSBQ
from MSIS_NPU_Instruction_SetV1.Data_Conversion import *

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


def pre_transform(im):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    return [letterbox(image=x) for x in im]


def Preprocessing(im):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.cuda()
    im = im.float()  # uint8 to fp32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def Postprocessing(preds, img, orig_imgs, img_path):
    max_det = 300
    conf = 0.25
    classes = None
    if isinstance(preds, dict):
        preds = preds["one2one"]

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if preds.shape[-1] == 6:
        pass
    else:
        preds = preds.transpose(-1, -2)
        bboxes, scores, labels = ops.v10postprocess(preds, max_det, preds.shape[-1]-4)
        bboxes = ops.xywh2xyxy(bboxes)
        preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

    mask = preds[..., 4] > conf
    if classes is not None:
        mask = mask & (preds[..., 5:6] == torch.tensor(classes, device=preds.device).unsqueeze(0)).any(2)
    
    preds = [p[mask[idx]] for idx, p in enumerate(preds)]

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 
             7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 
             13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
             21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
             29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
             36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
             44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 
             53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 
             63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
             73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=names, boxes=pred))
    return results


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    assert(distance.shape[dim] == 4)
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.LeakyReLU(negative_slope=0.125, inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        # self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(x))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.stride = [8., 16., 32.]
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        y = self.forward_feat(x, self.cv2, self.cv3)
        
        if self.training:
            return y

        return self.inference(y)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        if self.export:
            return dist2bbox(bboxes, anchors, xywh=False, dim=1)
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class YOLOv10n_Detector(Detect):

    max_det = 300

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), \
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), \
                                                nn.Conv2d(c3, self.nc, 1)) for i, x in enumerate(ch))

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
    
    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.export:
            one2many = super().forward(x)

        if not self.training:
            one2one = self.inference(one2one)
            if not self.export:
                return {"one2many": one2many, "one2one": one2one}
            else:
                assert(self.max_det != -1)
                boxes, scores, labels = ops.v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
        else:
            return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        super().bias_init()
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

def round_half_up(value):
    if isinstance(value, torch.Tensor):
        return torch.floor(value + 0.5)
    else:
        return math.floor(value + 0.5)
    
Weight_Array = [0] * 58
Bias_Array = [0] * 58

def Save_Weight_HW(weight, layer_idx, conv_idx): 
    Weight = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Weight_SW2HW(weight.shape[0], weight.shape[1], weight.shape[2], 
                                                             integer_to_uint8(weight.float().cpu().detach().numpy())))
    Weight_Array[conv_idx] = Weight.shape[0]


def Save_Bias_HW(bias, layer_idx, conv_idx):
    Bias = np.vectorize(lambda x:uint16_to_hexa(x).lower())(Bias_SW2HW(bias.shape[0], integer_to_uint16(bias.float().cpu().detach().numpy())))
    Bias_Array[conv_idx] = Bias.shape[0]


def Save_InFmap_HW(activation, layer_idx, conv_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_Output_HW(activation, layer_idx, conv_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Save_Head(activation, layer_idx, Head_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_Concat_Input_HW(activation, layer_idx, Concat_Idx, input_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Save_Concat_Output_HW(activation, layer_idx, Concat_Idx, output_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_EWAdd_Input_HW(activation, layer_idx, Add_idx, input_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Save_EWAdd_Output_HW(activation, layer_idx, Add_idx, output_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Save_MaxPool_Input_HW(activation, layer_idx, MaxPool_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_MaxPool_Output_HW(activation, layer_idx, MaxPool_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_MatMul_Input_HW(activation, layer_idx, MatMul_idx, input_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_MatMul_Output_HW(activation, layer_idx, MatMul_idx, output_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))
    

def Save_Upsample_Input_HW(activation, layer_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Save_Upsample_Output_HW(activation, layer_idx): 
    Activation = np.vectorize(lambda x:uint8_to_hexa(x).lower())(Activation_SW2HW(activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3], 
                                                             integer_to_uint8(activation.float().cpu().detach().numpy())))


def Print_Weight_Array():
    global Weight_Array, Bias_Array
    return Weight_Array, Bias_Array


class YOLOv10n(nn.Module):
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    
    def __init__(self, YOLOv10_Detector):
        super(YOLOv10n, self).__init__()
        # --------------------------------------- Layer0 -------------------------------------------
        self.Conv0  = nn.Conv2d(in_channels=3,   out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        # --------------------------------------- Layer1 -------------------------------------------
        self.Conv1  = nn.Conv2d(in_channels=16,  out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        # --------------------------------------- Layer2 -------------------------------------------
        self.Conv2  = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv3  = nn.Conv2d(in_channels=48,  out_channels=32, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv4  = nn.Conv2d(in_channels=16,  out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv5  = nn.Conv2d(in_channels=16,  out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer3 -------------------------------------------
        self.Conv6  = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        # --------------------------------------- Layer4 -------------------------------------------
        self.Conv7  = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv8  = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        # B0
        self.Conv9  = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv10 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # B1
        self.Conv11 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv12 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer5 -------------------------------------------
        self.Conv13 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv14 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=True)
        # --------------------------------------- Layer6 -------------------------------------------
        self.Conv15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv16 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv17 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv18 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv19 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv20 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer7 -------------------------------------------
        self.Conv21 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv22 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=True)
        # --------------------------------------- Layer8 -------------------------------------------
        self.Conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv24 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv25 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv26 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer9 -------------------------------------------
        self.Conv27 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv28 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        # ---------------------------------- Layer10: Attention ------------------------------------
        self.Conv29 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True) # cv1
        self.Conv30 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True) # cv2
        self.Conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True) # qkv
        self.Conv32 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True) # proj
        self.Conv33 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True) # pe
        self.Conv34 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True) # ffn0
        self.Conv35 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True) # ffn1
        # --------------------------------------- Layer11 -------------------------------------------
        # Upsample
        # --------------------------------------- Layer12 -------------------------------------------
        # Concat
        # --------------------------------------- Layer13 -------------------------------------------
        self.Conv36 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv37 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv38 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv39 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer14 -------------------------------------------
        # Upsample
        # --------------------------------------- Layer15 -------------------------------------------
        # Concat
        # --------------------------------------- Layer16 -------------------------------------------
        self.Conv40 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv41 = nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv42 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv43 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer17 -------------------------------------------
        self.Conv44 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        # --------------------------------------- Layer18 -------------------------------------------
        # Concat
        # --------------------------------------- Layer19 -------------------------------------------
        self.Conv45 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv46 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv47 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.Conv48 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # --------------------------------------- Layer20 -------------------------------------------
        self.Conv49 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv50 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=True)
        # --------------------------------------- Layer21 -------------------------------------------
        # Concat
        # --------------------------------------- Layer22 -------------------------------------------
        self.Conv51 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv52 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv53 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.Conv54 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv55 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=True)
        self.Conv56 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.Conv57 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        
        # --------------------------------- Additional Operation ------------------------------------
        self.ReplicatedPad = nn.ReplicationPad2d((0, 1, 0, 1))
        self.MaxPool  = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.LeakyReLU= nn.LeakyReLU(negative_slope=0.125, inplace=True)
        self.Upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.Identity = nn.Identity()
        self.Softmax  = nn.Softmax(dim=-1)
        
        # ------------------------------------ Attention Parameters -----------------------------------
        self.num_heads = 2 
        self.head_dim = 128 // self.num_heads
        self.attn_ratio = 0.5
        self.key_dim = int(self.head_dim * self.attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # ------------------------------------ YOLOv10_Detector --------------------------------------
        self.detector = YOLOv10_Detector.cuda()
        if not self.training:
            self.detector.eval() # --> Changing to Evaluation Model
    
    
    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    
    
    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        print(f"Distance shape: {distance.shape}")
        print(f"Anchor points shape: {anchor_points.shape}")
        assert(distance.shape[dim] == 4)
        lt, rb = distance.split([2, 2], dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
    
    
    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        print(f"Bounding boxes shape: {bboxes.shape}")
        return self.dist2bbox(bboxes, anchors, xywh=True, dim=1)
    
    
    def forward(self, InFmap, qw, Zpw, Y_Scale, ZP, Scale_fmap, q_min=-128, q_max=127, save_data=True):
        Hardware_Scale = []
        bit = 15

        # --------------------------------------- Layer0 ------------------------------------------
        # Quantization
        SF_Idx = 0
        InFmap = torch.clamp(torch.round(InFmap / Y_Scale[SF_Idx]), min=q_min, max=q_max)
        
        Output_Layer0 = torch.floor(self.LeakyReLU(self.Conv0(InFmap))).int()

        # -------------------------------------- Layer1 -------------------------------------------
        # Quantization: 
        SF_Idx = 1
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=0
        Output_Layer0 = torch.clip(((((Output_Layer0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=InFmap, layer_idx=0, conv_idx=0)
            Save_Weight_HW(weight=self.Conv0.weight, layer_idx=0, conv_idx=0)
            Save_Bias_HW(bias=self.Conv0.bias, layer_idx=0, conv_idx=0)
            Save_Output_HW(activation=Output_Layer0, layer_idx=0, conv_idx=0)  
        
        Output_Layer1 = torch.floor(self.LeakyReLU(self.Conv1(Output_Layer0))).int()

        # ----------------------------------- Layer2: C2f ----------------------------------------
        # cv1:
        # Quantization: 
        SF_Idx = 2
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=1
        Output_Layer1 = torch.clip(((((Output_Layer1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer0, layer_idx=1, conv_idx=1)
            Save_Weight_HW(weight=self.Conv1.weight, layer_idx=1, conv_idx=1)
            Save_Bias_HW(bias=self.Conv1.bias, layer_idx=1, conv_idx=1)
            Save_Output_HW(activation=Output_Layer1, layer_idx=1, conv_idx=1)  
        
        Output_Layer2_cv1 = torch.floor(self.LeakyReLU(self.Conv2(Output_Layer1))).int()
        
        # Splitting
        Split_Target = (16, 16)
        Output_Layer2_split_0, Output_Layer2_split_1 = torch.split(Output_Layer2_cv1, Split_Target, dim=1)
        
        # BottleNeck: cv1
        # Quantization: 
        SF_Idx = 4; Prev_SF_Idx = 2
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=2
        Output_Layer2_split_1_conv4 = torch.clip(((((Output_Layer2_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer1, layer_idx=2, conv_idx=2)
            Save_Weight_HW(weight=self.Conv2.weight, layer_idx=2, conv_idx=2)
            Save_Bias_HW(bias=self.Conv2.bias, layer_idx=2, conv_idx=2)
            Save_Output_HW(activation=Output_Layer2_split_1_conv4, layer_idx=2, conv_idx=2)  
        
        Output_Layer2_Bott0 = torch.floor(self.LeakyReLU(self.Conv4(Output_Layer2_split_1_conv4))).int()
        
        # BottleNeck: cv2
        # Quantization: 
        SF_Idx = 5
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=3
        Output_Layer2_Bott0 = torch.clip(((((Output_Layer2_Bott0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer2_split_1_conv4, layer_idx=2, conv_idx=4)
            Save_Weight_HW(weight=self.Conv4.weight, layer_idx=2, conv_idx=4)
            Save_Bias_HW(bias=self.Conv4.bias, layer_idx=2, conv_idx=4)
            Save_Output_HW(activation=Output_Layer2_Bott0, layer_idx=2, conv_idx=4)  
        
        Output_Layer2_Bott1 = torch.floor(self.LeakyReLU(self.Conv5(Output_Layer2_Bott0))).int()
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 3; Prev_SF_Idx = 5
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=4
        Output_Layer2_Bott1_Add = torch.clip(((((Output_Layer2_Bott1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer2_Bott0, layer_idx=2, conv_idx=5)
            Save_Weight_HW(weight=self.Conv5.weight, layer_idx=2, conv_idx=5)
            Save_Bias_HW(bias=self.Conv5.bias, layer_idx=2, conv_idx=5)
            Save_Output_HW(activation=Output_Layer2_Bott1_Add, layer_idx=2, conv_idx=5)  
        
        # Quantization: 
        SF_Idx = 3
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=5
        Output_Layer2_split_1_Add = torch.clip(((((Output_Layer2_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder0
        Output_Layer2_Add = torch.clip(torch.add(Output_Layer2_Bott1_Add, Output_Layer2_split_1_Add), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer2_Bott1_Add, Add_idx=0, layer_idx=2, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer2_split_1_Add, Add_idx=0, layer_idx=2, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer2_Add, Add_idx=0, layer_idx=2, output_idx=0)
        
        # Concentination
        # Quantization: 
        SF_Idx = 3
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=6
        Output_Layer2_split_0_Cat = torch.clip(((((Output_Layer2_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 3
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=7
        Output_Layer2_split_1_Cat = torch.clip(((((Output_Layer2_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer2_Concat = torch.cat((Output_Layer2_split_0_Cat, Output_Layer2_split_1_Cat, Output_Layer2_Add), dim=1)
        
        # Save_Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer2_split_0_Cat, Concat_Idx=0, layer_idx=2, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer2_split_1_Cat, Concat_Idx=0, layer_idx=2, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer2_Add, Concat_Idx=0, layer_idx=2, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer2_Concat, Concat_Idx=0, layer_idx=2, output_idx=0)
        
        # cv2
        Output_Layer2_cv2 = torch.floor(self.LeakyReLU(self.Conv3(Output_Layer2_Concat))).int()
        
        # ------------------------------------------ Layer3 --------------------------------------------
        # Quantization: 
        SF_Idx = 6; Prev_SF_Idx = 3
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=8
        Output_Layer2_cv2 = torch.clip((((Output_Layer2_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1, min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer2_Concat, layer_idx=2, conv_idx=3)
            Save_Weight_HW(weight=self.Conv3.weight, layer_idx=2, conv_idx=3)
            Save_Bias_HW(bias=self.Conv3.bias, layer_idx=2, conv_idx=3)
            Save_Output_HW(activation=Output_Layer2_cv2, layer_idx=2, conv_idx=3)  

        Output_Layer3 = torch.floor(self.LeakyReLU(self.Conv6(Output_Layer2_cv2))).int()
        
        # -------------------------------------- Layer4:C2f -------------------------------------------
        # cv1:
        # Quantization: 
        SF_Idx = 7
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=9
        Output_Layer3 = torch.clip(((((Output_Layer3 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer2_cv2, layer_idx=3, conv_idx=6)
            Save_Weight_HW(weight=self.Conv6.weight, layer_idx=3, conv_idx=6)
            Save_Bias_HW(bias=self.Conv6.bias, layer_idx=3, conv_idx=6)
            Save_Output_HW(activation=Output_Layer3, layer_idx=3, conv_idx=6)  
        
        Output_Layer4_cv1 = torch.floor(self.LeakyReLU(self.Conv7(Output_Layer3))).int()
        
        # Splitting
        Split_Target = (32, 32)
        Output_Layer4_split_0, Output_Layer4_split_1 = torch.split(Output_Layer4_cv1, Split_Target, dim=1)
        
        # BottleNeck0: cv1
        # Quantization: 
        SF_Idx = 9; Prev_SF_Idx = 7
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=10
        Output_Layer4_split_1_conv9 = torch.clip(((((Output_Layer4_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer3, layer_idx=4, conv_idx=7)
            Save_Weight_HW(weight=self.Conv7.weight, layer_idx=4, conv_idx=7)
            Save_Bias_HW(bias=self.Conv7.bias, layer_idx=4, conv_idx=7)
            Save_Output_HW(activation=Output_Layer4_split_1_conv9, layer_idx=4, conv_idx=7)  

        Output_Layer4_Bott0_cv1 = torch.floor(self.LeakyReLU(self.Conv9(Output_Layer4_split_1_conv9))).int()
        
        # BottleNeck0: cv2
        # Quantization: 
        SF_Idx = 10
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=11
        Output_Layer4_Bott0_cv1 = torch.clip(((((Output_Layer4_Bott0_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_Bott0_cv1, layer_idx=4, conv_idx=9)
            Save_Weight_HW(weight=self.Conv9.weight, layer_idx=4, conv_idx=9)
            Save_Bias_HW(bias=self.Conv9.bias, layer_idx=4, conv_idx=9)
            Save_Output_HW(activation=Output_Layer4_Bott0_cv1, layer_idx=4, conv_idx=9)  
        
        Output_Layer4_Bott0_cv2 = torch.floor(self.LeakyReLU(self.Conv10(Output_Layer4_Bott0_cv1))).int()
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 11
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=12
        Output_Layer4_Bott0_cv2_Add = torch.clip(((((Output_Layer4_Bott0_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_Bott0_cv1, layer_idx=4, conv_idx=10)
            Save_Weight_HW(weight=self.Conv10.weight, layer_idx=4, conv_idx=10)
            Save_Bias_HW(bias=self.Conv10.bias, layer_idx=4, conv_idx=10)
            Save_Output_HW(activation=Output_Layer4_Bott0_cv2_Add, layer_idx=4, conv_idx=10)  
        
        # Quantization: 
        SF_Idx = 11; Prev_SF_Idx = 7
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=13
        Output_Layer4_split_1_Add = torch.clip(((((Output_Layer4_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder0
        Output_Layer4_Add1 = torch.clip(torch.add(Output_Layer4_Bott0_cv2_Add, Output_Layer4_split_1_Add), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer4_Bott0_cv2_Add, Add_idx=0, layer_idx=4, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer4_split_1_Add, Add_idx=0, layer_idx=4, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer4_Add1, Add_idx=0, layer_idx=4, output_idx=0)
        
        # BottleNeck1: cv1
        Output_Layer4_Bott1_cv1 = torch.floor(self.LeakyReLU(self.Conv11(Output_Layer4_Add1))).int()
        
        # BottleNeck1: cv2
        # Quantization: 
        SF_Idx = 12
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=14
        Output_Layer4_Bott1_cv1_conv12 = torch.clip(((((Output_Layer4_Bott1_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_Add1, layer_idx=4, conv_idx=11)
            Save_Weight_HW(weight=self.Conv11.weight, layer_idx=4, conv_idx=11)
            Save_Bias_HW(bias=self.Conv11.bias, layer_idx=4, conv_idx=11)
            Save_Output_HW(activation=Output_Layer4_Bott1_cv1_conv12, layer_idx=4, conv_idx=11)  
        
        Output_Layer4_Bott1_cv2 = torch.floor(self.LeakyReLU(self.Conv12(Output_Layer4_Bott1_cv1_conv12))).int()
        
        # Additional: Element-Wise Adder: EWAdder1
        # Quantization: 
        SF_Idx = 8; Prev_SF_Idx = 10
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=15
        Output_Layer4_Bott0_cv2_Addi1 = torch.clip(((((Output_Layer4_Bott0_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 8
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=16
        Output_Layer4_split_1_Addi1 = torch.clip(((((Output_Layer4_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder1
        Output_Layer4_Addi1 = torch.clip(torch.add(Output_Layer4_Bott0_cv2_Addi1, Output_Layer4_split_1_Addi1), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer4_Bott0_cv2_Addi1, Add_idx=1, layer_idx=4, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer4_split_1_Addi1, Add_idx=1, layer_idx=4, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer4_Addi1, Add_idx=1, layer_idx=4, output_idx=0)
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 8; Prev_SF_Idx = 12
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=17
        Output_Layer4_Bott1_cv2_Add2 = torch.clip(((((Output_Layer4_Bott1_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_Bott1_cv1_conv12, layer_idx=4, conv_idx=12)
            Save_Weight_HW(weight=self.Conv12.weight, layer_idx=4, conv_idx=12)
            Save_Bias_HW(bias=self.Conv12.bias, layer_idx=4, conv_idx=12)
            Save_Output_HW(activation=Output_Layer4_Bott1_cv2_Add2, layer_idx=4, conv_idx=12)  
        
        # EWAdder2
        Output_Layer4_Add2 = torch.clip(torch.add(Output_Layer4_Addi1, Output_Layer4_Bott1_cv2_Add2), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer4_Addi1, Add_idx=2, layer_idx=4, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer4_Bott1_cv2_Add2, Add_idx=2, layer_idx=4, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer4_Add2, Add_idx=2, layer_idx=4, output_idx=0)
        
        # Concentination
        # Quantization: 
        SF_Idx = 8
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=18
        Output_Layer4_split_0_Cat = torch.clip(((((Output_Layer4_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 8
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=19
        Output_Layer4_split_1_Cat = torch.clip(((((Output_Layer4_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer4_Concat = torch.cat((Output_Layer4_split_0_Cat, Output_Layer4_split_1_Cat, Output_Layer4_Addi1, Output_Layer4_Add2), dim=1)
        
        # Save_Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer4_split_0_Cat, Concat_Idx=0, layer_idx=4, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer4_split_1_Cat, Concat_Idx=0, layer_idx=4, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer4_Addi1, Concat_Idx=0, layer_idx=4, input_idx=2)
            Save_Concat_Input_HW(activation=Output_Layer4_Add2, Concat_Idx=0, layer_idx=4, input_idx=3)
            Save_Concat_Output_HW(activation=Output_Layer4_Concat, Concat_Idx=0, layer_idx=4, output_idx=0)
        
        # cv2:
        Output_Layer4_cv2 = torch.floor(self.LeakyReLU(self.Conv8(Output_Layer4_Concat))).int() # --> SCDown (L5) & L15
        
        # -------------------------------------- Layer5: SCDown -------------------------------------------
        ###################################################################################################
        #                         Spatial Convolution with Downsampling                                   #
        ###################################################################################################
        # cv1: 
        # Quantization: 
        SF_Idx = 13; Prev_SF_Idx = 8
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=20
        Output_Layer4_cv2_conv13 = torch.clip(((((Output_Layer4_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_Concat, layer_idx=4, conv_idx=8)
            Save_Weight_HW(weight=self.Conv8.weight, layer_idx=4, conv_idx=8)
            Save_Bias_HW(bias=self.Conv8.bias, layer_idx=4, conv_idx=8)
            Save_Output_HW(activation=Output_Layer4_cv2_conv13, layer_idx=4, conv_idx=8)  
        
        Output_Layer5_cv1 = torch.floor(self.LeakyReLU(self.Conv13(Output_Layer4_cv2_conv13))).int()
        
        # cv2: 
        # Quantization: 
        SF_Idx = 14
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=21
        Output_Layer5_cv1 = torch.clip(((((Output_Layer5_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer4_cv2_conv13, layer_idx=5, conv_idx=13)
            Save_Weight_HW(weight=self.Conv13.weight, layer_idx=5, conv_idx=13)
            Save_Bias_HW(bias=self.Conv13.bias, layer_idx=5, conv_idx=13)
            Save_Output_HW(activation=Output_Layer5_cv1, layer_idx=5, conv_idx=13)  
        
        Output_Layer5_cv2 = self.Conv14(Output_Layer5_cv1).int()
        
        # -------------------------------------- Layer6 -------------------------------------------
        # cv1:
        # Quantization: 
        SF_Idx = 15
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=22
        Output_Layer5_cv2 = torch.clip(((((Output_Layer5_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer5_cv1, layer_idx=5, conv_idx=14)
            Save_Weight_HW(weight=self.Conv14.weight, layer_idx=5, conv_idx=14)
            Save_Bias_HW(bias=self.Conv14.bias, layer_idx=5, conv_idx=14)
            Save_Output_HW(activation=Output_Layer5_cv2, layer_idx=5, conv_idx=14)  
        
        Output_Layer6_cv1 = torch.floor(self.LeakyReLU(self.Conv15(Output_Layer5_cv2))).int()
        
        # Splitting
        Split_Target = (64, 64)
        Output_Layer6_split_0, Output_Layer6_split_1 = torch.split(Output_Layer6_cv1, Split_Target, dim=1)
        
        # BottleNeck0: cv1
        # Quantization: 
        SF_Idx = 17; Prev_SF_Idx = 15
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=23
        Output_Layer6_split_1_conv17 = torch.clip(((((Output_Layer6_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer5_cv2, layer_idx=6, conv_idx=15)
            Save_Weight_HW(weight=self.Conv15.weight, layer_idx=6, conv_idx=15)
            Save_Bias_HW(bias=self.Conv15.bias, layer_idx=6, conv_idx=15)
            Save_Output_HW(activation=Output_Layer6_split_1_conv17, layer_idx=6, conv_idx=15)  
        
        Output_Layer6_Bott0_cv1 = torch.floor(self.LeakyReLU(self.Conv17(Output_Layer6_split_1_conv17))).int()
        
        # BottleNeck0: cv2
        # Quantization: 
        SF_Idx = 18
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=24
        Output_Layer6_Bott0_cv1 = torch.clip(((((Output_Layer6_Bott0_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_split_1_conv17, layer_idx=6, conv_idx=17)
            Save_Weight_HW(weight=self.Conv17.weight, layer_idx=6, conv_idx=17)
            Save_Bias_HW(bias=self.Conv17.bias, layer_idx=6, conv_idx=17)
            Save_Output_HW(activation=Output_Layer6_Bott0_cv1, layer_idx=6, conv_idx=17)  
        
        Output_Layer6_Bott0_cv2 = torch.floor(self.LeakyReLU(self.Conv18(Output_Layer6_Bott0_cv1))).int()
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 19
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=25
        Output_Layer6_Bott0_cv2_Add = torch.clip(((((Output_Layer6_Bott0_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_Bott0_cv1, layer_idx=6, conv_idx=18)
            Save_Weight_HW(weight=self.Conv18.weight, layer_idx=6, conv_idx=18)
            Save_Bias_HW(bias=self.Conv18.bias, layer_idx=6, conv_idx=18)
            Save_Output_HW(activation=Output_Layer6_Bott0_cv2_Add, layer_idx=6, conv_idx=18)  
        
        # Quantization: 
        SF_Idx = 19; Prev_SF_Idx = 15
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=26
        Output_Layer6_split_1_Add = torch.clip(((((Output_Layer6_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder0
        Output_Layer6_Add1 = torch.clip(torch.add(Output_Layer6_Bott0_cv2_Add, Output_Layer6_split_1_Add), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer6_Bott0_cv2_Add, Add_idx=0, layer_idx=6, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer6_split_1_Add, Add_idx=0, layer_idx=6, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer6_Add1, Add_idx=0, layer_idx=6, output_idx=0)
        
        # BottleNeck1: cv1
        Output_Layer6_Bott1_cv1 = torch.floor(self.LeakyReLU(self.Conv19(Output_Layer6_Add1))).int()
        
        # BottleNeck1: cv2
        # Quantization: 
        SF_Idx = 20
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=27
        Output_Layer6_Bott1_cv1 = torch.clip(((((Output_Layer6_Bott1_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_Add1, layer_idx=6, conv_idx=19)
            Save_Weight_HW(weight=self.Conv19.weight, layer_idx=6, conv_idx=19)
            Save_Bias_HW(bias=self.Conv19.bias, layer_idx=6, conv_idx=19)
            Save_Output_HW(activation=Output_Layer6_Bott1_cv1, layer_idx=6, conv_idx=19)  
        
        Output_Layer6_Bott1_cv2 = torch.floor(self.LeakyReLU(self.Conv20(Output_Layer6_Bott1_cv1))).int()
        
        # Additional: Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 16; Prev_SF_Idx = 18
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=28
        Output_Layer6_Bott0_cv2_Addi1 = torch.clip(((((Output_Layer6_Bott0_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 16
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=29
        Output_Layer6_split_1_Addi1 = torch.clip(((((Output_Layer6_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder1
        Output_Layer6_Addi1 = torch.clip(torch.add(Output_Layer6_Bott0_cv2_Addi1, Output_Layer6_split_1_Addi1), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer6_Bott0_cv2_Addi1, Add_idx=1, layer_idx=6, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer6_split_1_Addi1, Add_idx=1, layer_idx=6, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer6_Addi1, Add_idx=1, layer_idx=6, output_idx=0)
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 16; Prev_SF_Idx = 20
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=30
        Output_Layer6_Bott1_cv2_Add2 = torch.clip(((((Output_Layer6_Bott1_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_Bott1_cv1, layer_idx=6, conv_idx=20)
            Save_Weight_HW(weight=self.Conv20.weight, layer_idx=6, conv_idx=20)
            Save_Bias_HW(bias=self.Conv20.bias, layer_idx=6, conv_idx=20)
            Save_Output_HW(activation=Output_Layer6_Bott1_cv2_Add2, layer_idx=6, conv_idx=20)  
        
        # EWAdder2
        Output_Layer6_Add2 = torch.clip(torch.add(Output_Layer6_Addi1, Output_Layer6_Bott1_cv2_Add2), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer6_Addi1, Add_idx=2, layer_idx=6, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer6_Bott1_cv2_Add2, Add_idx=2, layer_idx=6, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer6_Add2, Add_idx=2, layer_idx=6, output_idx=0)
        
        # Concentination
        # Quantization: 
        SF_Idx = 16
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=31
        Output_Layer6_split_0_Cat = torch.clip(((((Output_Layer6_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 16
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=32
        Output_Layer6_split_1_Cat = torch.clip(((((Output_Layer6_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer6_Concat = torch.cat([Output_Layer6_split_0_Cat, Output_Layer6_split_1_Cat, Output_Layer6_Addi1, Output_Layer6_Add2], dim=1)
        
        # Save_Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer6_split_0_Cat, Concat_Idx=0, layer_idx=6, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer6_split_1_Cat, Concat_Idx=0, layer_idx=6, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer6_Addi1, Concat_Idx=0, layer_idx=6, input_idx=2)
            Save_Concat_Input_HW(activation=Output_Layer6_Add2, Concat_Idx=0, layer_idx=6, input_idx=3)
            Save_Concat_Output_HW(activation=Output_Layer6_Concat, Concat_Idx=0, layer_idx=6, output_idx=0)
        
        # cv2:
        Output_Layer6_cv2 = torch.floor(self.LeakyReLU(self.Conv16(Output_Layer6_Concat))).int() # --> SCDown (L7) & L12
        
        # -------------------------------------- Layer7: SCDown -------------------------------------------
        ###################################################################################################
        #                         Spatial Convolution with Downsampling                                   #
        ###################################################################################################
        # cv1: 
        # Quantization: 
        SF_Idx = 21; Prev_SF_Idx = 16
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=33
        Output_Layer6_cv2_conv21 = torch.clip(((((Output_Layer6_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_Concat, layer_idx=6, conv_idx=16)
            Save_Weight_HW(weight=self.Conv16.weight, layer_idx=6, conv_idx=16)
            Save_Bias_HW(bias=self.Conv16.bias, layer_idx=6, conv_idx=16)
            Save_Output_HW(activation=Output_Layer6_cv2_conv21, layer_idx=6, conv_idx=16)  
        
        Output_Layer7_cv1 = torch.floor(self.LeakyReLU(self.Conv21(Output_Layer6_cv2_conv21))).int()
        
        # cv2: 
        # Quantization: 
        SF_Idx = 22
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=34
        Output_Layer7_cv1 = torch.clip(((((Output_Layer7_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer6_cv2_conv21, layer_idx=7, conv_idx=21)
            Save_Weight_HW(weight=self.Conv21.weight, layer_idx=7, conv_idx=21)
            Save_Bias_HW(bias=self.Conv21.bias, layer_idx=7, conv_idx=21)
            Save_Output_HW(activation=Output_Layer7_cv1, layer_idx=7, conv_idx=21)  
        
        Output_Layer7_cv2 = self.Conv22(Output_Layer7_cv1).int()
        
        # ----------------------------------- Layer8: C2f ----------------------------------------
        # cv1:
        # Quantization: 
        SF_Idx = 23
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=35
        Output_Layer7_cv2 = torch.clip(((((Output_Layer7_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer7_cv1, layer_idx=7, conv_idx=22)
            Save_Weight_HW(weight=self.Conv22.weight, layer_idx=7, conv_idx=22)
            Save_Bias_HW(bias=self.Conv22.bias, layer_idx=7, conv_idx=22)
            Save_Output_HW(activation=Output_Layer7_cv2, layer_idx=7, conv_idx=22)  
        
        Output_Layer8_cv1 = torch.floor(self.LeakyReLU(self.Conv23(Output_Layer7_cv2))).int()
        
        # Splitting
        Split_Target = (128, 128)
        Output_Layer8_split_0, Output_Layer8_split_1 = torch.split(Output_Layer8_cv1, Split_Target, dim=1)
        
        # BottleNeck: cv1
        # Quantization: 
        SF_Idx = 25; Prev_SF_Idx = 23
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=36
        Output_Layer8_split_1_conv25 = torch.clip(((((Output_Layer8_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer7_cv2, layer_idx=8, conv_idx=23)
            Save_Weight_HW(weight=self.Conv23.weight, layer_idx=8, conv_idx=23)
            Save_Bias_HW(bias=self.Conv23.bias, layer_idx=8, conv_idx=23)
            Save_Output_HW(activation=Output_Layer8_split_1_conv25, layer_idx=8, conv_idx=23)  
        
        Output_Layer8_Bott0 = torch.floor(self.LeakyReLU(self.Conv25(Output_Layer8_split_1_conv25))).int()
        
        # BottleNeck: cv2
        # Quantization: 
        SF_Idx = 26
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=37
        Output_Layer8_Bott0 = torch.clip(((((Output_Layer8_Bott0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer8_split_1_conv25, layer_idx=8, conv_idx=25)
            Save_Weight_HW(weight=self.Conv25.weight, layer_idx=8, conv_idx=25)
            Save_Bias_HW(bias=self.Conv25.bias, layer_idx=8, conv_idx=25)
            Save_Output_HW(activation=Output_Layer8_Bott0, layer_idx=8, conv_idx=25) 
        
        Output_Layer8_Bott1 = torch.floor(self.LeakyReLU(self.Conv26(Output_Layer8_Bott0))).int()
        
        # Element-Wise Adder: 
        # Quantization: 
        SF_Idx = 24; Prev_SF_Idx = 26
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=38
        Output_Layer8_Bott1_Add = torch.clip(((((Output_Layer8_Bott1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer8_Bott0, layer_idx=8, conv_idx=26)
            Save_Weight_HW(weight=self.Conv26.weight, layer_idx=8, conv_idx=26)
            Save_Bias_HW(bias=self.Conv26.bias, layer_idx=8, conv_idx=26)
            Save_Output_HW(activation=Output_Layer8_Bott1, layer_idx=8, conv_idx=26) 
        
        # Quantization: 
        SF_Idx = 24
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=39
        Output_Layer8_split_1_Add = torch.clip(((((Output_Layer8_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder0
        Output_Layer8_Add = torch.clip(torch.add(Output_Layer8_Bott1_Add, Output_Layer8_split_1_Add), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer8_Bott1_Add, Add_idx=0, layer_idx=8, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer8_split_1_Add, Add_idx=0, layer_idx=8, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer8_Add, Add_idx=0, layer_idx=8, output_idx=0)
        
        # Concentination
        # Quantization: 
        SF_Idx = 24
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=40
        Output_Layer8_split_0_Cat = torch.clip(((((Output_Layer8_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 24
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=41
        Output_Layer8_split_1_Cat = torch.clip(((((Output_Layer8_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer8_Concat = torch.cat((Output_Layer8_split_0_Cat, Output_Layer8_split_1_Cat, Output_Layer8_Add), dim=1)
        
        # Save_Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer8_split_0_Cat, Concat_Idx=0, layer_idx=8, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer8_split_1_Cat, Concat_Idx=0, layer_idx=8, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer8_Add, Concat_Idx=0, layer_idx=8, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer8_Concat, Concat_Idx=0, layer_idx=8, output_idx=0)
        
        # cv2
        Output_Layer8_cv2 = torch.floor(self.LeakyReLU(self.Conv24(Output_Layer8_Concat))).int()
        
        # -------------------------------------- Layer9: SPPF -------------------------------------------
        ###################################################################################################
        #                               Spatial Pyramid Pooling Fast                                      #
        ###################################################################################################
        # cv1: 
        # Quantization: 
        SF_Idx = 27; Prev_SF_Idx = 24
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=42
        Output_Layer8_cv2_conv27 = torch.clip(((((Output_Layer8_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer8_Concat, layer_idx=8, conv_idx=24)
            Save_Weight_HW(weight=self.Conv24.weight, layer_idx=8, conv_idx=24)
            Save_Bias_HW(bias=self.Conv24.bias, layer_idx=8, conv_idx=24)
            Save_Output_HW(activation=Output_Layer8_cv2_conv27, layer_idx=8, conv_idx=24) 
        
        Output_Layer9_cv1 = torch.floor(self.LeakyReLU(self.Conv27(Output_Layer8_cv2_conv27))).int()
        
        # MaxPool1 
        # Quantization: 
        SF_Idx = 28
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=43
        Output_Layer9_cv1_MaxPool1 = torch.clip(((((Output_Layer9_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer8_cv2_conv27, layer_idx=9, conv_idx=27)
            Save_Weight_HW(weight=self.Conv27.weight, layer_idx=9, conv_idx=27)
            Save_Bias_HW(bias=self.Conv27.bias, layer_idx=9, conv_idx=27)
            Save_Output_HW(activation=Output_Layer9_cv1_MaxPool1, layer_idx=9, conv_idx=27) 
        
        Output_Layer9_MaxPool1 = self.MaxPool(self.ReplicatedPad(Output_Layer9_cv1_MaxPool1))
        
        # Save_Data
        if save_data: 
            Save_MaxPool_Input_HW(activation=Output_Layer9_cv1_MaxPool1, layer_idx=9, MaxPool_idx=1)
            Save_MaxPool_Output_HW(activation=Output_Layer9_MaxPool1, layer_idx=9, MaxPool_idx=1)
        
        # MaxPool2
        Output_Layer9_MaxPool2 = self.MaxPool(self.ReplicatedPad(Output_Layer9_MaxPool1))
        
        # Save_Data
        if save_data: 
            Save_MaxPool_Input_HW(activation=Output_Layer9_MaxPool1, layer_idx=9, MaxPool_idx=2)
            Save_MaxPool_Output_HW(activation=Output_Layer9_MaxPool2, layer_idx=9, MaxPool_idx=2)
        
        # MaxPool3
        Output_Layer9_MaxPool3 = self.MaxPool(self.ReplicatedPad(Output_Layer9_MaxPool2))
        
        # Save_Data
        if save_data: 
            Save_MaxPool_Input_HW(activation=Output_Layer9_MaxPool2, layer_idx=9, MaxPool_idx=3)
            Save_MaxPool_Output_HW(activation=Output_Layer9_MaxPool3, layer_idx=9, MaxPool_idx=3)

        # Concentination
        Output_Layer9_Concat = torch.cat((Output_Layer9_cv1_MaxPool1, Output_Layer9_MaxPool1, Output_Layer9_MaxPool2, Output_Layer9_MaxPool3), dim=1)
        
        # Save_Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer9_cv1_MaxPool1,  Concat_Idx=0, layer_idx=9, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer9_MaxPool1, Concat_Idx=0, layer_idx=9, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer9_MaxPool2, Concat_Idx=0, layer_idx=9, input_idx=2)
            Save_Concat_Input_HW(activation=Output_Layer9_MaxPool3, Concat_Idx=0, layer_idx=9, input_idx=3)
            Save_Concat_Output_HW(activation=Output_Layer9_Concat,  Concat_Idx=0, layer_idx=9, output_idx=0)

        # cv2: 
        Output_Layer9_cv2 = torch.floor(self.LeakyReLU(self.Conv28(Output_Layer9_Concat))).int()
        
        # -------------------------------------- Layer10: PSA -------------------------------------------
        ###################################################################################################
        #                                   Pyramid Self-Attention                                        #
        ###################################################################################################
        # cv1: 
        # Quantization: 
        SF_Idx = 29
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=44
        Output_Layer9_cv2_conv29 = torch.clip(((((Output_Layer9_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer9_Concat, layer_idx=9, conv_idx=28)
            Save_Weight_HW(weight=self.Conv28.weight, layer_idx=9, conv_idx=28)
            Save_Bias_HW(bias=self.Conv28.bias, layer_idx=9, conv_idx=28)
            Save_Output_HW(activation=Output_Layer9_cv2_conv29, layer_idx=9, conv_idx=28)
        
        Output_Layer10_cv1 = torch.floor(self.LeakyReLU(self.Conv29(Output_Layer9_cv2_conv29))).int()
        
        # Splitting
        Split_Target = (128, 128)
        Output_Layer10_split_0, Output_Layer10_split_1 = torch.split(Output_Layer10_cv1, Split_Target, dim=1)
        
        # ******************************************* Attention *******************************************
        B, C, H, W = Output_Layer10_split_1.shape
        N = H * W
        
        # Quantization: 
        SF_Idx = 31; Prev_SF_Idx = 29
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=45
        Output_Layer10_split_1_conv31 = torch.clip(((((Output_Layer10_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer9_cv2_conv29, layer_idx=10, conv_idx=29)
            Save_Weight_HW(weight=self.Conv29.weight, layer_idx=10, conv_idx=29)
            Save_Bias_HW(bias=self.Conv29.bias, layer_idx=10, conv_idx=29)
            Save_Output_HW(activation=Output_Layer10_split_1_conv31, layer_idx=10, conv_idx=29)
        
        qkv = self.Conv31(Output_Layer10_split_1_conv31).int()
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        
        # Quantization: 
        SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=46
        q = torch.clip(((((q * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer10_split_1_conv31, layer_idx=10, conv_idx=31)
            Save_Weight_HW(weight=self.Conv31.weight, layer_idx=10, conv_idx=31)
            Save_Bias_HW(bias=self.Conv31.bias, layer_idx=10, conv_idx=31)
            Save_Output_HW(activation=q, layer_idx=10, conv_idx=31)
        
        # Quantization: 
        SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=47
        k = torch.clip(((((k * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        attn = (q.transpose(-2, -1) @ k)
        
        # Quantization: 
        SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] / Y_Scale[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=48
        attn = torch.clip(((((attn * H_Scale).int()>>(bit - 1 + 2)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_MatMul_Input_HW(activation=q[:, :1, :, :].reshape(1, 32, 20, 15), layer_idx=10, MatMul_idx=0, input_idx=0)
            Save_MatMul_Input_HW(activation=k[:, :1, :, :].reshape(1, 32, 20, 15), layer_idx=10, MatMul_idx=0, input_idx=1)
            Save_MatMul_Output_HW(activation=attn[:, :1, :, :].reshape(1,300,20,15), layer_idx=10, MatMul_idx=0, output_idx=0)
            Save_MatMul_Input_HW(activation=q[:, 1:, :, :].reshape(1, 32, 20, 15), layer_idx=10, MatMul_idx=0, input_idx=2)
            Save_MatMul_Input_HW(activation=k[:, 1:, :, :].reshape(1, 32, 20, 15), layer_idx=10, MatMul_idx=0, input_idx=3)
            Save_MatMul_Output_HW(activation=attn[:, 1:, :, :].reshape(1, 300, 20, 15), layer_idx=10, MatMul_idx=0, output_idx=1)

        # Apply LeakyReLU instead of softmax
        attn = torch.floor(self.LeakyReLU(attn))
        
        # Normalized Using Max Absolute Value for Visualization
        max_abs_val = attn.abs().max(dim=-1, keepdim=True)[0]
        max_abs_val = 2**torch.round(torch.log2(max_abs_val.clamp(min=1)))
        attn = round_half_up(attn / max_abs_val)

        # pe
        # Quantization: 
        SF_Idx = 33; Prev_SF_Idx = 31
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=49
        v_output_pe = torch.clip(((((v * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_pe = self.Conv33(v_output_pe.reshape(B, C, H, W)).int()
        
        # Quantization: 
        SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=50
        v = torch.clip(((((v * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Attention_Out = (v @ attn.transpose(-2, -1)).int()

        # Quantization: 
        SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] / Y_Scale[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=51
        Attention_Out = torch.clip(((((Attention_Out * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_MatMul_Input_HW(activation=v[:, :1, :, :].reshape(1, 64, 20, 15), layer_idx=10, MatMul_idx=1, input_idx=0)
            Save_MatMul_Input_HW(activation=attn.transpose(-2, -1)[:, :1, :, :].reshape(1, 300, 20, 15), layer_idx=10, MatMul_idx=1, input_idx=1)
            Save_MatMul_Output_HW(activation=Attention_Out[:, :1, :, :].reshape(1, 64, 20, 15), layer_idx=10, MatMul_idx=1, output_idx=0)
            Save_MatMul_Input_HW(activation=v[:, 1:, :, :].reshape(1, 64, 20, 15), layer_idx=10, MatMul_idx=1, input_idx=2)
            Save_MatMul_Input_HW(activation=attn.transpose(-2, -1)[:, 1:, :, :].reshape(1, 300, 20, 15), layer_idx=10, MatMul_idx=1, input_idx=3)
            Save_MatMul_Output_HW(activation=Attention_Out[:, 1:, :, :].reshape(1, 64, 20, 15), layer_idx=10, MatMul_idx=1, output_idx=1)

        # Quantization
        SF_Idx = 32; Prev_SF_Idx = 33
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=52
        Output_pe = torch.clip(((((Output_pe * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=v_output_pe.reshape(B, C, H, W), layer_idx=10, conv_idx=33)
            Save_Weight_HW(weight=self.Conv33.weight, layer_idx=10, conv_idx=33)
            Save_Bias_HW(bias=self.Conv33.bias, layer_idx=10, conv_idx=33)
            Save_Output_HW(activation=Output_pe, layer_idx=10, conv_idx=33)
        
        # EWAdder0
        Attention_Out = torch.clip(((Attention_Out.view(B, C, H, W) + Output_pe).int() + 1) >> 1, min=q_min, max=q_max) - 0.0
        
        # Save_Data: 
        if save_data: 
            Save_EWAdd_Input_HW(activation=Attention_Out.view(B, C, H, W), layer_idx=10, Add_idx=0, input_idx=0) 
            Save_EWAdd_Input_HW(activation=Output_pe, layer_idx=10, Add_idx=0, input_idx=1) 
            Save_EWAdd_Output_HW(activation=Attention_Out, layer_idx=10, Add_idx=0, output_idx=0) 
        
        # proj
        Attention_Out = self.Conv32(Attention_Out).int()
        
        # Element-wise Adder
        # Quantization: 
        SF_Idx = 34; Prev_SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=53
        Attention_Out_Add = torch.clip(((((Attention_Out * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Attention_Out, layer_idx=10, conv_idx=32)
            Save_Weight_HW(weight=self.Conv32.weight, layer_idx=10, conv_idx=32)
            Save_Bias_HW(bias=self.Conv32.bias, layer_idx=10, conv_idx=32)
            Save_Output_HW(activation=Attention_Out_Add, layer_idx=10, conv_idx=32)
        
        # Quantization: 
        SF_Idx = 34; Prev_SF_Idx = 29
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=54
        Output_Layer10_split_1_Add0 = torch.clip(((((Output_Layer10_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder1
        Output_Layer10_Add0 = torch.clip(torch.add(Output_Layer10_split_1_Add0, Attention_Out_Add), min=q_min, max=q_max)
        
        # Save_Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer10_split_1_Add0, layer_idx=10, Add_idx=1, input_idx=0)
            Save_EWAdd_Input_HW(activation=Attention_Out_Add, layer_idx=10, Add_idx=1, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer10_Add0, layer_idx=10, Add_idx=1, output_idx=0)

        # ************************************************* FFN *******************************************
        # ffn0
        Output_Layer10_ffn0 = torch.floor(self.LeakyReLU(self.Conv34(Output_Layer10_Add0))).int()
        
        # ffn1
        # Quantization: 
        SF_Idx = 35
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=55
        Output_Layer10_ffn0 = torch.clip(((((Output_Layer10_ffn0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer10_Add0, layer_idx=10, conv_idx=34)
            Save_Weight_HW(weight=self.Conv34.weight, layer_idx=10, conv_idx=34)
            Save_Bias_HW(bias=self.Conv34.bias, layer_idx=10, conv_idx=34)
            Save_Output_HW(activation=Output_Layer10_ffn0, layer_idx=10, conv_idx=34)
        
        Output_Layer10_ffn1 = self.Conv35(Output_Layer10_ffn0).int()
        
        # Additional: Element-wise Adder
        # Quantization: 
        SF_Idx = 30
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=56
        Output_Layer10_split_1_Addi0 = torch.clip(((((Output_Layer10_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 30; Prev_SF_Idx = 32
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=57
        Attention_Out_Addi0 = torch.clip(((((Attention_Out * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # EWAdder2
        Output_Layer10_Addi0 = torch.clip(torch.add(Output_Layer10_split_1_Addi0, Attention_Out_Addi0), min=q_min, max=q_max)
        
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer10_split_1_Addi0, layer_idx=10, Add_idx=2, input_idx=0)
            Save_EWAdd_Input_HW(activation=Attention_Out_Addi0, layer_idx=10, Add_idx=2, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer10_Addi0, layer_idx=10, Add_idx=2, output_idx=0)
        
        # Element-wise Adder
        # Quantization: 
        SF_Idx = 30; Prev_SF_Idx = 35
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=58
        Output_Layer10_ffn1_Add1 = torch.clip(((((Output_Layer10_ffn1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer10_ffn0, layer_idx=10, conv_idx=35)
            Save_Weight_HW(weight=self.Conv35.weight, layer_idx=10, conv_idx=35)
            Save_Bias_HW(bias=self.Conv35.bias, layer_idx=10, conv_idx=35)
            Save_Output_HW(activation=Output_Layer10_ffn1_Add1, layer_idx=10, conv_idx=35)
        
        # EWAdder3
        Output_Layer10_Add1 = torch.clip(torch.add(Output_Layer10_Addi0, Output_Layer10_ffn1_Add1), min=q_min, max=q_max)
        
        # Save Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer10_Addi0, layer_idx=10, Add_idx=3, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer10_ffn1_Add1, layer_idx=10, Add_idx=3, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer10_Add1, layer_idx=10, Add_idx=3, output_idx=0)
        
        # Concatenation
        # Quantization: 
        SF_Idx = 30
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=59
        Output_Layer10_split_0_Cat = torch.clip(((((Output_Layer10_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer10_Concat = torch.cat((Output_Layer10_split_0_Cat, Output_Layer10_Add1), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer10_split_0_Cat, layer_idx=10, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer10_Add1, layer_idx=10, Concat_Idx=0, input_idx=1)
            Save_Concat_Output_HW(activation=Output_Layer10_Concat, layer_idx=10, Concat_Idx=0, output_idx=0)
        
        # cv2: 
        Output_Layer10_cv2 = torch.floor(self.LeakyReLU(self.Conv30(Output_Layer10_Concat))).int() # --> Upsample & L21
        
        # -------------------------------------- Layer11-----------------------------------------------
        # Quantization: 
        SF_Idx = 36; Prev_SF_Idx = 30
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=60
        Output_Layer10_cv2_Upsample = torch.clip(((((Output_Layer10_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer10_Concat, layer_idx=10, conv_idx=30)
            Save_Weight_HW(weight=self.Conv30.weight, layer_idx=10, conv_idx=30)
            Save_Bias_HW(bias=self.Conv30.bias, layer_idx=10, conv_idx=30)
            Save_Output_HW(activation=Output_Layer10_cv2_Upsample, layer_idx=10, conv_idx=30)
        
        Output_Layer11 = self.Upsample(Output_Layer10_cv2_Upsample)
        
        # Save Data
        if save_data: 
            Save_Upsample_Input_HW(activation=Output_Layer10_cv2_Upsample, layer_idx=11)
            Save_Upsample_Output_HW(activation=Output_Layer11, layer_idx=11)
        
        # -------------------------------------- Layer12-----------------------------------------------
        # Quantization: 
        SF_Idx = 36; Prev_SF_Idx = 16
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=61
        Output_Layer6_cv2_Cat = torch.clip(((((Output_Layer6_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer12 = torch.cat((Output_Layer11, Output_Layer6_cv2_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer11, layer_idx=12, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer6_cv2_Cat, layer_idx=12, Concat_Idx=0, input_idx=1)
            Save_Concat_Output_HW(activation=Output_Layer12, layer_idx=12, Concat_Idx=0, output_idx=0)
        
        # ----------------------------------- Layer13: C2f ----------------------------------------
        # cv1:
        Output_Layer13_cv1 = torch.floor(self.LeakyReLU(self.Conv36(Output_Layer12))).int()
        
        # Splitting
        Split_Target = (64, 64)
        Output_Layer13_split_0, Output_Layer13_split_1 = torch.split(Output_Layer13_cv1, Split_Target, dim=1)

        # BottleNeck: cv1
        # Quantization: 
        SF_Idx = 38; Prev_SF_Idx = 36
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=62
        Output_Layer13_split_1_conv38 = torch.clip(((((Output_Layer13_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer12, layer_idx=13, conv_idx=36)
            Save_Weight_HW(weight=self.Conv36.weight, layer_idx=13, conv_idx=36)
            Save_Bias_HW(bias=self.Conv36.bias, layer_idx=13, conv_idx=36)
            Save_Output_HW(activation=Output_Layer13_split_1_conv38, layer_idx=13, conv_idx=36)
        
        Output_Layer13_Bott0 = torch.floor(self.LeakyReLU(self.Conv38(Output_Layer13_split_1_conv38))).int()
        
        # BottleNeck: cv2
        # Quantization: 
        SF_Idx = 39
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=63
        Output_Layer13_Bott0 = torch.clip(((((Output_Layer13_Bott0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer13_split_1_conv38, layer_idx=13, conv_idx=38)
            Save_Weight_HW(weight=self.Conv38.weight, layer_idx=13, conv_idx=38)
            Save_Bias_HW(bias=self.Conv38.bias, layer_idx=13, conv_idx=38)
            Save_Output_HW(activation=Output_Layer13_Bott0, layer_idx=13, conv_idx=38)
        
        Output_Layer13_Bott1 = torch.floor(self.LeakyReLU(self.Conv39(Output_Layer13_Bott0))).int()
        
        # Concentination
        # Quantization: 
        SF_Idx = 37
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=64
        Output_Layer13_split_0_Cat = torch.clip(((((Output_Layer13_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 37
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=65
        Output_Layer13_split_1_Cat = torch.clip(((((Output_Layer13_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 37; Prev_SF_Idx = 39
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=66
        Output_Layer13_Bott1_Cat = torch.clip(((((Output_Layer13_Bott1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer13_Bott0, layer_idx=13, conv_idx=39)
            Save_Weight_HW(weight=self.Conv39.weight, layer_idx=13, conv_idx=39)
            Save_Bias_HW(bias=self.Conv39.bias, layer_idx=13, conv_idx=39)
            Save_Output_HW(activation=Output_Layer13_Bott1_Cat, layer_idx=13, conv_idx=39)
        
        Output_Layer13_Concat = torch.cat((Output_Layer13_split_0_Cat, Output_Layer13_split_1_Cat, Output_Layer13_Bott1_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer13_split_0_Cat, layer_idx=13, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer13_split_1_Cat, layer_idx=13, Concat_Idx=0, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer13_Bott1_Cat, layer_idx=13, Concat_Idx=0, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer13_Concat, layer_idx=13, Concat_Idx=0, output_idx=0)
        
        # cv2
        Output_Layer13_cv2 = torch.floor(self.LeakyReLU(self.Conv37(Output_Layer13_Concat))).int() # --> Upsample & L18
        
        # -------------------------------------- Layer14-----------------------------------------------
        # Quantization: 
        SF_Idx = 40; Prev_SF_Idx = 37
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=67
        Output_Layer13_cv2_Upsample = torch.clip(((((Output_Layer13_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer13_Concat, layer_idx=13, conv_idx=37)
            Save_Weight_HW(weight=self.Conv37.weight, layer_idx=13, conv_idx=37)
            Save_Bias_HW(bias=self.Conv37.bias, layer_idx=13, conv_idx=37)
            Save_Output_HW(activation=Output_Layer13_cv2_Upsample, layer_idx=13, conv_idx=37)
        
        Output_Layer14 = self.Upsample(Output_Layer13_cv2_Upsample)
        
        # Save Data
        if save_data: 
            Save_Upsample_Input_HW(activation=Output_Layer13_cv2_Upsample, layer_idx=14)
            Save_Upsample_Output_HW(activation=Output_Layer14, layer_idx=14)
        
        # -------------------------------------- Layer15-----------------------------------------------
        # Quantization: 
        SF_Idx = 40; Prev_SF_Idx = 8
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=68
        Output_Layer4_cv2_Cat = torch.clip(((((Output_Layer4_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer15 = torch.cat((Output_Layer14, Output_Layer4_cv2_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer14, layer_idx=15, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer4_cv2_Cat, layer_idx=15, Concat_Idx=0, input_idx=1)
            Save_Concat_Output_HW(activation=Output_Layer15, layer_idx=15, Concat_Idx=0, output_idx=0)
        
        # ----------------------------------- Layer16: C2f ----------------------------------------
        # cv1:
        Output_Layer16_cv1 = torch.floor(self.LeakyReLU(self.Conv40(Output_Layer15))).int()
        
        # Splitting
        Split_Target = (32, 32)
        Output_Layer16_split_0, Output_Layer16_split_1 = torch.split(Output_Layer16_cv1, Split_Target, dim=1)
        
        # BottleNeck: cv1
        # Quantization: 
        SF_Idx = 42; Prev_SF_Idx = 40
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=69
        Output_Layer16_split_1_conv42 = torch.clip(((((Output_Layer16_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer15, layer_idx=16, conv_idx=40)
            Save_Weight_HW(weight=self.Conv40.weight, layer_idx=16, conv_idx=40)
            Save_Bias_HW(bias=self.Conv40.bias, layer_idx=16, conv_idx=40)
            Save_Output_HW(activation=Output_Layer16_split_1_conv42, layer_idx=16, conv_idx=40)
        
        Output_Layer16_Bott0 = torch.floor(self.LeakyReLU(self.Conv42(Output_Layer16_split_1_conv42))).int()
        
        # BottleNeck: cv2
        # Quantization: 
        SF_Idx = 43
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=70
        Output_Layer16_Bott0 = torch.clip(((((Output_Layer16_Bott0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer16_split_1_conv42, layer_idx=16, conv_idx=42)
            Save_Weight_HW(weight=self.Conv42.weight, layer_idx=16, conv_idx=42)
            Save_Bias_HW(bias=self.Conv42.bias, layer_idx=16, conv_idx=42)
            Save_Output_HW(activation=Output_Layer16_Bott0, layer_idx=16, conv_idx=42)
        
        Output_Layer16_Bott1 = torch.floor(self.LeakyReLU(self.Conv43(Output_Layer16_Bott0))).int()
        
        # Concentination
        # Quantization: 
        SF_Idx = 41
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=71
        Output_Layer16_split_0_Cat = torch.clip(((((Output_Layer16_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 41
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=72
        Output_Layer16_split_1_Cat = torch.clip(((((Output_Layer16_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 41; Prev_SF_Idx = 43
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=73
        Output_Layer16_Bott1_Cat = torch.clip(((((Output_Layer16_Bott1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer16_Bott0, layer_idx=16, conv_idx=43)
            Save_Weight_HW(weight=self.Conv43.weight, layer_idx=16, conv_idx=43)
            Save_Bias_HW(bias=self.Conv43.bias, layer_idx=16, conv_idx=43)
            Save_Output_HW(activation=Output_Layer16_Bott1_Cat, layer_idx=16, conv_idx=43)
        
        Output_Layer16_Concat = torch.cat((Output_Layer16_split_0_Cat, Output_Layer16_split_1_Cat, Output_Layer16_Bott1_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer16_split_0_Cat, layer_idx=16, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer16_split_1_Cat, layer_idx=16, Concat_Idx=0, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer16_Bott1_Cat, layer_idx=16, Concat_Idx=0, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer16_Concat, layer_idx=16, Concat_Idx=0, output_idx=0)
        
        # cv2
        Output_Layer16_cv2 = torch.floor(self.LeakyReLU(self.Conv41(Output_Layer16_Concat))).int() # L17 & Head1
        
        # -------------------------------------- Layer17-----------------------------------------------
        # Quantization: 
        SF_Idx = 44; Prev_SF_Idx = 41
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=74
        Output_Layer16_cv2_conv44 = torch.clip(((((Output_Layer16_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer16_Concat, layer_idx=16, conv_idx=41)
            Save_Weight_HW(weight=self.Conv41.weight, layer_idx=16, conv_idx=41)
            Save_Bias_HW(bias=self.Conv41.bias, layer_idx=16, conv_idx=41)
            Save_Output_HW(activation=Output_Layer16_cv2_conv44, layer_idx=16, conv_idx=41)
        
        Output_Layer17 = torch.floor(self.LeakyReLU(self.Conv44(Output_Layer16_cv2_conv44))).int()
        
        # -------------------------------------- Layer18-----------------------------------------------
        # Quantization: 
        SF_Idx = 45
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=75
        Output_Layer17_Cat = torch.clip(((((Output_Layer17 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer16_cv2_conv44, layer_idx=17, conv_idx=44)
            Save_Weight_HW(weight=self.Conv44.weight, layer_idx=17, conv_idx=44)
            Save_Bias_HW(bias=self.Conv44.bias, layer_idx=17, conv_idx=44)
            Save_Output_HW(activation=Output_Layer17_Cat, layer_idx=17, conv_idx=44)
        
        # Quantization: 
        SF_Idx = 45; Prev_SF_Idx = 37
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=76
        Output_Layer13_cv2_Cat = torch.clip(((((Output_Layer13_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer18 = torch.cat((Output_Layer17_Cat, Output_Layer13_cv2_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer17_Cat, layer_idx=18, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer13_cv2_Cat, layer_idx=18, Concat_Idx=0, input_idx=1)
            Save_Concat_Output_HW(activation=Output_Layer18, layer_idx=18, Concat_Idx=0, output_idx=0)
        
        # ----------------------------------- Layer19: C2f ----------------------------------------
        # cv1:
        Output_Layer19_cv1 = torch.floor(self.LeakyReLU(self.Conv45(Output_Layer18))).int()
        
        # Splitting
        Split_Target = (64, 64)
        Output_Layer19_split_0, Output_Layer19_split_1 = torch.split(Output_Layer19_cv1, Split_Target, dim=1)
        
        # BottleNeck: cv1
        # Quantization: 
        SF_Idx = 47; Prev_SF_Idx = 45
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=77
        Output_Layer19_split_1_conv47 = torch.clip(((((Output_Layer19_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer18, layer_idx=19, conv_idx=45)
            Save_Weight_HW(weight=self.Conv45.weight, layer_idx=19, conv_idx=45)
            Save_Bias_HW(bias=self.Conv45.bias, layer_idx=19, conv_idx=45)
            Save_Output_HW(activation=Output_Layer19_split_1_conv47, layer_idx=19, conv_idx=45)
        
        Output_Layer19_Bott0 = torch.floor(self.LeakyReLU(self.Conv47(Output_Layer19_split_1_conv47))).int()
        
        # BottleNeck: cv2
        # Quantization: 
        SF_Idx = 48
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=78
        Output_Layer19_Bott0 = torch.clip(((((Output_Layer19_Bott0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer19_split_1_conv47, layer_idx=19, conv_idx=47)
            Save_Weight_HW(weight=self.Conv47.weight, layer_idx=19, conv_idx=47)
            Save_Bias_HW(bias=self.Conv47.bias, layer_idx=19, conv_idx=47)
            Save_Output_HW(activation=Output_Layer19_Bott0, layer_idx=19, conv_idx=47)
        
        Output_Layer19_Bott1 = torch.floor(self.LeakyReLU(self.Conv48(Output_Layer19_Bott0))).int()
        
        # Concentination
        # Quantization: 
        SF_Idx = 46
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=79
        Output_Layer19_split_0_Cat = torch.clip(((((Output_Layer19_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 46
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=80
        Output_Layer19_split_1_Cat = torch.clip(((((Output_Layer19_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 46; Prev_SF_Idx = 48
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=81
        Output_Layer19_Bott1_Cat = torch.clip(((((Output_Layer19_Bott1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer19_Bott0, layer_idx=19, conv_idx=48)
            Save_Weight_HW(weight=self.Conv48.weight, layer_idx=19, conv_idx=48)
            Save_Bias_HW(bias=self.Conv48.bias, layer_idx=19, conv_idx=48)
            Save_Output_HW(activation=Output_Layer19_Bott1_Cat, layer_idx=19, conv_idx=48)
        
        Output_Layer19_Concat = torch.cat((Output_Layer19_split_0_Cat, Output_Layer19_split_1_Cat, Output_Layer19_Bott1_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer19_split_0_Cat, layer_idx=19, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer19_split_1_Cat, layer_idx=19, Concat_Idx=0, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer19_Bott1_Cat, layer_idx=19, Concat_Idx=0, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer19_Concat, layer_idx=19, Concat_Idx=0, output_idx=0)
        
        # cv2
        Output_Layer19_cv2 = torch.floor(self.LeakyReLU(self.Conv46(Output_Layer19_Concat))).int() # --> L20 & Head2
        
        # -------------------------------------- Layer20: SCDown -------------------------------------------
        ###################################################################################################
        #                         Spatial Convolution with Downsampling                                   #
        ###################################################################################################
        # cv1: 
        # Quantization: 
        SF_Idx = 49; Prev_SF_Idx = 46
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=82
        Output_Layer19_cv2_conv49 = torch.clip(((((Output_Layer19_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer19_Concat, layer_idx=19, conv_idx=46)
            Save_Weight_HW(weight=self.Conv46.weight, layer_idx=19, conv_idx=46)
            Save_Bias_HW(bias=self.Conv46.bias, layer_idx=19, conv_idx=46)
            Save_Output_HW(activation=Output_Layer19_cv2_conv49, layer_idx=19, conv_idx=46)
        
        Output_Layer20_cv1 = torch.floor(self.LeakyReLU(self.Conv49(Output_Layer19_cv2_conv49))).int()
        
        # cv2: 
        # Quantization: 
        SF_Idx = 50
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=83
        Output_Layer20_cv1 = torch.clip(((((Output_Layer20_cv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer19_cv2_conv49, layer_idx=20, conv_idx=49)
            Save_Weight_HW(weight=self.Conv49.weight, layer_idx=20, conv_idx=49)
            Save_Bias_HW(bias=self.Conv49.bias, layer_idx=20, conv_idx=49)
            Save_Output_HW(activation=Output_Layer20_cv1, layer_idx=20, conv_idx=49)
        
        Output_Layer20_cv2 = self.Conv50(Output_Layer20_cv1).int()
        
        # ------------------------------------------- Layer21 -----------------------------------------------
        # Quantization: 
        SF_Idx = 51
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=84
        Output_Layer20_cv2_Cat = torch.clip(((((Output_Layer20_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer20_cv1, layer_idx=20, conv_idx=50)
            Save_Weight_HW(weight=self.Conv50.weight, layer_idx=20, conv_idx=50)
            Save_Bias_HW(bias=self.Conv50.bias, layer_idx=20, conv_idx=50)
            Save_Output_HW(activation=Output_Layer20_cv2_Cat, layer_idx=20, conv_idx=50)
        
        # Quantization: 
        SF_Idx = 51; Prev_SF_Idx = 30
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=85
        Output_Layer10_cv2_Cat = torch.clip(((((Output_Layer10_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer21 = torch.cat((Output_Layer20_cv2_Cat, Output_Layer10_cv2_Cat), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer20_cv2_Cat, layer_idx=21, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer10_cv2_Cat, layer_idx=21, Concat_Idx=0, input_idx=1)
            Save_Concat_Output_HW(activation=Output_Layer21, layer_idx=21, Concat_Idx=0, output_idx=0)
        
        # ---------------------------------------- Layer22: C2fCIB ------------------------------------------
        ###################################################################################################
        #                            C2f:Conv2d followed by Convolutional Layer                           #
        ###################################################################################################
        # cv1:
        Output_Layer22_cv1 = torch.floor(self.LeakyReLU(self.Conv51(Output_Layer21))).int()
        
        # Splitting
        Split_Target = (128, 128)
        Output_Layer22_split_0, Output_Layer22_split_1 = torch.split(Output_Layer22_cv1, Split_Target, dim=1)
        
        ###################################################################################################
        #                             CIB: Convolutional Inverted Bottleneck                              #
        ###################################################################################################
        # conv0
        # Quantization: 
        SF_Idx = 53; Prev_SF_Idx = 51
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=86
        Output_Layer22_split_1_conv53 = torch.clip(((((Output_Layer22_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer21, layer_idx=22, conv_idx=51)
            Save_Weight_HW(weight=self.Conv51.weight, layer_idx=22, conv_idx=51)
            Save_Bias_HW(bias=self.Conv51.bias, layer_idx=22, conv_idx=51)
            Save_Output_HW(activation=Output_Layer22_split_1_conv53, layer_idx=22, conv_idx=51)
        
        Output_Layer22_conv0 = torch.floor(self.LeakyReLU(self.Conv53(Output_Layer22_split_1_conv53))).int()
        
        # conv1
        # Quantization: 
        SF_Idx = 54
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=87
        Output_Layer22_conv0 = torch.clip(((((Output_Layer22_conv0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_split_1_conv53, layer_idx=22, conv_idx=53)
            Save_Weight_HW(weight=self.Conv53.weight, layer_idx=22, conv_idx=53)
            Save_Bias_HW(bias=self.Conv53.bias, layer_idx=22, conv_idx=53)
            Save_Output_HW(activation=Output_Layer22_conv0, layer_idx=22, conv_idx=53)
        
        Output_Layer22_conv1 = torch.floor(self.LeakyReLU(self.Conv54(Output_Layer22_conv0))).int()
        
        # RepVGGDW
        # Quantization: 
        SF_Idx = 55
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=88
        Output_Layer22_conv1 = torch.clip(((((Output_Layer22_conv1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_conv0, layer_idx=22, conv_idx=54)
            Save_Weight_HW(weight=self.Conv54.weight, layer_idx=22, conv_idx=54)
            Save_Bias_HW(bias=self.Conv54.bias, layer_idx=22, conv_idx=54)
            Save_Output_HW(activation=Output_Layer22_conv1, layer_idx=22, conv_idx=54)
        
        Output_Layer22_RepVGGDW = torch.floor(self.LeakyReLU(self.Conv55(Output_Layer22_conv1))).int()
        
        # Conv3
        # Quantization: 
        SF_Idx = 56
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=89
        Output_Layer22_RepVGGDW = torch.clip(((((Output_Layer22_RepVGGDW * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_conv1, layer_idx=22, conv_idx=55)
            Save_Weight_HW(weight=self.Conv55.weight, layer_idx=22, conv_idx=55)
            Save_Bias_HW(bias=self.Conv55.bias, layer_idx=22, conv_idx=55)
            Save_Output_HW(activation=Output_Layer22_RepVGGDW, layer_idx=22, conv_idx=55)
        
        Output_Layer22_conv3 = torch.floor(self.LeakyReLU(self.Conv56(Output_Layer22_RepVGGDW))).int()
        
        # Conv4
        # Quantization: 
        SF_Idx = 57
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=90
        Output_Layer22_conv3 = torch.clip(((((Output_Layer22_conv3 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_RepVGGDW, layer_idx=22, conv_idx=56)
            Save_Weight_HW(weight=self.Conv56.weight, layer_idx=22, conv_idx=56)
            Save_Bias_HW(bias=self.Conv56.bias, layer_idx=22, conv_idx=56)
            Save_Output_HW(activation=Output_Layer22_conv3, layer_idx=22, conv_idx=56)
        
        Output_Layer22_conv4 = torch.floor(self.LeakyReLU(self.Conv57(Output_Layer22_conv3))).int()
        
        # Element-Wise Adder
        # Quantization: 
        SF_Idx = 52
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=91
        Output_Layer22_split_1_Add = torch.clip(((((Output_Layer22_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 52; Prev_SF_Idx = 57
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=92
        Output_Layer22_conv4_Add = torch.clip(((((Output_Layer22_conv4 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_conv3, layer_idx=22, conv_idx=57)
            Save_Weight_HW(weight=self.Conv57.weight, layer_idx=22, conv_idx=57)
            Save_Bias_HW(bias=self.Conv57.bias, layer_idx=22, conv_idx=57)
            Save_Output_HW(activation=Output_Layer22_conv4_Add, layer_idx=22, conv_idx=57)
        
        # EWAdder0
        Output_Layer22_Add = torch.clip(torch.add(Output_Layer22_split_1_Add, Output_Layer22_conv4_Add), min=q_min, max=q_max)
        
        # Save Data
        if save_data: 
            Save_EWAdd_Input_HW(activation=Output_Layer22_split_1_Add, layer_idx=22, Add_idx=0, input_idx=0)
            Save_EWAdd_Input_HW(activation=Output_Layer22_conv4_Add, layer_idx=22, Add_idx=0, input_idx=1)
            Save_EWAdd_Output_HW(activation=Output_Layer22_Add, layer_idx=22, Add_idx=0, output_idx=0)
        
        # Concentation
        # Quantization: 
        SF_Idx = 52
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=93
        Output_Layer22_split_0_Cat = torch.clip(((((Output_Layer22_split_0 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Quantization: 
        SF_Idx = 52
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[SF_Idx-1])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=94
        Output_Layer22_split_1_Cat = torch.clip(((((Output_Layer22_split_1 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        Output_Layer22_Concat = torch.cat((Output_Layer22_split_0_Cat, Output_Layer22_split_1_Cat, Output_Layer22_Add), dim=1)
        
        # Save Data
        if save_data: 
            Save_Concat_Input_HW(activation=Output_Layer22_split_0_Cat, layer_idx=22, Concat_Idx=0, input_idx=0)
            Save_Concat_Input_HW(activation=Output_Layer22_split_1_Cat, layer_idx=22, Concat_Idx=0, input_idx=1)
            Save_Concat_Input_HW(activation=Output_Layer22_Add, layer_idx=22, Concat_Idx=0, input_idx=2)
            Save_Concat_Output_HW(activation=Output_Layer22_Concat, layer_idx=22, Concat_Idx=0, output_idx=0)
        
        # cv2: 
        Output_Layer22_cv2 = torch.floor(self.LeakyReLU(self.Conv52(Output_Layer22_Concat))).int() # --> Head3
        
        # Quantization
        SF_Idx = 58; Prev_SF_Idx = 52
        H_Scale = np.round(Y_Scale[SF_Idx] * Scale_fmap[Prev_SF_Idx])
        H_Scale = np.int16((1 << bit) // H_Scale)
        Hardware_Scale.append(H_Scale) # idx=95
        Output_Layer22_cv2 = torch.clip(((((Output_Layer22_cv2 * H_Scale).int()>>(bit - 1)) + 1) >> 1), min=q_min, max=q_max) - 0.0
        
        # Save_Data
        if save_data: 
            Save_InFmap_HW(activation=Output_Layer22_Concat, layer_idx=22, conv_idx=52)
            Save_Weight_HW(weight=self.Conv52.weight, layer_idx=22, conv_idx=52)
            Save_Bias_HW(bias=self.Conv52.bias, layer_idx=22, conv_idx=52)
            Save_Output_HW(activation=Output_Layer22_cv2, layer_idx=22, conv_idx=52)
        
        ###################################################################################################
        #                                                                                                 #
        #                                 YOLOv10n-Detection (Detector)                                   #
        #                                                                                                 #
        ###################################################################################################
        # Dequantization: Head1
        SF_Idx = 44
        Output_Layer16_cv2 = Output_Layer16_cv2_conv44 * Y_Scale[SF_Idx]

        # Dequantization: Head2
        SF_Idx = 49
        Output_Layer19_cv2 = Output_Layer19_cv2_conv49 * Y_Scale[SF_Idx]

        # Dequantization: Head3
        SF_Idx = 58
        Output_Layer22_cv2 = Output_Layer22_cv2 * Y_Scale[SF_Idx]

        # Head for Detector: [Head1, Head2, Head3] = [Out_L16, Out_L19, Out_L22]
        Head = [Output_Layer16_cv2, Output_Layer19_cv2, Output_Layer22_cv2]
        
        Prediction = self.detector(Head)
        
        return Output_Layer16_cv2, Output_Layer19_cv2, Output_Layer22_cv2, Prediction, Hardware_Scale
    

def write_results(i, p, im, s, Results, save_path):
    """Write inference results to a file or directory."""
    string = ""  # print string
    string += "%gx%g " % im.shape[2:]
    result = Results[i]
    result.save_dir = save_path
    string += result.verbose() + f"{result.speed['inference']:.1f}ms"

    # Add predictions to image
    plotted_img = result.plot(
        line_width=None,
        boxes=True,
        conf=True,
        labels=True,
        im_gpu=im[i])

    # Save results
    save_predicted_images(plotted_img, save_path, frame=0)

    return string


def save_predicted_images(plotted_img, save_path="", frame=0):
    im = plotted_img # np array with 8-bits
    
    # Save images
    cv2.imwrite(save_path, im)


def Visualization(args, Prediction, image, im0s):
    # Post-Processing
    Results = Postprocessing(Prediction, image, im0s, args.image_path)
    
    # Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualization   
    seen, windows, batch = 0, [], None
    verbose = True; save = True; save_txt = False; show = False
    save_crop = False
    profilers = (
        ops.Profile(device=image.device),
        ops.Profile(device=image.device),
        ops.Profile(device=image.device),
    )

    with profilers[0]: 0
    with profilers[1]: 0
    with profilers[2]: 0

    n = len(im0s)
    for i in range(n):
        seen += 1
        Results[i].speed = {
            "preprocess": profilers[0].dt * 1e3 / n,
            "inference": profilers[1].dt * 1e3 / n,
            "postprocess": profilers[2].dt * 1e3 / n,
        }
        
    s = [f'image 1/1 {args.image_path}: ']
    paths = [args.image_path]
    s[0] += write_results(i, Path(paths[i]), image, s, Results, args.save_path)
    LOGGER.info("\n".join(s))


def Inference_YOLOv10n(args, model): 
    # Weight and Fmap Scale Factors 
    with open(f'Parameters_zone/YOLOv10n/Scaling_factors/Weight_Scale_Factors.txt' ,'r') as f:
        qw = f.read()
        qw_d=[float(i) for i in qw.split()]

    qw_d=np.reshape(qw_d,(int(len(qw_d)/2),2))
    qw  = qw_d[:,0]  # Weight Scale Factor                                                  
    Zpw = qw_d[:,1]

    def Read_Y_scale_ZP():                                                                                  #
        Path = f'Parameters_zone/YOLOv10n/Scaling_factors/'                                                    #
        with open(Path + 'Fmap_Y_Scale.txt'    ,'r') as f: Y_sc = f.read(); Y_sc = [float(i) for i in Y_sc.split()] #
        with open(Path + 'Fmap_Y_ZeroPoint.txt','r') as f: ZP   = f.read(); ZP   = [float(i) for i in ZP  .split()] #
        return  Y_sc, ZP                                                                                       

    Y_Scale, ZP = Read_Y_scale_ZP() #
    Scale_fmap = 1/(qw*Y_Scale[:58]) ## Scale_Mul = 1/(Fmap_Scale_Factor * W_Scale_Factor)

    # Loading Quantized Parameters
    # Loading Pretrained YOLOv5n Pretrained Parameters
    WeightLoader_SSBQ(args, model, Zpw, Scale_fmap)

    # Preprocessing Image
    im0s = [cv2.imread(args.image_path)]
    image = Preprocessing(im0s)

    # Model Processing 
    model.eval()

    # Prediction Result
    _, _, _, Prediction, H_scale = model(image.cuda(), qw, Zpw, Y_Scale, ZP, Scale_fmap)
    
    # Visualization
    Visualization(args, Prediction, image, im0s)
    
    # Weight Size and Bias Size
    Weight_Size, Bias_Size = Print_Weight_Array()
    
    return Weight_Size, Bias_Size, H_scale