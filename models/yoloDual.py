#Construct a yolo-like model which is dual input for multiple image inputs.
import sys
sys.path.append("/home/jasper/git/yolov5")

import torch.nn as nn
import torch 
from models.common import *
from utils.autoanchor import check_anchor_order
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

LOGGER = logging.getLogger(__name__)

#Imports from yolo OG, need to be culled
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

#From Yolo OG
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, ch=3, classes=1, anchors=None, siamese=False):
        super(Model, self).__init__()
        self.siamese = siamese
        self.ch = ch
        self.classes = classes
        LOGGER.info(f'Now building dual input Yolo Model')

        if(anchors is None):
            print("Using default anchors defined in yoloDual.py")
            anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchors = anchors

        self.B1S1 = nn.Sequential(
            Focus(3,64), #Ch in, Ch out
            Conv(64, 128, 3, 2), #Ch in, Ch out, kernel, stride
            C3(128, 128, 3), #Ch in, ch out, number
            Conv(128, 256, 3, 2), #Ch in, Ch out, kernel, stride
            C3(256, 256, 9) #Ch in, ch out, number
        )

        self.B1S2 = nn.Sequential(    
            Conv(256, 512, 3, 2), #Ch in, Ch out, kernel, stride
            C3(512, 512, 9) #Ch in, ch out, number
        )

        self.B1S3 = nn.Sequential(   
            Conv(512, 1024, 3, 2), #Ch in, Ch out, kernel, stride
            SPP(1024, 1024, [5, 9, 13]), #Ch in, Ch out, 3-tuple
            C3(1024, 1024, 3) #Ch in, ch out, number
        )

        self.B2S1 = nn.Sequential(
            Focus(3,64), #Ch in, Ch out
            Conv(64, 128, 3, 2), #Ch in, Ch out, kernel, stride
            C3(128, 128, 3), #Ch in, ch out, number
            Conv(128, 256, 3, 2), #Ch in, Ch out, kernel, stride
            C3(256, 256, 9) #Ch in, ch out, number
        )

        self.B2S2 = nn.Sequential(    
            Conv(256, 512, 3, 2), #Ch in, Ch out, kernel, stride
            C3(512, 512, 9) #Ch in, ch out, number
        )

        self.B2S3 = nn.Sequential(   
            Conv(512, 1024, 3, 2), #Ch in, Ch out, kernel, stride
            SPP(1024, 1024, [5, 9, 13]), #Ch in, Ch out, 3-tuple
            C3(1024, 1024, 3) #Ch in, ch out, number
        )

        self.H1 = nn.Sequential(
            Conv(2048, 512, 1, 1) #Ch in, Ch out, kernel, stride
        )
        self.H2 = nn.Sequential(
            nn.Upsample(None, 2, 'nearest') #Size, scale, method
        )
        #Cat backbone P4 goes here
        self.H3 = nn.Sequential(
            C3(1536, 512, 3), #Ch in, ch out, number
            Conv(512, 256, 1, 1) #Ch in, Ch out, kernel, stride
        )
        self.H4 = nn.Sequential(
            nn.Upsample(None, 2, 'nearest') #Size, scale, method
        )
        #Cat backbone P3 goes here
        self.H5 = nn.Sequential(
            C3(768, 256, 3) #Ch in, ch out, number
        )
        self.H6 = nn.Sequential(
            Conv(256, 256, 3, 2) #Ch in, Ch out, kernel, stride
        )
        #Cat head P4 goes here
        self.H7 = nn.Sequential(
            C3(512, 512, 3) #Ch in, ch out, number
        )
        self.H8 = nn.Sequential(
            Conv(512, 512, 3, 2) #Ch in, Ch out, kernel, stride
        )
        #Cat head P5 goes here
        self.H9 = nn.Sequential(
            C3(1024, 1024, 3) #Ch in, ch out, number
        )
        
        self.Detect = nn.Sequential(
            Detect(classes, anchors, [256,512,1024]) #classes, anchors, ch, inplace
        )

        # Build strides, anchors
        m = self.Detect  # Detect()
        m = m[0]
        s = 256  # 2x min stride
        m.inplace = True
        ch = self.ch
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s),torch.zeros(1, ch, s, s))])  # forward
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride
        self._initialize_biases()  # only run once
        
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.Detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        raise NotImplementedError

    def autoshape(self):  # add AutoShape module
        raise NotImplementedError

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def forward(self, img1, img2, augment=False):
        if(augment):
            raise NotImplementedError

        if(self.siamese):
            # LOGGER.info(f'Now linking siamese dual Yolo Model')
            x  = self.forward_siamese(img1,img2)
        else:
            # LOGGER.info(f'Now linking non-siamese dual Yolo Model')
            x = self.forward_split(img1,img2)
        return x
    
    def forward_siamese(self, img1, img2):
        #Backbones
        b1 = self.B1S1(img1)
        b3 = self.B1S2(b1)
        b5 = self.B1S3(b3)

        b2 = self.B1S1(img2)
        b4 = self.B1S2(b2)
        b6 = self.B1S3(b4)

        #Combine spatial pyramid outputs
        p7 = torch.cat((b5, b6), dim=1) #9
        p8 = torch.cat((b3, b4), dim=1) #6
        p9 = torch.cat((b1, b2), dim=1) #4

        #Head
        h1 = self.H1(p7)
        h2 = self.H2(h1)
        h3 = torch.cat((h2,p8), dim=1) #Combine w 6
        h4 = self.H3(h3)
        h5 = self.H4(h4)
        h6 = torch.cat((h5,p9), dim=1) #Combine w 4
        h7 = self.H5(h6)
        h8 = self.H6(h7)
        h9 = torch.cat((h8,h4), dim=1) #Combine w 14
        h10 = self.H7(h9)
        h11 = self.H8(h10)
        h12 = torch.cat((h11,h1), dim=1) #Combine w 10
        h13 = self.H9(h12)
        h14 = self.Detect([h7,h10,h13])

        x = h14

        return x

    def forward_split(self, img1, img2):
        #Backbones
        b1 = self.B1S1(img1)
        b2 = self.B2S1(img2)

        b3 = self.B1S2(b1)
        b4 = self.B2S2(b2)

        b5 = self.B1S3(b3)
        b6 = self.B2S3(b4)

        #Combine spatial pyramid outputs
        p7 = torch.cat((b5, b6), dim=1) #9
        p8 = torch.cat((b3, b4), dim=1) #6
        p9 = torch.cat((b1, b2), dim=1) #4

        #Head
        h1 = self.H1(p7)
        h2 = self.H2(h1)
        h3 = torch.cat((h2,p8), dim=1) #Combine w 6
        h4 = self.H3(h3)
        h5 = self.H4(h4)
        h6 = torch.cat((h5,p9), dim=1) #Combine w 4
        h7 = self.H5(h6)
        h8 = self.H6(h7)
        h9 = torch.cat((h8,h4), dim=1) #Combine w 14
        h10 = self.H7(h9)
        h11 = self.H8(h10)
        h12 = torch.cat((h11,h1), dim=1) #Combine w 10
        h13 = self.H9(h12)
        h14 = self.Detect([h7,h10,h13])

        x = h14

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.Detect[-1]
        # m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

if __name__ == '__main__':
    classes = 1
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    model = Model(classes, anchors, False)
    batch_size = 1
    x1 = torch.randn(batch_size, 3, 256, 256)
    output = model(x1, x1)
    print(output)