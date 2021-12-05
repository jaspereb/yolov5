#Construct a yolo-like model which is dual input for multiple image inputs.
import sys
sys.path.append("/home/jasper/git/yolov5")

import torch.nn as nn
import torch 
from models.common import *
from models.yolo import Detect
from utils.autoanchor import check_anchor_order
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

LOGGER = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, classes, anchors, siamese=False):
        super(Model, self).__init__()
        self.siamese = siamese
        LOGGER.info(f'Now building dual input Yolo Model')

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
        ch = 3
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s),torch.zeros(1, ch, s, s))])  # forward
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride
        self._initialize_biases()  # only run once
        
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, img1, img2):
        if(self.siamese):
            LOGGER.info(f'Now linking siamese dual Yolo Model')
            x  = self.forward_siamese(img1,img2)
        else:
            LOGGER.info(f'Now linking non-siamese dual Yolo Model')
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

        x = h13

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

        x = h13

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