#Test constructing the yolo dual network. Can't use a sequential model (because it's not sequential, duh)
#This is based on yolov5l.yaml and uses the same numbering system

import torch.nn as nn
import torch 
from models.common import *
from models.yolo import Detect

class YoloDual(nn.Module):
    def __init__(self, classes, anchors):
        super(YoloDual, self).__init__()
        # self.backbone1 = nn.Sequential(
        #     Focus(3,64), #Ch in, Ch out
        #     Conv(64, 128, 3, 2), #Ch in, Ch out, kernel, stride
        #     C3(128, 128, 3), #Ch in, ch out, number
        #     Conv(128, 256, 3, 2), #Ch in, Ch out, kernel, stride
        #     C3(256, 256, 9), #Ch in, ch out, number
        #     Conv(256, 512, 3, 2), #Ch in, Ch out, kernel, stride
        #     C3(512, 512, 9), #Ch in, ch out, number
        #     Conv(512, 1024, 3, 2), #Ch in, Ch out, kernel, stride
        #     SPP(1024, 1024, [5, 9, 13]), #Ch in, Ch out, 3-tuple
        #     C3(1024, 1024, 3) #Ch in, ch out, number

        #     #Equivalent of:
        #     # [[-1, 1, Focus, [64, 3]],  # 0-P1/2
        #     # [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        #     # [-1, 3, C3, [128]],
        #     # [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        #     # [-1, 9, C3, [256]],
        #     # [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        #     # [-1, 9, C3, [512]],
        #     # [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        #     # [-1, 1, SPP, [1024, [5, 9, 13]]],
        #     # [-1, 3, C3, [1024, False]],  # 9
        # )

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

        # Equivalent of
        # [[-1, 1, Conv, [512, 1, 1]],
        # [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # EXC [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        # [-1, 3, C3, [512, False]],  # 13

        # [-1, 1, Conv, [256, 1, 1]],
        # [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # EXC [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        # [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

        # [-1, 1, Conv, [256, 3, 2]],
        # EXC [[-1, 14], 1, Concat, [1]],  # cat head P4
        # [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

        # [-1, 1, Conv, [512, 3, 2]],
        # EXC [[-1, 10], 1, Concat, [1]],  # cat head P5
        # [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

        # [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
        # ]  
        
    def forward(self, img1, img2):
        x = self.forward_split(img1,img2)
        return x
    
    def forward_siamese():
        raise NotImplementedError("MAke ME")
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

classes = 1
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
model = YoloDual(classes, anchors)
batch_size = 1
x1 = torch.randn(batch_size, 3, 256, 256)

# output = model(x1, x2, x3)
output = model(x1, x1)
print(output)