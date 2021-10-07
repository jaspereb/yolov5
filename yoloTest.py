#Test constructing the yolo dual network. Can't use a sequential model (because it's not sequential, duh)
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
            Conv(1024, 512, 1, 1), #Ch in, Ch out, kernel, stride
            nn.Upsample(None, 2, 'nearest') #Size, scale, method
        )
        #Cat backbone P4 goes here
        self.H2 = nn.Sequential(
            C3(1024, 512, 3), #Ch in, ch out, number
            Conv(512, 256, 1, 1), #Ch in, Ch out, kernel, stride
            nn.Upsample(None, 2, 'nearest'), #Size, scale, method
        )
        #Cat backbone P3 goes here
        self.H3 = nn.Sequential(
            C3(512, 256, 3), #Ch in, ch out, number
            Conv(256, 256, 3, 2), #Ch in, Ch out, kernel, stride
        )
        #Cat head P4 goes here
        self.H4 = nn.Sequential(
            C3(512, 512, 3), #Ch in, ch out, number
            Conv(512, 512, 3, 2), #Ch in, Ch out, kernel, stride
        )
        #Cat head P5 goes here
        self.H3 = nn.Sequential(
            C3(1024, 1024, 3) #Ch in, ch out, number
        )
        
        self.Detect = nn.Sequential(
            Detect(classes, anchors) #classes, anchors, ch, inplace
        )

        # Equivalent of
        # [[-1, 1, Conv, [512, 1, 1]],
        # [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # EXC [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        # [-1, 3, C3, [512, False]],  # 13

        # [-1, 1, Conv, [256, 1, 1]],
        # [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        # [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

        # [-1, 1, Conv, [256, 3, 2]],
        # [[-1, 14], 1, Concat, [1]],  # cat head P4
        # [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

        # [-1, 1, Conv, [512, 3, 2]],
        # [[-1, 10], 1, Concat, [1]],  # cat head P5
        # [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

        # [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
        # ]  

        self.classifier = nn.Linear(65536, 4)
        
    def forward(self, img1, img2):
        b1 = self.B1S1(img1)
        b2 = self.B2S1(img2)
        b3 = self.B1S2(b1)
        b4 = self.B2S2(b2)
        ...



        x1 = self.backbone1(img1)
        x2 = self.backbone1(img2)

        x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        
        # x = torch.cat((x1, x2, x3), dim=1)

        x = self.classifier(x1)
        return x

classes = 1
anchors = ?
model = YoloDual(classes, anchors)
batch_size = 1
x1 = torch.randn(batch_size, 3, 256, 256)
# x2 = torch.randn(batch_size, 1, 64, 64)
# x3 = torch.randn(batch_size, 10)

# output = model(x1, x2, x3)
output = model(x1)
print(output)