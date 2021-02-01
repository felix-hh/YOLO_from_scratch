import torch
import torch.nn as nn

"""
Represent architecture for parameterized construction. 
A convolutional layer contains a Convolutional Layer, a Leaky ReLU and a BatchNorm. 

- Convolutional Block: tuple (out_channels, kernel_size, stride, padding)
- Maxpool layer: "M" always kernel_size= 2 and stride= 2
- Repetitions: list of the previous items [item, ..., item, num_reps: int]

Formula to calculate padding:
Out_dim = (In_dim - Kernel + 2*Padding + Stride)/Stride

In_channels is calculated autmatically.
"""

darknet_architecture = [ # INPUT: 3x448x448
    (64, 7, 2, 3), # 64x224x224
    "M", # 64x112x112
    (192, 3, 1, 1), # 192x112x112
    "M", # 192x56x56
    (128, 1, 1, 0), # 128x56x56
    (256, 3, 1, 1), # 256x56x56
    (256, 1, 1, 0), # 256x56x56
    (512, 3, 1, 1), # 512x56x56
    "M", # 512x28x28
    [(256, 1, 1, 0), (512, 3, 1, 1), 4], # 512x28x28
    (512, 1, 1, 0), # 512x28x28
    (1024, 3, 1, 1), # 1024x28x28
    "M", # 1024x14x14
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2], # 1024x14x14
    (1024, 3, 1, 1), # 1024x14x14
    (1024, 3, 2, 1), # 1024x7x7
    (1024, 3, 1, 1), # 1024x7x7
    (1024, 3, 1, 1), # 1024x7x7
]
#This is the darknet architecture for the original YOLOv1

"""
Defines the Convolutional Block. It is a nn.module that contains: 
- Conv2d
- LeakyReLU
- BatchNorm2d
"""

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias= False, **kwargs) #batchnorm already includes a bias. 
        self.activation = nn.LeakyReLU(0.1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, batch):
        x = self.conv(batch)
        x = self.activation(x)
        x = self.batchnorm(x)
        return x

class Yolo(nn.Module):
    def __init__(self, architecture, in_channels= 3, split_size= 7):
        super(Yolo, self).__init__()
        # Call super and init constants
        self.S = split_size
        # Execute construction functions for backbone and head
        self.backbone = self._create_backbone(architecture, in_channels)
        self.head = self._create_head()

    def forward(self, batch):
        x = self.backbone(batch)
        x = x.view(-1, self.S * self.S * 1024)
        x = self.head(x)
        x = x.view(self.S, self.S, 30)
        return x

    def _create_backbone(self, architecture, in_channels):
        #auxiliar functions to add ConvBlocks and Maxpools        

        #Loop through the architecture list and save to a list for nn.Sequential
        blocks = []
        for item in architecture:
            #If tuple: create a conv block. update in_channels. append to blocks
            if type(item) == tuple:
                out_channels = item[0]
                kwargs = {
                    "kernel_size": item[1],
                    "stride": item[2],
                    "padding": item[3]
                }
                block = ConvolutionalBlock(in_channels, out_channels, **kwargs)
                blocks.append(block)
                in_channels = out_channels
            #If "M": create a maxpool. in_channels stays constant. append to blocks
            elif type(item) == str:
                if item == "M":
                    blocks.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
                else:
                    raise Exception(f"Unrecognized string {item} to create a new block in the backbone")
            #If list: iterate by num repeats. create each item up to [:-1]
            elif type(item) == list:
                repeat_count = item[-1]
                list_items = item[:-1]
                for _ in range(repeat_count):
                    for list_item in list_items:
                        # This is repeat code. Ideally we would have a add_item function to abstract type checks.
                        # If tuple: create a conv block. update in_channels. append to blocks
                        if type(list_item) == tuple:
                            out_channels = list_item[0]
                            kwargs = {
                                "kernel_size": list_item[1],
                                "stride": list_item[2],
                                "padding": list_item[3]
                            }
                            block = ConvolutionalBlock(in_channels, out_channels, **kwargs)
                            blocks.append(block)
                            in_channels = out_channels
                # End list loop
            # End list type conditional
            #If it's not list, tuple or string, we have an unrecognized element. 
            else:
                raise Exception(f"Unrecognized type {str(type(item))} to create a new block in the backbone")
        #End architecture loop
        #Build a nn.Sequential module with our blocks, then return. 
        backbone = nn.Sequential(*blocks)
        return backbone

    def _create_head(self):
        blocks = [
            torch.nn.Linear(self.S * self.S * 1024, 4096),
            torch.nn.Linear(4096, self.S * self.S * 30)
        ]
        return nn.Sequential(*blocks)


def test1():
    darknet = darknet_architecture
    model = Yolo(architecture= darknet)
    x = torch.rand([1, 3, 448, 448])
    model(x)

    assert(list(model(x).shape) == [7, 7, 30])

    return 
