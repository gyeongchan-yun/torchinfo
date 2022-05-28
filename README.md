# torchinfo

Original code comes from [torchinfo](https://github.com/TylerYep/torchinfo/tree/v1.5.4) vesion 1.5.4.

## Software packages
- Python >=3.6

## Build from source code
```
python setup.py install
```

# Quick start

```python
import torchvision.models as models

from torchinfo import summary


model = models.resnet18(num_classes=10)
batch_size = 16
summary(model, input_size=(batch_size, 3, 32, 32))
```

## Output
Output shows **parameter size per layer** and **intermediate output size per layer**.
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   --                        --
├─Conv2d: 1-1                            [16, 64, 16, 16]          9,408
├─BatchNorm2d: 1-2                       [16, 64, 16, 16]          128
├─ReLU: 1-3                              [16, 64, 16, 16]          --
├─MaxPool2d: 1-4                         [16, 64, 8, 8]            --
├─Sequential: 1-5                        [16, 64, 8, 8]            --
│    └─BasicBlock: 2-1                   [16, 64, 8, 8]            --
│    │    └─Conv2d: 3-1                  [16, 64, 8, 8]            36,864
│    │    └─BatchNorm2d: 3-2             [16, 64, 8, 8]            128
│    │    └─ReLU: 3-3                    [16, 64, 8, 8]            --
│    │    └─Conv2d: 3-4                  [16, 64, 8, 8]            36,864
│    │    └─BatchNorm2d: 3-5             [16, 64, 8, 8]            128
│    │    └─ReLU: 3-6                    [16, 64, 8, 8]            --
│    └─BasicBlock: 2-2                   [16, 64, 8, 8]            --
│    │    └─Conv2d: 3-7                  [16, 64, 8, 8]            36,864
│    │    └─BatchNorm2d: 3-8             [16, 64, 8, 8]            128
│    │    └─ReLU: 3-9                    [16, 64, 8, 8]            --
│    │    └─Conv2d: 3-10                 [16, 64, 8, 8]            36,864
│    │    └─BatchNorm2d: 3-11            [16, 64, 8, 8]            128
│    │    └─ReLU: 3-12                   [16, 64, 8, 8]            --

...

==========================================================================================
Total params: 11,181,642
Trainable params: 11,181,642
Non-trainable params: 0
Total mult-adds (M): 564.97
==========================================================================================
Input size (MB): 0.19
Forward/backward pass size (MB): 12.38
Params size (MB): 42.65
Estimated Total Size (MB): 55.22
==========================================================================================
Parameter size per layer (MB)
Conv2d (conv1): 0.036
BatchNorm2d (bn1): 0.00049
Conv2d (conv1): 0.14
BatchNorm2d (bn1): 0.00049
Conv2d (conv2): 0.14
BatchNorm2d (bn2): 0.00049
Conv2d (conv1): 0.14
BatchNorm2d (bn1): 0.00049

...

==========================================================================================
Output size per layer (MB)
Conv2d (conv1): 1.0
BatchNorm2d (bn1): 1.0
Conv2d (conv1): 0.25
BatchNorm2d (bn1): 0.25
Conv2d (conv2): 0.25
BatchNorm2d (bn2): 0.25
Conv2d (conv1): 0.25
BatchNorm2d (bn1): 0.25
Conv2d (conv2): 0.25
BatchNorm2d (bn2): 0.25

==========================================================================================
```

# Verbosity
If the level of verbosity is 2 (default: 1), summary shows **the memory size (MB) of each parameters**.

```python
summary(model, input_size=(batch_size, 3, 32, 32), verbose=2)
```

## Output
```
============================================================================================================================================
Layer (type:depth-idx)                   Output Shape              Param #                   Param Mem (MB)            Output Mem (MB)
============================================================================================================================================
ResNet                                   --                        --                                                  --
├─Conv2d: 1-1                            [16, 64, 16, 16]          9,408                                               1.0
│    └─weight                                                      └─9,408                   └─0.036
├─BatchNorm2d: 1-2                       [16, 64, 16, 16]          128                                                 1.0
│    └─weight                                                      ├─64                      ├─0.0
│    └─bias                                                        └─64                      └─0.0
├─ReLU: 1-3                              [16, 64, 16, 16]          --                                                  --
├─MaxPool2d: 1-4                         [16, 64, 8, 8]            --                                                  --
├─Sequential: 1-5                        [16, 64, 8, 8]            --                                                  --
│    └─0.conv1.weight                                              ├─36,864                  ├─0.141
│    └─0.bn1.weight                                                ├─64                      ├─0.0
│    └─0.bn1.bias                                                  ├─64                      ├─0.0
│    └─0.conv2.weight                                              ├─36,864                  ├─0.141
│    └─0.bn2.weight                                                ├─64                      ├─0.0
│    └─0.bn2.bias                                                  ├─64                      ├─0.0
│    └─1.conv1.weight                                              ├─36,864                  ├─0.141
│    └─1.bn1.weight                                                ├─64                      ├─0.0
│    └─1.bn1.bias                                                  ├─64                      ├─0.0
│    └─1.conv2.weight                                              ├─36,864                  ├─0.141
│    └─1.bn2.weight                                                ├─64                      ├─0.0
│    └─1.bn2.bias                                                  └─64                      └─0.0

...

├─Sequential: 1-8                        [16, 512, 1, 1]           --                                                  --
│    └─0.conv1.weight                                              ├─1,179,648               ├─4.5
│    └─0.bn1.weight                                                ├─512                     ├─0.002
│    └─0.bn1.bias                                                  ├─512                     ├─0.002
│    └─0.conv2.weight                                              ├─2,359,296               ├─9.0
│    └─0.bn2.weight                                                ├─512                     ├─0.002
│    └─0.bn2.bias                                                  ├─512                     ├─0.002
│    └─0.downsample.0.weight                                       ├─131,072                 ├─0.5
│    └─0.downsample.1.weight                                       ├─512                     ├─0.002
│    └─0.downsample.1.bias                                         ├─512                     ├─0.002
│    └─1.conv1.weight                                              ├─2,359,296               ├─9.0
│    └─1.bn1.weight                                                ├─512                     ├─0.002
│    └─1.bn1.bias                                                  ├─512                     ├─0.002
│    └─1.conv2.weight                                              ├─2,359,296               ├─9.0
│    └─1.bn2.weight                                                ├─512                     ├─0.002
│    └─1.bn2.bias                                                  └─512                     └─0.002

...
```
