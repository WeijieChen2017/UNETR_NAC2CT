from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm3d, GroupNorm, InstanceNorm3d
from torch.nn import ELU, LeakyReLU, ReLU, Linear
from torch.nn import Dropout3d, MaxPool3d, AdaptiveMaxPool3d
from torch import nn

class Net(nn.Module):

    def __init__(self, block_size=64, num_filters=16, num_level=2, verbose=False):
        super(Net, self).__init__()
        self.block_size = block_size
        self.num_filters = num_filters
        self.num_level = num_level
        self.net_list = unet3d(self.block_size, 
                               self.num_filters,
                               self.num_level)
        if verbose:
            network_visualization(self.net_list)

        self.network_layer = nn.Sequential()
        for idx, item in enumerate(self.net_list):
            self.network_layer.add_module(str(idx), item)

    def forward(self, x):
        return self.network_layer(x)

    def summary(self, x):
        print("\33[0m", "-"*25, "Start", "-"*25)
        for item in enumerate(self.network_layer):
            layer = item[1]
            color_layer(layer)
            x = layer(x)
            print(x.size())
        print("\33[0m", "-"*25, "Over!", "-"*25)


def color_layer(layer):
    if isinstance(layer, Conv3d):
        print("\33[34m", layer)
    elif isinstance(layer, (ELU, LeakyReLU, ReLU)):
        print("\33[33m", layer)
    elif isinstance(layer, (BatchNorm3d, GroupNorm, InstanceNorm3d)):
        print("\33[32m", layer)
    elif isinstance(layer, (MaxPool3d, ConvTranspose3d)):
        print("\33[31m", layer)
    else:
        print("\33[35m", layer)

def convBlock(in_channels, out_channels, num_groups, norm_type, acti_type):

    convBlock = []

    # add conv layer
    # input size (N,C,D,H,W)
    convBlock.append(Conv3d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            groups=num_groups,
                            padding=1))
    # add norm layer
    if norm_type == "batch":
        # num_features
        convBlock.append(BatchNorm3d(num_features=out_channels))
    elif norm_type == "group":
        convBlock.append(GroupNorm(num_groups=num_groups, num_features=out_channels))
    elif norm_type == "instance":
        convBlock.append(InstanceNorm3d(num_features=out_channels))
    elif norm_type == "none":
        pass

    # add activation layer
    if acti_type == "ReLU":
        convBlock.append(ReLU())
    elif acti_type == "ELU":
        convBlock.append(ELU())
    elif acti_type == "LeakyReLU":
        convBlock.append(LeakyReLU())

    return convBlock

def unet3d(block_size=64, num_filters=16, num_level=2, num_groups=1):

    unet3d = []

    block = convBlock(in_channels = 1,
                      out_channels = num_filters,
                      num_groups = num_groups,
                      norm_type = "batch",
                      acti_type = "LeakyReLU")
    unet3d.append(block)

    # encoder
    for idx in range(num_level):
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        unet3d.append(block)
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters * 2,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        unet3d.append(MaxPool3d(kernel_size=3, stride=2, padding=1))
        num_filters = num_filters * 2

    # lowest learning
    block = convBlock(in_channels = num_filters,
                      out_channels = num_filters,
                      num_groups = num_groups,
                      norm_type = "batch",
                      acti_type = "LeakyReLU")
    unet3d.append(block)
    unet3d.append(block)
    unet3d.append(Dropout3d())
    unet3d.append(block)
    unet3d.append(block)

    # decoder
    for idx in range(num_level):
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(ConvTranspose3d(in_channels=num_filters,
                                      out_channels=num_filters,
                                      kernel_size=3,
                                      groups=num_groups,
                                      stride=2,
                                      padding=1,
                                      output_padding=1))
        unet3d.append(block)
        unet3d.append(block)
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters // 2,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        num_filters = num_filters // 2
    unet3d.append(Conv3d(in_channels=num_filters,
                         out_channels=num_filters,
                         kernel_size=3,
                         groups=num_groups,
                         padding=1))
    unet3d.append(Conv3d(in_channels=num_filters,
                         out_channels=1,
                         kernel_size=3,
                         groups=num_groups,
                         padding=1))
    unet3d.append(Linear(in_features = block_size, out_features = block_size))

    # flatten the list of layer
    unet3d_flatten = []
    for item in unet3d:
        if isinstance(item, list):
            for elem in item:
                unet3d_flatten.append(elem)
        else:
            unet3d_flatten.append(item)
    # network_visualization(unet3d_flatten)

    return unet3d_flatten


