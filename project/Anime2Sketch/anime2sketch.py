import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import functools
import pdb

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # input_nc = 3
        # output_nc = 1
        # num_downs = 8
        # ngf = 64
        # norm_layer = functools.partial(<class 'torch.nn.modules.instancenorm.InstanceNorm2d'>, 
        # affine=False, track_running_stats=False)
        # use_dropout = False


        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        # For imporoved model ...
        base = self.model.model[1]
        # swap deconvolution layers with reszie + conv layers for 2x upsampling
        for _ in range(6):
            inc, outc = base.model[5].in_channels, base.model[5].out_channels
            base.model[5] = Upsample(inc, outc)
            base = base.model[3]

        # pdb.set_trace()
        self.load_weights()

    def load_weights(self, model_path="models/Anime2Sketch.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            weight_state = torch.load(checkpoint)
            # for normal weigth file
            # for key in list(weight_state.keys()):
            #     if 'module.' in key:
            #         weight_state[key.replace('module.', '')] = weight_state[key]
            #         del weight_state[key]
            self.load_state_dict(weight_state)
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")


    def forward(self, input):
        """Standard forward"""
        input = (input - 0.5) * 2.0
        output = self.model(input)
        output = (output + 1.0)/2.0
        
        return output


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)
        

class Upsample(nn.Module):
    def __init__(self, inc, outc, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.smooth = Smooth()
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(outc, 4 * outc, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * outc, outc, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.smooth(self.up(x))
        x = self.conv(x)
        x = self.mlp(x) + x
        return x


# def create_model(model):
#     """Create a model for anime2sketch
#     hardcoding the options for simplicity
#     """

#     norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#     net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)

#     if model == 'default':
#         weight_state = torch.load('weights/netG.pth')
#         for key in list(weight_state.keys()):
#             if 'module.' in key:
#                 weight_state[key.replace('module.', '')] = weight_state[key]
#                 del weight_state[key]
#         net.load_state_dict(weight_state)

#     elif model == 'improved':
#         weight_state = torch.load('weights/improved.bin', map_location=torch.device('cpu'))
#         base = net.model.model[1]

#         # swap deconvolution layers with reszie + conv layers for 2x upsampling
#         for _ in range(6):
#             inc, outc = base.model[5].in_channels, base.model[5].out_channels
#             base.model[5] = Upsample(inc, outc)
#             base = base.model[3]

#         net.load_state_dict(weight_state)
    
#     else:
#         raise ValueError(f"model should be one of ['default', 'improved'], but got {model}")

#     return net


