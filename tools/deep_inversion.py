from torch import nn
import torch   
from .cnn_tools import batch_norm

class DeepInversionDiscriminator(nn.Module):
    '''
    in_ch - number of images to compare
    Note that sigmoid is removed in favor to better generalizability
    The depth of the network is adjusted such that input images be 64x64
    Code taken from https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/discriminator/dis.py
    See https://arxiv.org/pdf/1811.05910.pdf
    '''
    def __init__(self, in_ch, bn='BatchNorm'):
        super().__init__()

        self.in_ch = in_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1,
                padding_mode='circular'),
            res_unit(32, 32, ny=64, nx=64, bn='None'),
            down(32,   64, ny=64, nx=64, bn=bn), # 64x64 -> 32x32
            down(64,  128, ny=32, nx=32, bn=bn), # 32x32 -> 16x16
            down(128, 256, ny=16, nx=16, bn=bn), # 16x16 -> 8x8
            down(256, 512, ny=8,  nx=8,  bn=bn)  # 8x8   -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(4*4*512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def test(self):
        x = torch.randn(2,self.in_ch,64,64)
        error_x = torch.roll(self(torch.roll(x,1,-1)),-1,-1)-self(x)
        error_y = torch.roll(self(torch.roll(x,1,-2)),-1,-2)-self(x)
        e = (error_x**2+error_y**2).mean()
        return f'Circular error = {e}'

class DeepInversionGenerator(nn.Module):
    '''
    n_in - number of input images
    n_out - number of output images
    Code is written according to picture 8 from:
    https://arxiv.org/pdf/1811.05910.pdf
    Upsampling is inspired by:
    https://github.com/mateuszbuda/brain-segmentation-pytorch
    I.e., concatenation happens with signal after upsampling and
    before downsampling
    '''
    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        
        self.conv32 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1,
            padding_mode='circular')
        self.res32_start = res_unit(32, 32, ny=64, nx=64, bn='None')
        
        self.down64  = down(32, 64,   ny=64, nx=64)    # 32x64x64  -> 64x32x32
        self.down128 = down(64, 128,  ny=32, nx=32)    # 64x32x32  -> 128x16x16
        self.down256 = down(128, 256, ny=16, nx=16)    # 128x16x16 -> 256x8x8
        self.down512 = down(256, 512, ny=8,  nx=8)     # 256x8x8   -> 512x4x4

        self.res512 = res_unit(512, 512, ny=4, nx=4)   # 512x4x4   -> 512x4x4

        self.up512 = up(512, 256, ny=4, nx=4)          # 512x4x4   -> cat(256x8x8,256x8x8) -> 256x8x8
        self.up256 = up(256, 128, ny=8, nx=8)
        self.up128 = up(128, 64,  ny=16, nx=16)
        self.up64  = up(64, 32,   ny=32, nx=32)
        self.res32_end = res_unit(32, 32, ny=64, nx=64, bn='None')
        self.conv_end = nn.Conv2d(32, n_out, kernel_size=1)
        
    def forward(self, x):
        x = self.conv32(x)
        im64 = self.res32_start(x)
        im32 = self.down64(im64)
        im16 = self.down128(im32)
        im8  = self.down256(im16)
        im4  = self.down512(im8)

        im4 = self.res512(im4)

        im8  = self.up512(im4, im8) # upsample and concatenate
        im16 = self.up256(im8, im16)
        im32 = self.up128(im16, im32)
        im64 = self.up64(im32, im64)
        x = self.res32_end(im64)
        x = self.conv_end(x)
        return x
    
    def test(self):
        x = torch.randn(2,self.n_in,64,64)
        error_x = torch.roll(self(torch.roll(x,1,-1)),-1,-1)-self(x)
        error_y = torch.roll(self(torch.roll(x,1,-2)),-1,-2)-self(x)
        e = (error_x**2+error_y**2).mean()
        return f'Circular error = {e}'


class res_unit(nn.Module):
    def __init__(self, in_ch, out_ch, ny=2, nx=2, bn='BatchNorm'):
        '''
        ny, nx - resolution of input and output image. In most
        cases (except of bn='LayerNorm' it is not necessary)
        '''
        super().__init__()
        self.bn = batch_norm(bn, in_ch, ny, nx)
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1,
                padding_mode='circular'),
            batch_norm(bn, out_ch, ny, nx),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, 
                padding_mode='circular'),
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.bn(x)) + self.conv1(self.bn(x))

class down(nn.Module):
    '''
    Takes tensor: Nbatch x in_ch x Ny x Nx.
    Returns tensor: Nbatch x out_ch x Ny/2 x Nx/2 
    '''
    def __init__(self, in_ch, out_ch, ny=2, nx=2, bn='BatchNorm'):
        '''
        ny, nx - resolution before coarsening
        '''
        super().__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(2, 2),
            res_unit(in_ch, out_ch, ny//2, nx//2, bn)
        )
    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    '''
    Takes tensor: Nbatch x in_ch x Ny x Nx.
    Transforms to: Nbatch x in_ch/2 x Ny*2 x Nx*2
    Concatenetes with tensor: Nbatch x in_ch/2 x Ny*2 x Nx*2
    Returns tensor: Nbatch x out_ch x Ny*2 x Nx*2 
    '''
    def __init__(self, in_ch, out_ch, ny=2, nx=2, bn='BatchNorm'):
        '''
        ny, nx - resolution before upsampling
        '''
        super().__init__()
        # https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py#L32-L34
        self.upsampling = nn.ConvTranspose2d(
            in_ch, int(in_ch/2), kernel_size=2, stride=2
        )
        self.conv = res_unit(in_ch, out_ch, ny, nx, bn)
    def forward(self, x, y):
        return self.conv(torch.cat((self.upsampling(x),y),dim=1))