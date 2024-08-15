
from utils.ResNet import ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time

class HAIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
            
        self.latent= HyLevel(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, 
                              bias=bias, LayerNorm_type=LayerNorm_type,num_blocks=num_blocks[3],num_box=int(max(4,num_blocks[3]))+1,ch_fea=dim*2,mlp_depth=3) 
        self.reduce_latent=nn.Conv2d(int(dim*2**3),int(dim*2**2),3,1,1)
        self.feature=ResNet(ch=int(dim*1.5),out_ch=dim*2,in_channels=int(dim*2**3),resolution=10)
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = HyLevel(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,\
            LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2],num_box=int(max(4,num_blocks[2])),ch_fea=dim*2,mlp_depth=3)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
   
        self.decoder_level2 = HyLevel(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,\
            bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1], num_box=int(max(4,num_blocks[1]))+1,ch_fea=dim*2,mlp_depth=3)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = HyLevel(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,\
            LayerNorm_type=LayerNorm_type,num_blocks=num_blocks[0],num_box=int(max(4,num_blocks[0]))+1,ch_fea=dim*2,mlp_depth=3)
        
        
        self.refinement = HyLevel(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, \
            LayerNorm_type=LayerNorm_type,num_blocks=num_refinement_blocks,num_box=int(max(4,num_refinement_blocks))+1,ch_fea=dim*2,mlp_depth=3)
    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)     
        
        feature=self.feature(inp_enc_level4)
        feature=feature.mean(dim=(-1,-2))
           
        latent = self.latent(inp_enc_level4,feature)
        latent = self.reduce_latent(latent)
                               
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3,feature) 
    
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2,feature)
           
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1,feature)
        out_dec_level1 = self.refinement(out_dec_level1,feature)
        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


class HyConv2d(nn.Module):
    """
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', bias=None):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, w):
        assert x.shape[0] == w.shape[0]
        batch_size = x.shape[0]
        x = x.view(1, -1, *x.shape[2:])
        w = w.reshape(w.shape[0] * self.out_channels, self.in_channels // self.groups, self.kernel_size,self.kernel_size)
 
        x = F.conv2d(x, w, bias=None, stride=self.stride, dilation=self.dilation, padding=self.padding,
                     groups=batch_size * self.groups)
        x = x.view(batch_size, -1, *x.shape[2:])

        return x

class HyFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = HyConv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = HyConv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = HyConv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, ws):
        x = self.project_in(x,ws[0])
        x1, x2 = self.dwconv(x,ws[1]).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x,ws[2])
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Hyattn(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = HyConv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = HyConv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = HyConv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, ws):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x,ws[0]),ws[1])
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out,ws[2])
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class Hytrans(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Hyattn(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HyFF(dim, ffn_expansion_factor, bias)

    def forward(self, x,wei_attn,wei_ffn):
        x = x + self.attn(self.norm1(x),wei_attn)
        x = x + self.ffn(self.norm2(x),wei_ffn)

        return x

class FullyCon(nn.Module):
    def __init__(self,depth,num_box,ch_fea):
        super().__init__()
        
        ch=ch_fea
        self.net=nn.Sequential()
        for i in range(depth):
            self.net.append(nn.Linear(ch,ch//2))
            self.net.append(nn.GELU())
            ch=ch//2 
        
        self.net.append(nn.Linear(ch,num_box))
    
    def forward(self,x):
        return F.softmax(self.net(x),dim=-1)

class HyLevel(nn.Module):
    
    def __init__(self,dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, num_box, ch_fea, mlp_depth):
        
        super().__init__()
        self.blocks=nn.Sequential(*[Hytrans(dim=dim ,num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, \
            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
        hidden=int(dim*ffn_expansion_factor)
        self.num_blocks=num_blocks
        self.extracts=nn.Sequential(*[FullyCon(mlp_depth,num_box,ch_fea) for i in range(num_blocks) ])
        num_weights=[3*dim*dim*1*1,3*dim*1*3*3,dim*dim*1*1,2*hidden*dim*1*1,2*hidden*1*3*3,hidden*dim*1*1]
        
        self.kernel_box=torch.zeros(num_box,sum(num_weights))
        Sum=[0]
        for i in range(6):
            Sum.append(Sum[i]+num_weights[i])
        self.sum_weights=Sum
        
        for i in range(num_box):
            params=[nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias),
                    nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias),
                    nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                    nn.Conv2d(dim, hidden*2, kernel_size=1, bias=bias),
                    nn.Conv2d(hidden*2, hidden*2, kernel_size=3, stride=1, padding=1, groups=hidden*2, bias=bias),
                    nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)]
            for j in range(len(params)):
                self.kernel_box[i,Sum[j]:Sum[j+1]]=params[j].weight.reshape(-1)
        
        self.kernel_box=nn.Parameter(self.kernel_box)
        
        
    def forward(self,x,feature):
        
        Sum=self.sum_weights
        
        for i in range(self.num_blocks):
            fea_now=self.extracts[i](feature)
            weights=fea_now@self.kernel_box
            wei_attn=[ weights[:,Sum[0]:Sum[1]], weights[:,Sum[1]:Sum[2]], weights[:,Sum[2]:Sum[3]] ]
            wei_ffn=[ weights[:,Sum[3]:Sum[4]], weights[:,Sum[4]:Sum[5]], weights[:,Sum[5]:] ]
            x=self.blocks[i](x,wei_attn, wei_ffn)
        
        return x 
        

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out




class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x