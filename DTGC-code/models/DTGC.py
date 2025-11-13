from timm.models.layers import trunc_normal_, DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class upsample(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.layer_norm = LayerNorm2D(in_dim, eps=1e-6)
        self.trans_conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.gelu = nn.GELU()
    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x).permute(0, 3, 1, 2)
        return self.gelu(self.trans_conv(x))

class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class globalextractor(nn.Module):
    def __init__(self,mode='gap'):
        super(globalextractor, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mode = mode
    def forward(self, x):
        T,C,H,W = x.size()
        if self.mode == 'gap':
            x = self.gap(x).view(T,C)
            return x
        elif self.mode =='gwap':
            x_flat = x.view(T,C,H * W)
            M_Gt = F.softmax(x_flat, dim=2)
            gwap = (M_Gt * x_flat).sum(dim=2).view(T,C)
            return gwap
        elif self.mode =='cgwap':
            x_perm = x.permute(0, 2, 3, 1)
            x_soft = F.softmax(x_perm, dim=3)
            x_weighted = x_perm * x_soft
            gwap = x_weighted.sum(dim=3)
            return gwap
class globalfeaturefuse(nn.Module):
    def __init__(self, dim, heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, y):
        T, C = x.shape
        q = self.to_q(x).view(T, self.heads, self.head_dim) 
        k = self.to_k(y).view(T, self.heads, self.head_dim)
        v = self.to_v(y).view(T, self.heads, self.head_dim)
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale 
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  
        out = out.contiguous().view(T, C)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out + x 



class temporalfuse(nn.Module):
    def __init__(self, dim, heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Conv2d(dim,dim,1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, y, z):
        T, C, H, W = z.shape
        q = self.to_q(x).view(T, self.heads, self.head_dim)  
        k = self.to_k(y).view(T, self.heads, self.head_dim)  
        q = q.permute(1,0,2)  
        k = k.permute(1,0,2)  
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        z = z.permute(0, 2, 3, 1)  
        z_heads = z.view(T, H, W, self.heads, self.head_dim).permute(3, 0, 1, 2, 4) 
        out = torch.einsum('htt,htwcd->thdcw', attn, z_heads) 
        out = out.permute(0, 2, 3, 1, 4).reshape(T, C, H, W) 
        out = self.proj_drop(self.to_out(out))
        z = z.permute(0,3,1,2)
        return out + z

class spacefuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.cgwap = globalextractor(mode='cgwap')
        self.gwap = globalextractor(mode='gap')

    def forward(self, x):
        t, c, h, w = x.size()
        x1 = self.cgwap(x)  
        x2 = self.gwap(x)   
        x1_flat = x1.view(t, -1)  
        M_S = torch.einsum('bi,bj->bij', x1_flat, x1_flat) 
        M_S = torch.sigmoid(M_S)
        channel_map = torch.einsum('bi,bj->bij', x2, x2)  
        channel_map = torch.sigmoid(channel_map)

        x_flat = x.view(t, c, -1)  
        x_attn = torch.einsum('bij,bjk->bik', channel_map, x_flat)  
        x_attn = x_attn.view(t, c, h, w)

        x_flat_t = x.view(t, c, -1).transpose(1, 2)  
        y_attn = torch.einsum('bij,bjk->bik', M_S, x_flat_t)  
        y_attn = y_attn.transpose(1, 2).view(t, c, h, w)
        return x_attn + y_attn

class MGCA(nn.Module):
    def __init__(self,dim,heads,mode='tpf',attn_drop=0., proj_drop=0.,tp_drop=0.):
        super().__init__()
        if mode == 'tpf':
            self.globalfuse = temporalfuse(dim,heads=heads,attn_drop=attn_drop,proj_drop=proj_drop)
        elif mode == 'glf':
            self.globalfuse = globalfeaturefuse(dim)
        self.globlef1 = globalextractor(mode='gwap')
        self.globlef2 = globalextractor(mode='gap')
        self.spacefuse = spacefuse()
        self.droppath1 = DropPath(tp_drop) if tp_drop > 0 else nn.Identity()

    def forward(self,x):
        x0 = self.spacefuse(x)
        x1 = self.droppath1(x0) + x
        y1 = self.globlef1(x1)
        x2 = self.globalfuse(y1,y1,x) + x0
        return x2

class CMR(nn.Module):
    def __init__(self, dim,re):
        super(CMR, self).__init__()
        self.mid_dim = dim // re
        self.dwconv1 = nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim)
        self.dw_weights = nn.Parameter(torch.ones(3))
        self.down = nn.Conv2d(dim, self.mid_dim, 1)
        self.up = nn.Conv2d(self.mid_dim, dim, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            LayerNorm2D(dim, eps=1e-6, data_format='channels_first'),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            LayerNorm2D(dim, eps=1e-6, data_format='channels_first'),
            nn.GELU()
        )
        self.norm_fuse = LayerNorm2D(dim, eps=1e-6, data_format='channels_first')
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        w = torch.softmax(self.dw_weights, dim=0)
        x2 = w[0] * self.dwconv1(x) + w[1] * self.dwconv2(x) + w[2] * self.dwconv3(x)
        x2 = self.act(self.norm_fuse(x2))
        y1 = self.conv1(y)
        y2 = F.adaptive_avg_pool2d(y1, output_size=(1, 1))
        y3 = self.sigmoid(self.up(self.act(self.down(y2))) + y2)
        fused = y3 * x2
        fused = self.conv2(fused)
        return fused + x


class GCS(nn.Module):
    def __init__(self, fPlane, reverse=False, ratio=0.5, shift_first=True):
        super(GCS, self).__init__()
        self.reverse = reverse
        self.fPlane = fPlane
        self.ratio = ratio  # 通道参与时序建模的比例
        self.shift_first = shift_first

        C_shift = int(fPlane * ratio)
        assert C_shift % 2 == 0, "C_shift must be divisible by 2 for 2-group version"

        # 两个 3D 卷积（左移、右移）
        self.conv3D_l = nn.Conv3d(C_shift // 2, 1, (3, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3D_r = nn.Conv3d(C_shift // 2, 1, (3, 3, 3), stride=1, padding=(0, 1, 1))

        for conv in [self.conv3D_l, self.conv3D_r]:
            nn.init.constant_(conv.weight, 0)
            nn.init.constant_(conv.bias, 0)

        self.tanh = nn.Tanh()
        self.ln = LayerNorm2D(fPlane, data_format='channels_first')

    def lshift_cyclic(self, x, shift):
        return torch.roll(x, shifts=-shift, dims=2)

    def rshift_cyclic(self, x, shift):
        return torch.roll(x, shifts=shift, dims=2)

    def cyclic_pad(self, x, pad):
        left = x[:, :, -pad:]
        right = x[:, :, :pad]
        return torch.cat([left, x, right], dim=2)

    def forward(self, x):
        T, C, H, W = x.shape
        assert C == self.fPlane
        x_ln = x.permute(1, 0, 2, 3).unsqueeze(0)
        C_shift = int(C * self.ratio)
        C_keep = C - C_shift
        assert C_shift % 2 == 0, "C_shift must be divisible by 2"

        if self.shift_first:
            x_shift = x_ln[:, :C_shift]
            x_keep = x_ln[:, C_shift:]
        else:
            x_keep = x_ln[:, :C_keep]
            x_shift = x_ln[:, C_keep:]

        half = C_shift // 2
        if self.reverse:
            x_r = x_shift[:, :half]
            x_l = x_shift[:, half:]
        else:
            x_l = x_shift[:, :half]
            x_r = x_shift[:, half:]

        x_l_pad = self.cyclic_pad(x_l, pad=1)
        x_r_pad = self.cyclic_pad(x_r, pad=1)
        gate_l = self.tanh(self.conv3D_l(x_l_pad))
        gate_r = self.tanh(self.conv3D_r(x_r_pad))
        y_l = gate_l * x_l
        y_r = gate_r * x_r
        y_l = self.lshift_cyclic(y_l, 1) + (x_l - y_l)
        y_r = self.rshift_cyclic(y_r, 1) + (x_r - y_r)
        y_shift = torch.cat([y_l, y_r], dim=1)
        y_all = torch.cat([y_shift, x_keep], dim=1)
        y_out = y_all.squeeze(0).permute(1, 0, 2, 3)
        return y_out

class convblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv2 = nn.Conv2d(dim,dim, kernel_size=3, groups=dim, padding=1,dilation=1)
        self.silu = nn.SiLU()
        self.layernorm = LayerNorm2D(dim, data_format='channels_first')
    def forward(self, x):
        x = self.silu(self.layernorm(self.dwconv2(x)))
        return x


class TCI(nn.Module):
    def __init__(self, dim, ratio,se_ratio, reverse=False,shift_first=True):
        super().__init__()
        self.convblock = convblock(dim)
        self.gsm = GCS(fPlane=dim,reverse=reverse,ratio=ratio,shift_first=shift_first)
        self.lnorm = LayerNorm2D(dim, data_format='channels_first')
        self.seblock = SE_Block(dim,ratio=se_ratio)

    def forward(self, x):
        x = self.convblock(x)
        x_fused = self.gsm(x) + self.lnorm(x)
        x_fused = self.seblock(x_fused)
        return x_fused

class GTCI(nn.Module):
    def __init__(self, fPlane, ratio, se_ratio):
        super(GTCI, self).__init__()
        self.forward_tci = TCI(fPlane, reverse=False, ratio=ratio, se_ratio=se_ratio)
        self.reverse_tci = TCI(fPlane, reverse=True, ratio=ratio, se_ratio=se_ratio)

    def forward(self, x):
        x1 = self.forward_tci(x)
        x2 = self.reverse_tci(x1)
        return x2

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2D(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 扩展比为4的逆瓶颈
        self.act = nn.GELU()  # 替代ReLU的激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C) -> (B,H,C,W)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # 层缩放操作
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # 残差连接与DropPath
        x = input + self.drop_path(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=False, ln=False,
                 gelu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.ln = LayerNorm2D(out_channels,data_format='channels_first') if ln else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        y = self.conv(x)
        if self.ln is not None:
            y = self.ln(y)
        if self.gelu is not None:
            y = self.gelu(y)
        return y
class ConvNeXt(nn.Module):
    def __init__(
            self,
            in_channels=3,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1,
            layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2D(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2D(dims[i], eps=1e-6, data_format='channels_first'),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            self.stages.append(
                nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value)
                      for j in range(depths[i])]
                )
            )
            cur += depths[i]
        # --------------------------------------------------------------------------------------------------
        # self.shiftvit = ShiftVitCount(is_train=True)
        self.upsample1 = upsample(768, 384)
        self.upsample3 = upsample(384, 192)

        self.MGCA1 = MGCA(384, lf=False,heads=8,reratio=1)

        self.GTCI1 = GTCI(384, ratio=0.0, se_ratio=0)
        self.GTCI2 = GTCI(384, ratio=0.0, se_ratio=0)
        self.GTCI1 = GTCI(192, ratio=0.0, se_ratio=0)
        self.GTCI1 = GTCI(192, ratio=0.0, se_ratio=0)

        self.CMR1 = CMR(192,re=8)
        self.CMR2 = CMR(192,re=8)
        k_size = 3
        # self.SFETFI = SFEandTFI()
        self.decoder1 = nn.Sequential(
            ConvBlock(192, 96,kernel_size=k_size),
            ConvBlock(96, 48,kernel_size=k_size),
            ConvBlock(48, 1,kernel_size=k_size),
        )
        self.apply(self._init_weights)
        # self.tfi_proj = nn.Conv2d(384,192,1)
        self.proj1 = nn.Conv2d(384, 192, 1)
        self.act = nn.GELU()
        self.a = nn.Parameter(torch.tensor(0.3))  # 初始为 0.5，可自动调整比例

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, LayerNorm2D)):

            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def CMR(self, x, y_prior, MGCA, tpf):
        x1 = torch.cat([x, y_prior], 0).unsqueeze(0)
        x, y = MGCA(x1)
        y = tpf(y, y_prior)
        return x, y

    def forward(self, x):
        # ----------------- ConvNeXt Backbone -----------------
        x0 = self.stages[0](self.downsample_layers[0](x))
        x1 = self.stages[1](self.downsample_layers[1](x0))
        x2 = self.stages[2](self.downsample_layers[2](x1))
        x3 = self.stages[3](self.downsample_layers[3](x2))
        # ----------------- Decoder Stage 2 -----------------
        z0 = self.upsample1(x3)
        y1 = self.MGCA1(z0)
        z1 = self.GTCI1(z0)
        z2 = self.GTCI2(z1)
        # ----------------- Decoder Stage 2 -----------------
        z3 = self.upsample3(z2)
        y2 = self.upsample3(y1)
        z4 = self.GTCI3(z3)
        z5 = self.GTCI4(z4)
        y3 = self.CMR1(y2,z4)
        y4 = self.CMR2(y3,z5)
        # ----------------- Final Projections and Decoding -----------------
        z_final = torch.cat([z5,y4],dim=1)
        out = self.act(self.proj1(z_final))
        out1 = self.decoder1(out)
        return out1

def DTGF(pretrained=True, in_22k=True, custom_ckpt_path=None, **kwargs):
    model = ConvNeXt(**kwargs)
    if pretrained:
        if custom_ckpt_path:
            ckpt = torch.load(custom_ckpt_path, map_location='cpu', weights_only=True)
            state_dict = ckpt.get('model', ckpt)
            convnext_state_dict = {
                k: v for k, v in state_dict.items()
                if k.startswith(('downsample_layers.', 'stages.')) and
                   v.shape == model.state_dict()[k].shape
            }
            model.load_state_dict(convnext_state_dict, strict=False)
        else:
            url_key = 'convnext_tiny_22k' if in_22k else 'convnext_tiny_1k'
            ckpt = torch.hub.load_state_dict_from_url(model_urls[url_key], map_location='cpu')
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    return model
