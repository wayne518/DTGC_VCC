import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from thop import profile


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


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
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


class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=(1, 1, 1)):
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv3d(c1, sum(c2s), k, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


def CBFuse(xs):
    target_size = xs[-1].shape[2:]
    res = [F.interpolate(x, size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
    out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
    return out


class LayerNorm3D(nn.Module):
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
        x = x.squeeze(0).permute(1, 0, 2, 3)
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).permute(1, 0, 2, 3).unsqueeze(0)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            x = x.permute(1, 0, 2, 3).unsqueeze(0)
            return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MatAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.conv3 = nn.Conv3d(dim, dim, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x1 = self.conv1(x)
        x1_a = torch.mean(x1.reshape(B, C, T, H * W), dim=-1, keepdim=True)
        x1_b = torch.mean(x1.reshape(B, C, T, H * W), dim=-2, keepdim=True)
        out1 = (x1_a @ x1_b).reshape(B, C, T, H, W)
        x2 = self.conv2(x)
        x2_a = torch.mean(x2.reshape(B, C, T * H, W), dim=-1, keepdim=True)
        x2_b = torch.mean(x2.reshape(B, C, T * H, W), dim=-2, keepdim=True)
        out2 = (x2_a @ x2_b).reshape(B, C, T, H, W)
        x3 = self.conv3(x)
        x3_a = torch.mean(x3.permute(0, 1, 2, 4, 3).reshape(B, C, T * W, H), dim=-1, keepdim=True)
        x3_b = torch.mean(x3.permute(0, 1, 2, 4, 3).reshape(B, C, T * W, H), dim=-2, keepdim=True)
        out3 = (x3_a @ x3_b).reshape(B, C, T, W, H).permute(0, 1, 2, 4, 3)
        add = out1 * out2 * out3
        out = self.proj(add)
        return out


class Block3D(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(Block3D, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = LayerNorm3D(dim // 2, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm3D(dim, eps=1e-6, data_format='channels_first')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ma1 = MatAttention(dim // 6)
        self.ma2 = MatAttention(dim // 6)
        self.ma3 = MatAttention(dim // 6)

    def forward(self, x):
        _, _, t, _, _ = x.shape
        x_no, x_yes = torch.split(x, self.dim // 2, dim=1)
        shortcut1 = x_yes
        x_yes = self.norm1(x_yes)

        x1, x2, x3 = torch.split(x_yes, self.dim // 6, dim=1)
        x1_1, x1_2, x1_3, x1_4 = torch.split(x1, t // 4, dim=2)
        x1 = torch.cat((x1_1, x1_2, x1_3, x1_4), dim=0)
        x2_1, x2_2 = torch.split(x2, t // 2, dim=2)
        x2 = torch.cat((x2_1, x2_2), dim=0)
        ma1 = self.ma1(x1)
        ma1_1, ma1_2, ma1_3, ma1_4 = torch.split(ma1, 1, dim=0)
        ma1 = torch.cat((ma1_1, ma1_2, ma1_3, ma1_4), dim=2)
        ma2 = self.ma2(x2)
        ma2_1, ma2_2 = torch.split(ma2, 1, dim=0)
        ma2 = torch.cat((ma2_1, ma2_2), dim=2)
        ma3 = self.ma3(x3)
        ma_concat = torch.cat((ma1, ma2, ma3), dim=1)
        out_yes = shortcut1 + ma_concat
        out1 = torch.concat((x_no, out_yes), dim=1)

        shortcut2 = out1
        out2 = shortcut2 + self.drop_path(self.mlp(self.norm2(out1)))
        return out2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):

        x_gap = self.gap(x).flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)

        B1, N1, C1 = x_gap.shape
        B2, N2, C2 = y.shape
        assert B1 == B2
        assert C1 == C2

        q = self.wq(x_gap).reshape(B1, 1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(y).reshape(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(y).reshape(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_gap = (attn @ v).transpose(1, 2).reshape(B1, 1, C1)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x_gap = self.proj(x_gap)
        x_gap = self.proj_drop(x_gap)
        x_gap = self.sig(x_gap.reshape(B1, C1, 1, 1))
        x_ca = x * x_gap.expand_as(x)
        return x + x_ca


class MultiFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_0 = nn.Conv2d(in_channels, 128, 1)
        self.branch_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.branch_2 = nn.Conv2d(in_channels, 32, 5, padding=2)
        self.branch_3 = nn.Conv2d(128, 128, 1)
        self.branch_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.branch_5 = nn.Conv2d(32, 32, 5, padding=2)
        self.fuse = nn.Conv2d(224, 1, 3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_0 = self.gelu(x_0)
        x_0 = self.branch_3(x_0)
        x_0 = self.gelu(x_0)
        x_1 = self.branch_1(x)
        x_1 = self.gelu(x_1)
        x_1 = self.branch_4(x_1)
        x_1 = self.gelu(x_1)
        x_2 = self.branch_2(x)
        x_2 = self.gelu(x_2)
        x_2 = self.branch_5(x_2)
        x_2 = self.gelu(x_2)
        out = torch.cat((x_0, x_1, x_2), dim=1)
        out = self.fuse(out)
        return out


class SingleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 32, 3, padding=2, dilation=2)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.conv5 = nn.Conv2d(32, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        x = self.conv4(x)
        x = self.gelu(x)
        out = self.conv5(x)
        return out


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], depths_3d=[2, 2, 2, 2], drop_path_rate=0.,
                 layer_scale_init_value=1e-6,):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        '---------------------------------------------------------------------------------------------'

        self.downsample_3d = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_3d = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            LayerNorm3D(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_3d.append(stem_3d)
        for i in range(3):
            downsample_3d = nn.Sequential(
                LayerNorm3D(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            )
            self.downsample_3d.append(downsample_3d)

        self.stages_3d = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stage_3d = nn.Sequential(
                *[Block3D(dim=dims[i]) for _ in range(depths_3d[i])]
            )
            self.stages_3d.append(stage_3d)

        self.integrated_layer_2d_0 = nn.Sequential(
            LayerNorm(dims[3], eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(in_channels=dims[3], out_channels=dims[3] // 4, kernel_size=2, stride=2),
        )

        self.integrated_layer_2d_1 = nn.Sequential(
            LayerNorm(dims[3] // 4, eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(in_channels=dims[3] // 4, out_channels=dims[3] // 16, kernel_size=2, stride=2),
        )

        self.integrated_layer_2d_2 = nn.Conv2d(in_channels=dims[3] // 16, out_channels=1, kernel_size=1)

        self.integrated_layer_3d_0 = nn.Sequential(
            LayerNorm3D(dims[3], eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose3d(in_channels=dims[3], out_channels=dims[3] // 4, kernel_size=(1, 2, 2),
                               stride=(1, 2, 2)),
        )

        self.integrated_layer_3d_1 = nn.Sequential(
            LayerNorm3D(dims[3] // 4, eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose3d(in_channels=dims[3] // 4, out_channels=dims[3] // 16, kernel_size=(1, 2, 2),
                               stride=(1, 2, 2)),
        )

        self.integrated_layer_3d_2 = nn.Conv3d(in_channels=dims[3] // 16, out_channels=1, kernel_size=1)

        self.CBLinear_0 = CBLinear(dims[0], [dims[0], dims[1], dims[2], dims[3]])
        self.CBLinear_1 = CBLinear(dims[1], [dims[0], dims[1], dims[2], dims[3]])
        self.CBLinear_2 = CBLinear(dims[2], [dims[0], dims[1], dims[2], dims[3]])
        self.CBLinear_3 = CBLinear(dims[3], [dims[0], dims[1], dims[2], dims[3]])

        self.cross_2d_0 = CrossAttention(dims[3])
        self.cross_2d_1 = CrossAttention(dims[3] // 4)
        self.cross_2d_2 = CrossAttention(dims[3] // 16)
        self.cross_3d_0 = CrossAttention(dims[3])
        self.cross_3d_1 = CrossAttention(dims[3] // 4)
        self.cross_3d_2 = CrossAttention(dims[3] // 16)

        self.gelu = nn.GELU()

        # self.mfusion = MultiFusion(2)
        self.sfusion = SingleFusion(2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_0 = self.stages[0](self.downsample_layers[0](x))
        y_0 = x_0.permute(1, 0, 2, 3).unsqueeze(0)
        x_1 = self.stages[1](self.downsample_layers[1](x_0))
        y_1 = x_1.permute(1, 0, 2, 3).unsqueeze(0)
        x_2 = self.stages[2](self.downsample_layers[2](x_1))
        y_2 = x_2.permute(1, 0, 2, 3).unsqueeze(0)
        x_3 = self.stages[3](self.downsample_layers[3](x_2))
        y_3 = x_3.permute(1, 0, 2, 3).unsqueeze(0)

        z_0 = self.CBLinear_0(y_0)
        z_1 = self.CBLinear_1(y_1)
        z_2 = self.CBLinear_2(y_2)
        z_3 = self.CBLinear_3(y_3)

        q_0 = self.downsample_3d[0](x.permute(1, 0, 2, 3).unsqueeze(0))
        fuse_0 = CBFuse([z_0[0], z_1[0], z_2[0], z_3[0], q_0])
        q_1 = self.stages_3d[0](fuse_0)
        q_2 = self.downsample_3d[1](q_1)
        fuse_1 = CBFuse([z_0[1], z_1[1], z_2[1], z_3[1], q_2])
        q_3 = self.stages_3d[1](fuse_1)
        q_4 = self.downsample_3d[2](q_3)
        fuse_2 = CBFuse([z_0[2], z_1[2], z_2[2], z_3[2], q_4])
        q_5 = self.stages_3d[2](fuse_2)
        q_6 = self.downsample_3d[3](q_5)
        fuse_3 = CBFuse([z_0[3], z_1[3], z_2[3], z_3[3], q_6])
        q_7 = self.stages_3d[3](fuse_3)

        x_4 = self.cross_2d_0(x_3, q_7.squeeze(0).permute(1, 0, 2, 3))
        q_8 = self.cross_3d_0(q_7.squeeze(0).permute(1, 0, 2, 3), x_3)
        x_5 = self.integrated_layer_2d_0(x_4)
        q_9 = self.integrated_layer_3d_0(q_8.permute(1, 0, 2, 3).unsqueeze(0))
        x_6 = self.gelu(self.cross_2d_1(x_5, q_9.squeeze(0).permute(1, 0, 2, 3)))
        q_10 = self.gelu(self.cross_3d_1(q_9.squeeze(0).permute(1, 0, 2, 3), x_5))
        x_7 = self.integrated_layer_2d_1(x_6)
        q_11 = self.integrated_layer_3d_1(q_10.permute(1, 0, 2, 3).unsqueeze(0))
        x_8 = self.gelu(self.cross_2d_2(x_7, q_11.squeeze(0).permute(1, 0, 2, 3)))
        q_12 = self.gelu(self.cross_3d_2(q_11.squeeze(0).permute(1, 0, 2, 3), x_7))

        output_1 = self.integrated_layer_2d_2(x_8)
        output_2 = self.integrated_layer_3d_2(q_12.permute(1, 0, 2, 3).unsqueeze(0))
        output_2 = output_2.squeeze(0).permute(1, 0, 2, 3)
        output_3 = torch.cat((output_1, output_2), dim=1)
        output_3 = self.sfusion(output_3)

        return output_1, output_2, output_3


def dsti(pretrained=True, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], depths_3d=[2, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

