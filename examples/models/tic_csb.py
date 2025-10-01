import math
from click import prompt
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from layers import RSTB
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from numpy import ceil

from compressai.models.utils import conv, deconv, update_registered_buffers

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
from torch import Tensor

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

SCALES_LEVELS = 64



def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class SPA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm2 = nn.BatchNorm2d(dim)
        # 简单像素注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 深度可分离卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 像素注意力
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积，降维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积，输出单通道
            nn.Sigmoid()  # Sigmoid激活函数
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，升维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，降维
        )

    def forward(self, x):
        identity = x  # 保存输入以便残差连接
        x = self.norm2(x)  # 批归一化
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)  # 拼接不同注意力机制的输出
        x = self.mlp2(x)  # 通过MLP层
        x = identity + x  # 残差连接
        return x
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim=1, mlp_ratio=4.0):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim, int(mlp_ratio * dim))  # expand channels
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(int(mlp_ratio * dim), dim)  # project back

    def forward(self, x):
        shortcut = x  # [B, C, H, W]

        x = self.dw_conv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C] for LayerNorm

        x = self.norm(x)  # LayerNorm on C
        x = self.pw_conv1(x)  # -> [B, H, W, 4C]
        x = self.act(x)
        x = self.pw_conv2(x)  # -> [B, H, W, C]

        x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        return x + shortcut  # residual add

class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        if p is None:
            # 如果卷积核大小为整数，则使用卷积核的一半作为自动填充值；
            # 若卷积核大小为列表，则对每个卷积核尺寸计算其一半作为对应的填充值。
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)  # 初始化二维卷积层
        self.bn = nn.BatchNorm2d(c2)  # 初始化批归一化层
        self.act = nn.SiLU()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.bn(self.conv(x))              # V1 对输入张量应用卷积、批归一化
        # return self.act(self.bn(self.conv(x)))  # V2 对输入张量应用卷积、批归一化以及激活函数



class CSB(nn.Module):
    def __init__(self, in_dim=128, middle_dim=64, adapt_factor=1):
        super().__init__()
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, kernel_size=1)
        self.cheb_filter = ChannelwiseLearnableChebFilter(middle_dim, K=20)
        self.factor_h = nn.Parameter(torch.ones(middle_dim))  # Learnable gate
        self.spa = SPA(middle_dim)
        self.s_up = nn.Conv2d(middle_dim, in_dim, kernel_size=1)
        self.adapt_factor = adapt_factor

    def forward(self, x):
        B, C, H, W = x.shape
        x_1 = self.s_down1(x)
        x_filtered = self.cheb_filter(x_1)
        factor = torch.clamp(self.factor_h, min=0.0, max=3.0).view(1, -1, 1, 1)
        res_1 = (x_1 - x_filtered) * factor
        x_1 = F.relu(x_1 + self.adapt_factor * res_1)

        x_1 = self.spa(x_1)
        x_1 = self.s_up(x_1)

        return x + x_1
class ChannelwiseLearnableChebFilter(nn.Module):
    def __init__(self, dim, K=20):
        super().__init__()
        self.dim = dim
        self.K = K
        self.theta = nn.Parameter(torch.empty(dim, K + 1))
        nn.init.uniform_(self.theta, a=-0.1, b=0.1)  # 稳定初始化

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = torch.clamp(x * 2 - 1, -1.0, 1.0)  # 保证输入在 [-1, 1]

        T = [torch.ones_like(x_norm), x_norm]
        for k in range(2, self.K + 1):
            T_k = 2 * x_norm * T[-1] - T[-2]
            T.append(T_k)

        T_stack = torch.stack(T, dim=0)  # [K+1, B, C, H, W]

        # theta: [C, K+1] → [K+1, 1, C, 1, 1]
        theta = self.theta.permute(1, 0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # 广播乘法
        filtered = (T_stack * theta).sum(dim=0)  # [B, C, H, W]

        return filtered


class TIC_CSB(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False



        self.savepath = "/home/mzc/Experiments/AdpatICMH/ECCV2024-AdpatICMH-main/examples/test/featureMap"
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_CSB = nn.Sequential(
            CSB(N),
            CSB(N),
            CSB(N)
        )
        self.decoder_CSB  = nn.Sequential(
            CSB(N),
            CSB(N),
            CSB(N)
        )

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)

    def g_a(self, x, x_size=None):
    # def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))

      
        x = self.encoder_CSB[0](x)

        attns.append(attn)

        x = self.g_a2(x)


        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))



   
        x = self.encoder_CSB[1](x)


        attns.append(attn)


        x = self.g_a4(x)


        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))



        x = self.encoder_CSB[2](x)

        attns.append(attn)


        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns
    #
    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)

        x = self.decoder_CSB[2](x)

        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)

        x = self.decoder_CSB[1](x)

        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)

        x = self.decoder_CSB[0](x)

        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        x = self.g_s7(x)
        return x, attns

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):

        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        # print("ni hao 00000000000")
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, attns_s = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



