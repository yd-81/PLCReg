import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint        
from timm.layers import DropPath, to_3tuple, trunc_normal_
from einops import rearrange
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, H, W, L, -1)
    return x



class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  


        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t])) 
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape  
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv2[0]
        k, v = qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PolaLinearAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kernel_size=5, alpha=4):

        super().__init__()
        self.dim = dim    
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim        

        
        self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)              
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)             

        
        self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=(kernel_size, kernel_size, kernel_size),
                             groups=head_dim, padding=(kernel_size // 2,)*3)
        
        
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1] * window_size[2], dim)))


    def forward(self, x, y, mask=None):
        
        B, N, C = x.shape
        qkv = self.qkvg(x).reshape(B, N, 4, C).permute(2, 0, 1, 3)
        _, k, v, g = qkv.unbind(0)

        
        qkv2 = self.qkvg(y).reshape(B, N, 4, C).permute(2, 0, 1, 3)
        q, _, _, _ = qkv2.unbind(0)

        
        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        
        q = q / scale
        k = k / scale
        
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        
        q_pos = kernel_function(q) ** power 
        q_neg = kernel_function(-q) ** power 
        k_pos = kernel_function(k) ** power 
        k_neg = kernel_function(-k) ** power 

        
        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)
        k = torch.cat([k_pos, k_neg],dim=-1)     

        
        v1,v2 = torch.chunk(v,2,dim=-1)
        
        
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v1 * (N ** -0.5))
        x_sim = q_sim @ kv * z
       
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v2 * (N ** -0.5))
        x_opp = q_opp @ kv * z
        
        
        x = torch.cat([x_sim, x_opp],dim=-1)
        x = x.transpose(1,2).reshape(B, N, C)
        
       
        Wh, Ww, Wt = self.window_size
        v = v.reshape(
            B * self.num_heads,    
            Wh, Ww, Wt,           
            self.head_dim          
        ).permute(0, 4, 1, 2, 3)   
        v = self.dwc(v)          
       
        v = v.permute(0, 2, 3, 4, 1).reshape(
            B, N, self.num_heads * self.head_dim
        )
        
        x = x + v
        x = x * g  

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    def eval(self):
        super().eval()
        print('eval')

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(7,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_diy=96,
                 alpha=4, kernel_size=5, attn_type='L', attn_mix=None):
        super().__init__()
        dim = dim_diy
        self.dim = dim
        self.input_resolution = input_resolution  
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim_diy)

        if attn_mix is None:
            
            assert attn_type in ['L', 'S']
            if attn_type == 'L':
                self.attn = PolaLinearAttention(
                    dim, window_size=self.window_size, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    alpha=alpha, kernel_size=kernel_size)
            else:
                self.attn = WindowAttention(
                    dim, window_size=self.window_size, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            if attn_mix == 'P':
                self.attn = PolaLinearAttention(
                    dim, window_size=self.window_size, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    alpha=alpha, kernel_size=kernel_size)
            elif attn_mix == 'W':
                self.attn = WindowAttention(
                    dim, window_size=self.window_size, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            else:
                raise ValueError(f"Unknown attn_kind: {attn_mix}. Use 'P' or 'W'.")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_diy)
        mlp_hidden_dim = int(dim_diy * mlp_ratio)
        self.mlp = Mlp(in_features=dim_diy, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y, mask_matrix):
        H, W, T = self.input_resolution
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        y = self.norm1(y)
        x = x.view(B, H, W, T, C)
        y = y.view(B, H, W, T, C)

        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_y = torch.roll(y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_y = y
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  
        y_windows = window_partition(shifted_y, self.window_size)  
        y_windows = y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  

        attn_windows = self.attn(x_windows, y_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  
        # shifted_y = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp) 

        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 alpha=4, kernel_size=5, attn_mix=None, attn_type='L', pat_merg_rf=2, dim_diy=96):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        window_size = to_3tuple(window_size)

        def parse_attn_types(attn_type, depth):
            if isinstance(attn_type, str):
                s = attn_type.strip().upper()
                if s.startswith('M') and s[1:].isdigit():
                    m = int(s[1:])
                    return ['L' if i < m else 'S' for i in range(depth)]
                if len(s) == 1:
                    return [s] * depth
                out = list(s)
                if len(out) < depth:
                    out += [out[-1]] * (depth - len(out))
                return out[:depth]
            return ['L'] * depth

        attn_types = parse_attn_types(attn_type, depth)

        short_side = min(self.input_resolution) if isinstance(self.input_resolution, (tuple, list)) else 56
        swa_win = (7, 7, 7) if short_side <= 56 else (12, 12, 12)

        self.block_window_sizes = [
            window_size if attn_types[i] == 'L' else swa_win for i in range(depth)
        ]
        self.block_shift_sizes = [
            (0, 0, 0) if (i % 2 == 0) else
            (self.block_window_sizes[i][0] // 2,
             self.block_window_sizes[i][1] // 2,
             self.block_window_sizes[i][2] // 2)
            for i in range(depth)
        ]

        if isinstance(drop_path, list):
            assert len(drop_path) == depth
            dpr = drop_path
        else:
            dpr = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=self.block_window_sizes[i],
                shift_size=self.block_shift_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                dim_diy=dim_diy,
                alpha=alpha,
                kernel_size=kernel_size,
                attn_type=attn_types[i],
                attn_mix=attn_mix,
            )
            for i in range(depth)
        ])

        self.downsample = downsample

    @torch.no_grad()
    def _build_attn_mask(self, H, W, T, win_size, shift_size, device):
        
        Wh, Ww, Wt = win_size
        Sh, Sw, St = shift_size

        Hp = int(math.ceil(H / Wh)) * Wh
        Wp = int(math.ceil(W / Ww)) * Ww
        Tp = int(math.ceil(T / Wt)) * Wt

        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=device)

        h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None)) if Sh > 0 else (slice(0, Hp),)
        w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None)) if Sw > 0 else (slice(0, Wp),)
        t_slices = (slice(0, -Wt), slice(-Wt, -St), slice(-St, None)) if St > 0 else (slice(0, Tp),)

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, win_size)  
        mask_windows = mask_windows.view(-1, Wh * Ww * Wt)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, H, W, T):
        
        last_Wh, last_Ww, last_Wt = None, None, None

        for i, blk in enumerate(self.blocks):
            
            blk.H, blk.W, blk.T = H, W, T
          
            win_size  = self.block_window_sizes[i]
            shift_size = self.block_shift_sizes[i]
            attn_mask = self._build_attn_mask(H, W, T, win_size, shift_size, x.device)

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y, attn_mask)
            else:
                x = blk(x, y, attn_mask)

            last_Wh, last_Ww, last_Wt = win_size

        
        return x, H, W, T, x, last_Wh, last_Ww, last_Wt


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class PolaSwinTransformer(nn.Module):

    def __init__(self, pretrain_img_size=(160,192,224), patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(7,7,7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, spe=False, patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 dim_diy=96,
                 alpha=4, kernel_size=5,attn_mix=None, attn_type='L', **kwargs):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dim_diy = dim_diy

        if self.ape:
            patch_size = to_3tuple(patch_size)
            patches_resolution = [self.pretrain_img_size[0] // patch_size[0], self.pretrain_img_size[1] // patch_size[1], self.pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)
       
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=self.pretrain_img_size,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               dim_diy=dim_diy,
                               alpha=alpha,
                               kernel_size=kernel_size,
                               attn_type=attn_type[i_layer],
                               attn_mix=attn_mix,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = dim_diy

        for i_layer in out_indices:
            layer = norm_layer(dim_diy)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, y):
        # x = self.patch_embed(x)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
            y = y.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        y = self.pos_drop(y)

        for i in range(self.num_layers):  # num_layers  = len(depths)  = 4
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, y, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()
        return out

    def train(self, mode=True):
        super(PolaSwinTransformer, self).train(mode)
        self._freeze_stages()