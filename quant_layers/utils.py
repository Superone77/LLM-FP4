import torch


def fp4_121_positive(x:torch.Tensor, stochastic_rounding:bool=False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    
    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def ue5m3(x:torch.Tensor) -> torch.Tensor:
    # NOTE: Assume that array values are in [0, 114688]. (14*2**13 = 114688)
    mask = x <= 2**(-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask

    x_1 = torch.round(x_1 / 2**(-17)) * (2**(-17))

    e = torch.floor(torch.log2(x_2)) - 3
    s = 2**e
    x_2 = torch.round(x_2 / s) * s

    return x_1 * mask + x_2 * (~mask)


FP8_E4M3_MAX = 240.0
def fp4_121_scaled(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scale_format:str='e8m0',
                   scaling_factor = None) -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_format == 'e8m0':
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
    
    elif scale_format == 'e4m3':
        if scaling_factor == None:
            nvfp4_max = fp4_121_max * FP8_E4M3_MAX
            scale_per_t = x_abs.max() / nvfp4_max
            x_abs_scaled = x_abs / scale_per_t

            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            input_tensor = fp4_121_max / scale_per_b
            down_cast = input_tensor.to(torch.float8_e4m3fn)
            # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
            up_cast = down_cast.to(scale_per_b.dtype)
            scale_per_b = up_cast
            scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
        else:
            scale_per_b = scaling_factor
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t
    
    elif scale_format == 'ue5m3':
        UE5M3_MAX = 114688.0
        nvfp4_max = fp4_121_max * UE5M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]

        scale_per_b = ue5m3(fp4_121_max / scale_per_b)
        
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t

    
    else: # scale_format == 'bf16'
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return sign * x_fp4_abs


def fake_quant_fp4(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   dim:int=-1, 
                   format:str='fp4_e2m1',
                   block_size:int=32, 
                   scale_format:str='e8m0',
                   grid:bool=False) -> torch.Tensor:
    # TODO:
    # 1) enable dim
    # 2) enable e3m0
    shape = x.shape
    if grid:
        assert len(shape) == 2, 'grid enabled for 2d tensors only'
        x = x.reshape(shape[0] // block_size, block_size, shape[1] // block_size, block_size).permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    else:
        x = x.reshape(-1, block_size)
    
    x = fp4_121_scaled(x, stochastic_rounding, scale_format)
    
    if grid:
        x = x.reshape(shape[0] // block_size, shape[1] // block_size, block_size, block_size).permute(0, 2, 1, 3).reshape(shape)
    else:
        x = x.reshape(shape)
    
    return x


def update_scale(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_format: str = 'e8m0',
                scaling_factor = None):
    
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    
    if scaling_factor is None:
        if scale_format == 'e8m0':
            scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
        elif scale_format == 'e4m3':
            nvfp4_max = fp4_121_max * FP8_E4M3_MAX
            scale_per_t = x_abs.max() / nvfp4_max
            x_abs_scaled = x_abs / scale_per_t
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            input_tensor = fp4_121_max / scale_per_b
            down_cast = input_tensor.to(torch.float8_e4m3fn)
            up_cast = down_cast.to(scale_per_b.dtype)
            scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), up_cast, 1.0)
            scale = scale_per_b
        
        elif scale_format == 'ue5m3':
            UE5M3_MAX = 114688.0
            nvfp4_max = fp4_121_max * UE5M3_MAX
            scale_per_t = x_abs.max() / nvfp4_max
            x_abs_scaled = x_abs / scale_per_t
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            scale_per_b = ue5m3(fp4_121_max / scale_per_b)  # 假设 ue5m3 是一个已定义的函数
            scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
            scale = scale_per_b
        else:  # scale_format == 'bf16'
            scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]
    else:
        # 如果 scaling_factor 存在，则使用它和计算出来的 scale 比较取较大者
        # 先按原始逻辑算出 scale
        if scale_format == 'e8m0':
            base_scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
        elif scale_format == 'e4m3':
            nvfp4_max = fp4_121_max * FP8_E4M3_MAX
            scale_per_t = x_abs.max() / nvfp4_max
            x_abs_scaled = x_abs / scale_per_t
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            input_tensor = fp4_121_max / scale_per_b
            down_cast = input_tensor.to(torch.float8_e4m3fn)
            up_cast = down_cast.to(scale_per_b.dtype)
            base_scale = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), up_cast, 1.0)
        
        elif scale_format == 'ue5m3':
            UE5M3_MAX = 114688.0
            nvfp4_max = fp4_121_max * UE5M3_MAX
            scale_per_t = x_abs.max() / nvfp4_max
            x_abs_scaled = x_abs / scale_per_t
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            scale_per_b = ue5m3(fp4_121_max / scale_per_b)
            base_scale = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
        else:  # bf16
            base_scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]

        # 取 base_scale 和 scaling_factor 中较大的那个
        scale = torch.maximum(base_scale, scaling_factor)

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    return scale
    

if __name__ == '__main__':

    device = torch.device('hpu')

    t = torch.randn([2, 32]).to(device)
    t_q = fake_quant_fp4(t, stochastic_rounding=True)
    print(t_q)
