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

FP8_E4M3_MAX = 240.0
def quant_nvfp4(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_per_t = None,
                scale_per_b = None):
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_per_t == None:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
    
    x_abs_scaled = x_abs / scale_per_t

    if scale_per_b == None:
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        input_tensor = fp4_121_max / scale_per_b
        down_cast = input_tensor.to(torch.float8_e4m3fn)
        # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
    
    x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

    return sign * x_fp4_abs * scale_per_t

def update_scale_nvfp4(x: torch.Tensor, 
                        stochastic_rounding: bool = False, 
                        scale_per_t = None,
                        scale_per_b = None):
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    nvfp4_max = fp4_121_max * FP8_E4M3_MAX
    scale_per_t_base = x_abs.max() / nvfp4_max
    if scale_per_t != None:
        scale_per_t = torch.maximum(scale_per_t, scale_per_t_base)
    
    x_abs_scaled = x_abs / scale_per_t

    scale_per_base = x_abs_scaled.max(dim=-1, keepdim=True)[0]
    input_tensor = fp4_121_max / scale_per_base
    down_cast = input_tensor.to(torch.float8_e4m3fn)
    # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_base, 1.0, False, False, torch.float8_e4m3fn)[0]
    up_cast = down_cast.to(scale_per_base.dtype)
    scale_per_base = up_cast
    scale_per_base = torch.where((0 < scale_per_base) * (scale_per_base < torch.inf), scale_per_base, 1.0)
    if scale_per_b != None:
        scale_per_b = torch.maximum(scale_per_b, scale_per_b_base)

    
    x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

    return sign * x_fp4_abs * scale_per_t, scale_per_t, scale_per_b