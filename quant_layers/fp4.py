import torch
import os

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

FP8_E4M3_MAX = 448.0
def quant_nvfp4_0(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_per_t = None,
                scale_per_b = None):
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    
    if scale_per_t == None:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
    scale_per_t = 1
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

def quant_nvfp4(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_per_t = None,
                scale_per_b = None,
                batch_size = 1,
                vari_length = False):
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_per_t == None:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
    quant_mode = os.environ['QUANT_MODE']
    if quant_mode == "Dynamic_Block" or quant_mode == "Calib_Block":
        scale_per_t = 1
    x_abs_scaled = x_abs / scale_per_t

    if scale_per_b == None:
        if batch_size == 1:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        else:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            # scale_per_b = scale_per_b.max(dim=0, keepdim=True)[0]
        input_tensor = fp4_121_max / scale_per_b
        down_cast = input_tensor.to(torch.float8_e4m3fn)
        # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
    if vari_length:
        token_length = x_abs_scaled.shape[1]
        # print(x_abs_scaled.shape, scale_per_b[:,:token_length,:,:].shape)
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b[:,:token_length,:,:], stochastic_rounding) / scale_per_b[:,:token_length,:,:]
    else:
        # print(x_abs_scaled.shape, scale_per_b.shape)
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

    return sign * x_fp4_abs * scale_per_t



def update_scale_nvfp4(x: torch.Tensor, 
                        stochastic_rounding: bool = False, 
                        scale_per_t = None,
                        scale_per_b = None,
                        batch_size = 1,
                        vari_length = False):
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    nvfp4_max = fp4_121_max * FP8_E4M3_MAX
    scale_per_t_base = x_abs.max() / nvfp4_max
    if scale_per_t != None:
        scale_per_t = torch.maximum(scale_per_t, scale_per_t_base)
    else:
        scale_per_t = scale_per_t_base
    quant_mode = os.environ['QUANT_MODE']
    if quant_mode == "Calib_Block":
        scale_per_t = 1
    x_abs_scaled = x_abs / scale_per_t

    if batch_size == 1:
        scale_per_b_base = x_abs_scaled.max(dim=-1, keepdim=True)[0]
    else:
        scale_per_b_base = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        # scale_per_b_base = scale_per_b_base.max(dim=0, keepdim=True)[0]
    input_tensor = fp4_121_max / scale_per_b_base
    down_cast = input_tensor.to(torch.float8_e4m3fn)
    # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b_base, 1.0, False, False, torch.float8_e4m3fn)[0]
    up_cast = down_cast.to(scale_per_b_base.dtype)
    scale_per_b_base = up_cast
    scale_per_b_base = torch.where((0 < scale_per_b_base) * (scale_per_b_base < torch.inf), scale_per_b_base, 1.0)
    if quant_mode == "Calib_Global":
        scale_per_b = None
    if scale_per_b != None:
        scale_per_b = torch.maximum(scale_per_b, scale_per_b_base)
    else:
        scale_per_b = scale_per_b_base

    
    if vari_length:
        token_length = x_abs_scaled.shape[1]
        # print(x_abs_scaled.shape, scale_per_b[:,:token_length,:,:].shape)
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b[:,:token_length,:,:], stochastic_rounding) / scale_per_b[:,:token_length,:,:]
    else:
        # print(x_abs_scaled.shape, scale_per_b.shape)
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b
    if quant_mode == "Calib_Global":
        scale_per_b = None
    return sign * x_fp4_abs * scale_per_t, scale_per_t, scale_per_b


if __name__ == "__main__":
    import torch
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    # 加载模型（不初始化权重）
    os.environ['QUANT_MODE'] = "Dynamic_Double"
    model_name = "/local/mnt/workspace/wanqi/hf_model/AI-ModelScope/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    # 查看模型结构（可选）
    print(model)
    def mse(a, b):
            return F.mse_loss(a, b)
    LOG_EPS = 1e-12

    def sqnr(x, x_hat):
        mse = (x - x_hat).pow(2).mean()
        power = x.pow(2).mean()
        return 10 * torch.log10(power / (mse + LOG_EPS))
    # 获取第 12 层的 self attention 的 q_proj 权重
    # for layer in model.model.layers:
    

    #     # 打印这一层的所有参数名（可选）
    #     # for name, param in layer.named_parameters():
    #     #     print(name)

    #     # 获取 q_proj 的 weight tensor
    #     q_proj_weight = layer.self_attn.q_proj.weight

    #     # 输出形状
    #     print("Shape of q_proj weight:", q_proj_weight.shape)

    mse_0 = []
    mse_1 = []
    sqnr_0 = []
    sqnr_1 = []
    layer_names = []  # To store layer names for x-axis labels (optional)

    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_names.append(name)
            print(f"Processing Linear layer: {name}")
            weight = module.weight

            # print("Weight shape:", weight.shape)
            # 生成一个形状为 (3, 4) 的张量，服从标准正态分布 N(0, 1)
            
            # tensor = torch.randn(128, 16)
            tensor = weight.reshape(-1, 16)
            new_tensor = quant_nvfp4_0(tensor)
            mse_b = mse(tensor, new_tensor)
            mse_0.append(mse_b.detach())
            sqnr_b = sqnr(tensor, new_tensor)
            sqnr_0.append(sqnr_b.detach())
            
            
            new_tensor = quant_nvfp4(tensor)
            mse_sb = mse(tensor, new_tensor)
            mse_1.append(mse_sb.detach())
            sqnr_sb = sqnr(tensor, new_tensor)
            sqnr_1.append(sqnr_sb.detach())
            print("===============================")
            print(f"block-wise mse: {mse_b}")
            print(f"global + block-wise mse: {mse_sb}")
            print(f"block-wise sqnr: {sqnr_b}")
            print(f"global + block-wise sqnr: {sqnr_sb}")

    # 计算均值
    mse_0_mean = torch.mean(torch.tensor(mse_0))
    mse_1_mean = torch.mean(torch.tensor(mse_1))
    print("===============Summary===============")
    print(f"block-wise mse: {mse_0_mean.item()}")
    print(f"global + block-wise mse: {mse_1_mean.item()}")

    sqnr_0_mean = torch.mean(torch.tensor(sqnr_0))
    sqnr_1_mean = torch.mean(torch.tensor(sqnr_1))
    print("===============Summary===============")
    print(f"block-wise sqnr: {sqnr_0_mean.item()}")
    print(f"global + block-wise sqnr: {sqnr_1_mean.item()}")

    # Plotting
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(25, 6))
    plt.plot(layer_names, mse_0, label='Block-wise MSE', marker='o', markersize=4)
    plt.plot(layer_names, mse_1, label='Global + Block-wise MSE', marker='s', markersize=4)

    # 设置 x 轴刻度：每隔一个显示一个
    plt.xticks(ticks=layer_names[::4], labels=[str(i) for i in layer_names[::4]], fontsize=8)

    plt.xlabel('Layer Index')
    plt.ylabel('MSE')
    plt.title('MSE per Linear Layer')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("mse_per_layer.png")

    plt.figure(figsize=(25, 6))
    plt.plot(layer_names, sqnr_0, label='Block-wise SQNR', marker='o', markersize=4)
    plt.plot(layer_names, sqnr_1, label='Global + Block-wise SQNR', marker='s', markersize=4)

    # 设置 x 轴刻度：每隔一个显示一个
    plt.xticks(ticks=layer_names[::4], labels=[str(i) for i in layer_names[::4]], fontsize=8)

    plt.xlabel('Layer Index')
    plt.ylabel('SQNR')
    plt.title('SQNR per Linear Layer')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("sqnr_per_layer.png")
        

        
    
        
        
    