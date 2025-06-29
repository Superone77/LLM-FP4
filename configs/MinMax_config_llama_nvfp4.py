from quant_layers.fp_linear import FPMinMaxBlockQuantLinear_FixedFormat
from quant_layers.fp_embed import FPMinMaxBlockQuantEmbedding_FixedFormat

bit = 8
exp_bit = 4
embed_name_list = ["qembedding"]
fc_name_list = [ "qlinear_query", "qlinear_key", "qlinear_value", "qlinear_o","qlinear_gate","qlinear_down","qlinear_up","qlinear_score"]
matmul_name_list = [ "qmatmul_qk", "qmatmul_scorev"]
w_bit = {name: bit for name in fc_name_list}
a_bit = {name: bit for name in fc_name_list}
embed_bit = {name: bit for name in embed_name_list}
A_bit = {name: bit for name in matmul_name_list}
B_bit = {name: bit for name in matmul_name_list}
w_exp_bit = {name: exp_bit for name in fc_name_list}
a_exp_bit = {name: exp_bit for name in fc_name_list}
embed_exp_bit = {name: exp_bit for name in embed_name_list}
A_exp_bit = {name: exp_bit for name in matmul_name_list}
B_exp_bit = {name: exp_bit for name in matmul_name_list}



def get_module(module_type, *args, **kwargs):

    if "embedding" in module_type:
        module= FPMinMaxBlockQuantEmbedding_FixedFormat(*args,**kwargs,padding_idx=0)


    elif "qlinear" in module_type:
        if module_type == "qlinear_score":
            module= FPMinMaxBlockQuantLinear_FixedFormat(*args,**kwargs)
        else:
            module= FPMinMaxBlockQuantLinear_FixedFormat(*args,**kwargs)
            
         
    return module