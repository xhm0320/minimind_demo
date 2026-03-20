from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,#句子开始token 的 ID
        eos_token_id: int = 2,#句子结束 token 的 ID
        hidden_act: str = "silu",#激活函数
        hidden_size: int = 512,#模型隐藏层维度
        intermediate_size: int = None,#FFN 层中间维度（None 时会自动计算为 hidden_size*8/3 并对齐 64 倍数）
        max_position_embeddings: int = 32768,#模型支持的最大序列长度
        num_attention_heads: int = 8,#注意力头数
        num_hidden_layers: int = 8,#Transformer 层数
        num_key_value_heads: int = 2,#K/V 头数
        vocab_size: int = 6400,#词表大小
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,#	是否启用 YaRN 缩放
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


#继承nn.Module类
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,esp:float=1e-5):
        super().__init__()
        #维度
        self.dim=dim
        self.esp=esp
        self.weight=nn.parameter(torch.ones(dim))
#_norm
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.esp)
#forward
    def forward(self,x):
        return self.weight*self._norm(x,float()).type_as(x)
  
#RoPE    
def precompute_freqs_cis(dim:int,end: int = int(32 * 1024),rope_base: float = 1e6,rope_scaling:Optional[dict]=None):
    #初始化RoPE
    #att_factor：注意力参数，温差缩放
    freqs = 1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))

    #需要使用YaRN
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",4),
            rope_scaling.get("beta_fast",4),
            rope_scaling.get("beta_slow"),
        )

        #推断的长度大于训练长度,使用缩放
        # if end > orig_max:
        #     #波长b到i的映射
        #     inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        #     #划分高低维度
        #     #low：不需要缩放高频部分
        #     #high：需要缩放的低频部分
        #     low,high = (max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))

        #     #计算缩放因子ramp
        #     #low之前，ramp为0，high之后，ramp等于1，在low和high之间，线性过渡
        #     ramp = torch.clamp(
        #         (torch.arange(dim//2,device = freqs.device).float()-low)/max(high-low,0.001),
        #     0,
        #     1,
        #     )

        #     #当ramp = 0 时（高频），系数等于1，保持频率不变
        #     #当ramp = 1 时（低频），系数等于1/factor，对频率进行线性插值缩放
        #     #当ramp在0-1之间时，平滑过渡
        #     freqs = freqs * (1-ramp+ramp/factor)
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max),dim // 2)

            power = torch.arange(0,dim//2,device=freqs.device).float()/max(dim // 2-1,1)

            beta = beta_slow +(beta_fast-beta_slow)*power

            scale = torch.where(torch.arange(dim//2,device=freqs.device)< corr_dim,(beta * factor - beta +1)/(beta * factor),1.0 / factor)

            freqs = freqs * scale
        #根据end,生成位置索引t
        t = torch.arange(end,device=freqs.device)

        #计算外积,将t和频率部分相乘，得到每一个的位置旋转角度
        freqs = torch.outer(t,freqs).float()
        freqs_cos = torch.cos(freqs).repeat_interleave(2,dim=-1)
        freqs_sin = torch.sin(freqs).repeat_interleave(2,dim=-1)

        return freqs_cos,freqs_sin    
    
#编写RoPE
def apply_rotary_pos_emb(q,k,cos,sin,position_ids = None,unsqueeze_dim = 1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x.shape[-1]:取最后一个维度的重点
        #x[...,x.shape[-1] // 2 :]：取出x的后半部分
        return torch.cat((-x[...,x.shape[-1] // 2 :],x[...,:x.shape[-1] // 2]),dim = -1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed,k_embed
        

def repeat_kv(x:torch.Tensor,n_rap:int)->torch.Tensor:#[bs, slen, num_kv_heads, head_dim]
    #n_rap：重复倍数
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rap == 1:
        #bs = 批次大小；slen = 序列长度；num_key_value_heads=KV 头数，head_dim = 单头维度
        return x
    return (x[:,:,:,None,:]#[bs, slen, num_kv_heads, 1, head_dim]
    .expand(bs,slen,num_key_value_heads,n_rap,head_dim)#[bs, slen, num_kv_heads,n_rap , head_dim]
    .reshape(bs,slen,num_key_value_heads*n_rap,head_dim)#[bs, slen, num_kv_heads*n_nap, head_dim]
    )

class Attention(nn.Module):
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        #num_attention_heads:Q头数;num_key_value_heads :kv头数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        #强制约束：Q 头数必须是 KV 头数的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # 设置注意力头配置
        self.n_local_heads = args.num_attention_heads          # query头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv头需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # 定义线性投影层 (无偏置，节省参数)
        # nn.Linear(in_features, out_features, bias=False)
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)     # Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Value投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)     # 输出投影

        # Dropout层用于正则化
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接dropout
        self.dropout = args.dropout                      # 保存dropout率

         # 检查是否支持Flash Attention
        # hasattr(obj, 'attr'): 检查对象是否有指定属性
        # Flash Attention需要PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
        # 如果不支持可以打印警告: print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
        x: torch.Tensor,#注意力层输入，[batch_size, seq_len, hidden_size]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收cos和sin
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None):#注意力掩码，[batch_size, seq_len]
        bsz, seq_len, _ = x.shape
         # 线性投影为Q,K,V
        # q_proj: hidden -> num_heads * head_dim
        # k_proj/v_proj: hidden -> num_kv_heads * head_dim (GQA情形)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 将投影结果reshape成多头格式
        # q: [bsz, seq_len, n_local_heads, head_dim]
        # k/v: [bsz, seq_len, n_local_kv_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # position_embeddings是预计算的(cos, sin)，按序列位置切片并应用RoPE
        cos, sin = position_embeddings
        # 只取当前序列长度的前缀（用于inference时从start_pos开始）
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])


        # -------------------- KV cache 处理 --------------------
        # past_key_value: (past_k, past_v) 或 None
        # 当存在past时，将past拼接到当前k,v的时间维度上，便于自回归推理
        if past_key_value is not None:
            # past_key_value[0] 的shape为 [bsz, past_seq_len, n_local_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)#past_k + 当前k
            xv = torch.cat([past_key_value[1], xv], dim=1) # past_v + 当前v

        # 如果需要缓存，返回拼接后的(k,v)，否则past_kv置为None
        past_kv = (xk, xv) if use_cache else None


        # -------------------- GQA: 对KV重复以匹配Q头 --------------------
        # transpose到形状 [bsz, n_heads, seq_len, head_dim] 以便矩阵乘法
        xq = xq.transpose(1, 2)
        # repeat_kv会把k/v的头数从 n_local_kv_heads -> n_local_kv_heads * n_rep (即等于n_local_heads)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask, # 注意力掩码（pad/因果）
                dropout_p=self.dropout if self.training else 0.0,#训练时dropout，推理时关闭
                is_causal=True  # 自回归（因果）注意力，启用因果掩码（防止看到未来token）
            )
        else:
            # 计算注意力分数：scores = Q @ K^T / sqrt(d)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # causal mask: 上三角（对角线以上）置为 -inf，防止看到未来信息
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展batch和head维度，[bsz, n_heads, seq_len, seq_len]

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9#0→-1e9，1→0
                scores = scores + extended_attention_mask

            # softmax得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 加权求和得到输出
            output = scores @ xv

        # 恢复形状并做输出投影 + 残差dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, n_heads, seq_len, head_dim] → [bsz, seq_len, hidden_size]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            #按hidden_size的8/3倍计算基础中间维度
            intermediate_size = int(config.hidden_size * 8 / 3)
            #向上取整到64的倍数（硬件对齐优化，GPU计算更高效）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)
        # gate_proj: hidden -> intermediate (用于计算gate部分)->生成 “门控开关”
        # up_proj: hidden -> intermediate (用于被gate的部分)->生成 “原始特征”
        # down_proj: intermediate -> hidden (用于投影回hidden维度)->将筛选后的高维特征还原为隐藏层维度
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # ACT2FN是transformers里激活函数的映射表，支持'silu','gelu'等
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        forward实现使用SwiGLU风格的门控激活：
        output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        并在输出前应用dropout
        """
        #计算门控值 = 激活函数(gate_proj(输入)) * up_proj(输入)
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
    
 #单个Transformer 编码器块   
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)# 初始化注意力层

        self.layer_id = layer_id#当前 Block 的编号
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)#注意力层输入前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)## 前馈层输入前的归一化
        self.mlp = FeedForward(config)#初始化门控前馈层(FFN)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 残差连接模式：先做RMSNorm -> Attention -> 残差相加 -> RMSNorm -> FFN -> 残差相加
        # 保存残差以供后续相加
        residual = hidden_states

        # 注意力子层：输入先归一化（RMSNorm），返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        # 注意力输出与残差相加
        hidden_states = hidden_states + residual

        # 前馈子层（post-attention layernorm）并相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,## 单头维度（如512/8=64）
                                                    end=config.max_position_embeddings, # 最大序列长度
                                                    rope_base=config.rope_theta, # RoPE基础频率（1000000）
                                                    rope_scaling=config.rope_scaling) # YaRN缩放配置
        # 将预计算的cos/sin注册为buffer（不参与参数更新，可随模型移到GPU）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,#输入 token ID[batch_size, seq_len]
                attention_mask: Optional[torch.Tensor] = None,#注意力掩码
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,#每层的 KV Cache 列表
                use_cache: bool = False,#是否返回新的 KV Cache（推理时设为 True
                **kwargs):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        # 兼容性检查：某些框架会传入包含.layers属性的对象，视为不携带past信息
        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        # past_key_values为每层的(past_k, past_v)列表，如果为None则创建与层数相同的None列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        # past_key_values[0] 形如 (k, v)，k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [bsz, seq_len, hidden_size]

        # 从注册的buffer中取出对应位置范围的cos/sin作为position_embeddings
        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向，通过zip把layer和对应的past_key_value配对
        presents = []# # 保存每层新生成的KV Cache（推理时返回）
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,# 上一层的输出
                position_embeddings,# 当前序列的RoPE编码
                past_key_value=past_key_value, # 该层的历史KV Cache
                use_cache=use_cache,# 是否保存新的KV Cache
                attention_mask=attention_mask # 注意力掩码
            )
            presents.append(present) # 收集该层的新KV Cache

        # 最后做归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents
    

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.model = MiniMindModel(config)
         # 输出头：将隐藏层向量映射为vocab_size维的logits（token概率得分）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享：嵌入层和输出头共享权重（LLM标配优化，减少参数+提升效果）
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,#输入 token ID：[bsz, seq_len]
        attention_mask: Optional[torch.Tensor] = None,#注意力掩码（掩蔽 pad token）
        labels: Optional[torch.Tensor] = None,#训练标签
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,#保留的 logits 范围
        **args,
    ):
        hidden_states, past_key_values= self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        #logits to keep是整数，就保留最后n个位置
        #生成的时候只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        return CausalLMOutputWithPast(
            Logits = logits,
            past_key_values = past_key_values,
            hidden_states=hidden_states)

