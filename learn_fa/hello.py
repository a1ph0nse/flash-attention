import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import math
import sys
# sys.path.append("/home/chenjw/Learn/fa/naive-fa/src/")
# import naive_fa2
sys.path.append("/home/chenjw/Learn/fa/flash-attention/")
import flash_attn_2_cuda

def attention_pytorch(q, k, v):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, d = q.shape
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum('bhts,bshd->bthd', attention , v)
    return output.to(dtype=q.dtype)

def pytorch_lse(q, k, v):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, d = q.shape
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    scores = torch.exp(scores)
    max_scores = torch.max(scores, dim=-1).values
    sum_scores = torch.sum(scores, dim=-1)
    result = max_scores + torch.log(sum_scores)
    return result
    
# naive_fa2.hello()

batch_size = 32
seqlen = 512
nheads = 32
d = 64

device = 'cuda'
dtype = torch.float16

q = torch.randn(
    batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
)

k = torch.randn(
    batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
)

v = torch.randn(
    batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
)

kblockM = 128
kblockN = 128

# is_even_MN = seqlen % kblockN == 0 and seqlen % kblockM == 0
# is_even_K = d % 32 == 0

# assert(is_even_MN and is_even_K)

o, lse, _, _  = flash_attn_2_cuda.fwd(q, k, v, None, None, 0.0, 1.0 / math.sqrt(d), False, -1, -1, 0.0, False, None)

# print(f"is_even_MN: {is_even_MN}, is_even_K: {is_even_K}")

# print(q.shape)
# print(k.shape)
# print(v.shape)
print(o.shape)
print(lse.shape)

# print(f"smem per block: {smem // 1024}")

# dout = torch.randn(
#     batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
# )

# dq, dk, dv, dsoftmax = naive_fa2.flash_bwd(dout, q, k, v, o, lse, 
#                                            None, None, None, 1.0 / math.sqrt(d), True)

# print(dq.shape)
# print(dk.shape)
# print(dv.shape)
# print(dsoftmax.shape)


# from flash_attn import flash_attn_func


# o_fa, lse_fa, _ = flash_attn_func(q=q, k=k, v=v,return_attn_probs=True)

# o_pytorch = attention_pytorch(q, k, v)

# print(f"scale %f" % (1.0 / math.sqrt(d)))
# print(f"scale %f" % (q.shape[-1] ** (-0.5)))

# print(f"o: %f" % o.flatten()[1])
# print(f"o_fa: %f" % o_fa.flatten()[1])
# print(f"o_pt: %f" % o_pytorch.flatten()[1])

# error_o = torch.max(torch.abs(o - o_fa))
# print(f"Error max between o and o_fa: {error_o.item()}")

# error_o2 = torch.max(torch.abs(o_fa - o_pytorch))
# print(f"Error max between o_fa and o_pt: {error_o2.item()}")

# error_o3 = torch.max(torch.abs(o - o_pytorch))
# print(f"Error max between o and o_pt: {error_o3.item()}")

# print(f"lse: %f" % lse.flatten()[1])
# print(f"lse_fa: %f" % lse_fa.flatten()[1])

# error_lse = torch.max(torch.abs(lse - lse_fa))
# print(f"Error max between lse and lse_fa: {error_lse.item()}")