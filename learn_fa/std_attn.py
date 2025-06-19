import math
import torch


def multi_head_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # 为了方便计算，转换为：batch size, head num. seq len, head dim
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    d_factor = 1.0 / math.sqrt(q.shape[-1])

    # S = QK^T/\sqrt(d)
    # (batch size, head num, seq len, seq len)
    s = torch.matmul(q, k.transpose(2, 3)) * d_factor
    # P = Softmax(S)
    # (batch size, head num, seq len, seq len)
    p = torch.softmax(s, dim=-1)
    # O = PV
    o = torch.matmul(p, v)
    return o.to(dtype=q.dtype)

batch_size = 8
seqlen = 2048
nheads = 32
d = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

# BSD: batch size, sequence length, dim(head num, head dim)
Q = torch.randn(batch_size, seqlen, nheads, d, dtype=dtype, requires_grad=True).to(device=device)
K = torch.randn(batch_size, seqlen, nheads, d, dtype=dtype, requires_grad=True).to(device=device)
V = torch.randn(batch_size, seqlen, nheads, d, dtype=dtype, requires_grad=True).to(device=device)

Q_fa = Q.clone().detach().requires_grad_(True)
K_fa = K.clone().detach().requires_grad_(True)
V_fa = V.clone().detach().requires_grad_(True)


res = multi_head_attn(Q, K, V)
# print("my std attn:")
# print(res.flatten())



# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# out = flash_attn_func(Q_fa, K_fa, V_fa)
# print("fa:")
# print(out.flatten())