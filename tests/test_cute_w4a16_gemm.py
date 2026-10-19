import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_INCLUDE = os.path.expanduser(
    os.environ.get("CUTLASS_INCLUDE", "~/projects/cutlass/include")
)

load(
    name="lowbitlinear_cute_w4a16_ext",
    sources=[str(ROOT / "csrc" / "cute_w4a16_gemm.cu")],
    extra_include_paths=[CUTLASS_INCLUDE],
    extra_cflags=["-std=c++20"],
    extra_cuda_cflags=["-std=c++20", "-lineinfo"],
    is_python_module=False,
    verbose=True,
)

op = torch.ops.lowbitlinear_cute_w4a16.gemm

def pack_uint4(q: torch.Tensor) -> torch.Tensor:
    """[N,K] values 0..15 -> [N,ceil(K/8)] packed int32."""
    n, k = q.shape
    padded_k = ((k + 7) // 8) * 8
    q64 = torch.zeros((n, padded_k), device=q.device, dtype=torch.int64)
    q64[:, :k] = q.to(torch.int64)
    packed = torch.zeros((n, padded_k // 8), device=q.device, dtype=torch.int64)
    for nibble in range(8):
        packed |= q64[:, nibble::8] << (4 * nibble)
    return packed.to(torch.int32).contiguous()


def unpack_uint4(packed: torch.Tensor, k: int) -> torch.Tensor:
    n = packed.shape[0]
    words = packed.to(torch.int64)
    return torch.stack(
        [(words >> (4 * nibble)) & 0xF for nibble in range(8)], dim=-1
    ).reshape(n, packed.shape[1] * 8)[:, :k]


def check_case(label, M, N, K, *, exact=False, highbit=False):
    device = "cuda"
    X = torch.randn((M, K), device=device, dtype=torch.float16)
    q = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    if highbit:
        q[:, 7::8] = 15  # ensure packed INT32 high bit is set
    groups = (K + 63) // 64
    scales = (torch.rand((N, groups), device=device) * 0.08 + 0.01).float()
    zeros = torch.randint(0, 16, (N, groups), device=device, dtype=torch.int32)

    if exact:
        X.fill_(1)
        q.fill_(5)
        scales.fill_(0.25)
        zeros.fill_(3)

    packed = pack_uint4(q)
    unpacked = unpack_uint4(packed, K)
    torch.testing.assert_close(unpacked.to(torch.int32), q, atol=0, rtol=0)

    # Reference matches the kernel's FP32 dequant -> FP16 MMA operand policy.
    group_index = torch.arange(K, device=device) // 64
    w_half = ((unpacked.float() - zeros[:, group_index].float()) *
              scales[:, group_index]).half()
    expected = (X.float() @ w_half.float().T).half()
    actual = op(X.contiguous(), packed, scales.contiguous(), zeros.contiguous())
    torch.cuda.synchronize()
    atol = 0 if exact else 0.02
    rtol = 0 if exact else 0.01
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_err = ((actual.float() - expected.float()).abs().max().item()
               if actual.numel() else 0.0)
    print(f"{label:27s} M={M:3d} N={N:3d} K={K:3d} "
          f"PASS  max_abs_err={max_err:.6f}")

def main():
    torch.manual_seed(42)
    assert torch.cuda.is_available(), "CUDA GPU required"
    cases = [
        (32, 32, 64),   # original tile, now loop-compatible
        (1, 1, 1),      # all three dimensions smaller than a tile
        (17, 23, 65),   # M/N/K tails and 2 quant groups
        (17, 23, 129),  # exact multi-K accumulation across 3 groups
        (31, 33, 127),  # N crosses a CTA, K not multiple of 64
        (33, 37, 129),  # M/N cross CTA, third group, K tail
        (64, 65, 130),  # multiple CTAs and multiple K stages
        (79, 70, 192),  # nontrivial M/N tails, full K stages
        (4, 9, 7),      # incomplete packed word
        (32, 32, 0),    # no K: zero result
        (0, 32, 65),    # empty M
        (32, 0, 65),    # empty N
    ]
    for M, N, K in cases:
        check_case("fixed + arbitrary MNK", M, N, K,
                   exact=(M, N, K) in ((32, 32, 64), (17, 23, 129)),
                   highbit=(M, N, K) == (33, 37, 129))
    print("All arbitrary-MNK W4A16 checks PASS")


if __name__ == "__main__":
    main()