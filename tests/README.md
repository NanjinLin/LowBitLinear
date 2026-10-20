# LowBitLinear 测试维护与审计记录

审计日期：2026-10-20。范围为当前 `setup.py` 的正式安装产物与 Python 调用路径。
仅修改 `tests/`；不修改 CUDA/CuTe、Python 生产实现、依赖版本或历史 benchmark。

## 当前实现契约

判断依据：`setup.py`、`pyproject.toml`、`lowbitlinear/*.py` 与 `csrc/*.cu`。
根目录 README、部分源码 docstring 和构建脚本注释仍描述早期实现，不能作为当前接口依据。

- 公共导出：`lowbitlinear.QuantizedLinear`。
- `from_float(weight, bits=4, group_size=64)`：非空 2D FP16/FP32 权重；仅 INT4。
- `QuantizedLinear(packed_weight, scales, zeros, group_size=64, *, in_features)`：
  `in_features` 是必传关键字；用于恢复尾部 packed word 的真实 K。
- 量化与 reference 支持 CPU/CUDA、任意正 K、任意正整数 group_size；最后一组可不足 group_size。
- 每组将权重范围扩展到包含 0，再进行 unsigned affine INT4 量化；round 使用 ties-to-even。
  全零组 scale=1；q/zero 在 0..15；反量化结果为 FP32。
- packed weight：INT32 `[N, ceil(K/8)]`，每 word 8 个 INT4，低 nibble 在前。
  末 word 补零，最高 nibble 的符号位不得丢失；`unpack_int4(..., original_k=K)` 裁剪尾部。
- metadata：scales FP32、zeros INT32，shape 均为 `[N, ceil(K/group_size)]`。
- Python forward：非空 2D contiguous FP16 x；x/weight/metadata 同设备；输出 FP16 `[M,N]`。
- `state_dict` 包含 packed_weight/scales/zeros 三个 buffer；形状及 group_size 由构造函数恢复。

正式扩展仅编译以下两个源文件，不在测试中 JIT 编译：

| 源文件 | 注册算子 | 约束与数值语义 |
| --- | --- | --- |
| `csrc/w4a16_gemv_warp_vec.cu` | `lowbitlinear_ext::w4a16_gemv_warp_vec(x, packed_w, scales, zeros, group_size)` | M=1，正 N，正 K 且 K%64=0，group_size=64；四个 contiguous CUDA tensor 同 GPU；x 16-byte aligned；FP32 反量化与累加，FP16 输出 |
| `csrc/cute_w4a16_gemm.cu` | `lowbitlinear_cute_w4a16::gemm(X, packed_w, scales, zeros)` | group_size 固定 64；同 GPU、2D、contiguous；支持任意 M/N/K 尾块；原始算子支持 M/N/K=0；FP32 反量化后转 FP16 MMA operand，FP32 累加、FP16 输出 |

Python dispatch：CUDA 且 group_size=64 时，只有 M=1、K%64=0、N%128=0 走 Warp Vec；
其余走 CuTe。CPU 或其他 group_size 走 reference。CUDA 正式路径缺少扩展时必须报错。
原始 GEMV 支持非 128 倍数 N，与 Python 的 dispatch 限制不同。
原始 CuTe 的空维度合法，但 Python API/量化函数拒绝空矩阵，两者分别测试。

## 原测试逐文件处置

| 原文件 | 原目标与问题 | 处置 |
| --- | --- | --- |
| `conftest.py` | TF32、GPU、GEMV 数据 fixture；原 GPU fixture 无条件要求扩展 | 分离 GPU 可用性与正式扩展注册；统一可复现 RNG 与独立 packed reference；CUDA 测试结束同步，归属异步错误 |
| `test_quantization.py` | 量化/pack/metadata；错误地拒绝 K=65、group_size=63、非 8 对齐 pack | 按当前实现修复，补齐手算 affine、ties-to-even、任意 K/group_size/尾组、原始 word 位序、original_k 边界与 CPU/CUDA 一致性 |
| `test_linear.py` | CPU API/reference、state；旧 constructor 缺少 in_features | 修复签名与错误断言；补齐尾部、直接构造、状态恢复、buffer 迁移与 forward 重校验 |
| `test_api_dispatch.py` | 原先假设所有 M=1 用 GEMV，CUDA M>1 用 reference；检查三个旧 GEMV 注册 | 按实际 M/N/K/group_size 选择算子；wrap 真实算子执行并断言调用；只检查两个正式 schema；保留隔离进程自动加载与无扩展 CPU 验证 |
| `test_cuda_gemv.py` | 旧 naive GEMV、三个 GEMV 的重复参数检查 | 仅保留正式 Warp Vec；合并有效 shape、非法参数、CPU tensor、对齐与 stream 测试 |
| `test_cuda_gemv_warp.py` | Warp V1/Vec 重复 shape、nibble、对齐、stream | 删除；V1 不在安装路径，Vec 的独特验证全部迁入 `test_cuda_gemv.py` |
| `test_cute_w4a16_gemm.py` | 正式 CuTe 但 import 时 JIT；主要检查在 main，pytest 不执行 | 改为真实 pytest 参数用例；保留所有独特尾块、空维度、高位、精确多 stage，增加 one-hot、FP16 policy、真实量化与非法输入/stream |
| `test_cute_fp16_gemm.py` | 固定 32x32x64 FP16 教学 kernel，收集时 JIT | 删除；不属于正式扩展/Python 路径；映射、多 K 累加的验证意图迁至正式 CuTe W4A16 |
| `test_fp16_gemm_tensor_core_pipeline.py` | FP16 WMMA/cp.async 实验，收集时 JIT | 删除；不属于正式交付路径，不再测试实验 kernel 的流水实现 |
| `test_linear_api.py` | 模块顶层 CUDA API 脚本，使用 FP32 反量化 oracle 比较 CuTe | 删除文件；形状、尾部、属性验证合并到 `test_linear.py` / `test_api_dispatch.py`，改用匹配的 MMA oracle |
| `test_w4a16_gemm.py` | 单 shape synthetic INT4，调用旧 `lowbitlinear_ext.w4a16_gemm` | 删除文件；synthetic INT4 与独立 reference 验证迁至正式 CuTe |
| `test_w4a16_gemm_real_quant.py` | 旧 GEMM、手工 padding、模块顶层执行 | 删除文件；真实量化端到端验证迁至 CuTe 参数测试，使用真实任意 K 量化、不手工 padding |
| `test_w4a16_gemm_e2e.py` | 旧 SIMT/baseline operator 候选，main 未被 pytest 执行 | 删除文件；有效尾部/真实量化目标合并到正式 CuTe 与 API 测试，不动态过滤注册名来降低覆盖 |

`csrc/` 中其他 naive/warp/SIMT/padding/WMMA/FP16 文件仍可作实验参考，但当前 setup 不编译、Python 不调用。
不因这些文件仍存在就把它们的历史测试继续列为正式验收。

## 最终目录与覆盖

```text
tests/
├── README.md
├── conftest.py
├── test_quantization.py       # affine、位序、尾组/尾word、错误输入、CPU/CUDA 对齐
├── test_linear.py             # reference、构造/forward、buffer、state_dict、设备
├── test_api_dispatch.py       # 真实 dispatch、schema、扩展生命周期、无扩展、多 GPU
├── test_cuda_gemv.py          # Warp Vec 正确性、边界、约束、对齐、stream
└── test_cute_w4a16_gemm.py    # synthetic/真实量化、任意 MNK、空维度、MMA、约束、stream
```

每个用例 seed=42；GEMV correctness 另外验证 seed=123，K 最大到 4096。
共享 fixture 保存恢复 RNG 与 TF32 设置，避免模块 import 时随机初始化、运行 CUDA 或改变全局精度。
GPU 纯 PyTorch 量化/reference 不要求扩展；正式 kernel 用例明确要求加载成功与 CUDA 注册存在。

数值比较先独立解包，按索引展开 metadata，与相同量化状态比较，不与原始浮点权重直接比较。
CuTe oracle 显式将反量化权重转 FP16 再以 FP32 matmul 累加；额外有可区分两种舍入政策的精确测试。
普通 kernel case 使用 atol=2e-3 / rtol=1e-3，仅覆盖 FP32 reduction 次序及 FP16 输出舍入；
one-hot 与精确多 stage case 使用零容差。量化误差单独按 scale 步长验证，不混入 kernel 容差。

## 运行与环境

审计使用已有 `operator-dev` 环境和已安装扩展；未安装或更改依赖。
扩展文件时间晚于两个正式源文件；已在独立进程确认两个 schema、CUDA dispatch 注册与实际执行。

- WSL Ubuntu；Python 3.11；PyTorch 2.14.0+cu130；pytest 9.1.1。
- NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB，SM 8.6，单 GPU。
- CUDA toolkit 与 CUTLASS/CuTe headers 存在。
- 当前 shell 找不到 Ninja，旧测试 import 时 JIT 因此报环境错误；正式路径使用已有扩展，无需 JIT。
  不能将 Ninja 缺失归类为 kernel 回归，也没有通过删除正式 CuTe 覆盖来绕过它。
- 需要重建时沿用已有说明：`python -m pip install -e . --no-build-isolation`，保持项目原依赖约束，并先准备 toolkit/CuTe/构建工具。

在 WSL 执行：

```bash
cd /home/linxinan/projects/lowBitLinear
/home/linxinan/miniconda3/envs/operator-dev/bin/python -m pytest -q -ra
```

Windows PowerShell 等价命令：

```powershell
wsl -d Ubuntu --cd /home/linxinan/projects/lowBitLinear -- /home/linxinan/miniconda3/envs/operator-dev/bin/python -m pytest -q -ra
```

基线完整 pytest：5 个 collection error（3 个 JIT/Ninja 错误、2 个旧 GEMM AttributeError），未执行完整用例。
另外单独运行原来可收集的 5 个 pytest 文件：103 passed、60 failed；失败按上表逐项归类修复。
历史脚本中的 main 从未由 pytest 调用，不能算通过的用例。

最终结果：

| 验证 | Passed | Failed | Skipped | Collection errors | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3050 完整 pytest | 247 | 0 | 2 | 0 | 9.41s |
| 隐藏 GPU 的补充验证 | 133 | 0 | 116 | 0 | 2.69s |

完整集共 249 项：quantization 94、linear 41、API/extension 19、GEMV 37、CuTe 58。
GPU 运行仅跳过两个需要双 GPU 的非当前设备用例；未发生 import、注册或 CUDA Runtime Error。
隐藏 GPU 的运行是 CPU 兼容性的额外验证，不能代替上面的真实 GPU 完整运行：

```bash
CUDA_VISIBLE_DEVICES= /home/linxinan/miniconda3/envs/operator-dev/bin/python -m pytest -q
```

无 GPU 时只按硬件条件跳过 GPU 用例；在 GPU 机器上缺扩展、ABI/加载错误、算子未注册、
Runtime Error 均直接失败。没有 xfail 或全局 skip。`git diff --check` 通过；生产文件无 diff。

## 剩余问题与覆盖边界

- 当前单 GPU 验证未发现正式路径的功能回归。
- 静态审计发现 `cute_w4a16_gemm_launcher` 未像 Warp Vec 一样使用 CUDAGuard，
  launch 使用无 device 参数的 `getCurrentCUDAStream()`。当前设备与输入 GPU 不同时可能在错误设备启动。
  已保留 `test_operator_on_noncurrent_gpu[gemm]` 与 GEMV 对照用例，在子进程隔离潜在 CUDA context 损坏。
  当前仅一个 GPU，这两个用例按真实硬件条件 skip；尚未实测复现，不能报告为已通过或已确认的多 GPU 行为。
- 未进行跨 CUDA/PyTorch/SM 版本矩阵、CUDA memory sanitizer、极大 grid/维度压力、吞吐/性能回归验证。
- INT3、bias、backward、3D/batched input、torch.compile/FakeTensor/meta 注册不是当前实现交付能力，未虚构成功测试。
- 根 README 的“无 CUDA kernel / K 必须整除 group_size”、Python class docstring 的旧 fallback 描述，
  以及部分 benchmark/build 注释仍过时；本次记录但未越过 tests 范围修改。
