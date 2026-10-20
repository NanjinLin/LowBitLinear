# LowBitLinear Benchmark 使用说明与审计记录

审计日期：2026-10-20。以当前 `setup.py`、Python forward 和正式注册为依据。
本次只改 `benchmarks/`，不改生产代码、Kernel、量化格式、dtype 或依赖版本。

## 最终结构

```text
benchmarks/
├── bench_w4a16.py      # 唯一延迟/加速比入口：GEMV、CuTe GEMM、API 或 graph 模式
├── profile_w4a16.py    # Nsight Compute 原始算子目标；不输出 benchmark latency
├── _common.py         # 数据、数值语义、设备检查、计时、统计与环境信息
├── README.md
└── results/           # 本地原始 JSON，沿用仓库已有 ignore，不自动提交
```

当前扩展只包含：

- `lowbitlinear_ext::w4a16_gemv_warp_vec(x, packed_w, scales, zeros, group_size)`。
- `lowbitlinear_cute_w4a16::gemm(X, packed_w, scales, zeros)`。

统一 benchmark 调用 `QuantizedLinear.from_float(..., bits=4, group_size=64)` 准备权重，
再计时公开的 `layer(x)`。量化/pack/构造不计入 forward latency。
公开 dispatch：M=1、N%128=0、K%64=0 选择 GEMV；其余 CUDA group_size=64 选择 CuTe。
profile 为隔离 Nsight 中的目标 Kernel，才直接调用已安装的正式算子；不 JIT 编译或绕过包加载。

## 执行命令

先在项目原有 `operator-dev` 环境运行 Tests：

```bash
cd /home/linxinan/projects/lowBitLinear
python -m pytest -q -ra
```

以下 `python` 指已安装 LowBitLinear 的当前环境。本次实际使用
`/home/linxinan/miniconda3/envs/operator-dev/bin/python`，无需修改依赖或重装扩展。
若确需重建，沿用现有入口 `python -m pip install -e . --no-build-isolation`，
并按 `setup.py` 准备 CUDA/CUTLASS 头文件。

```bash
# 公开 API 的同步单请求延迟；默认 3 个 GEMV + 6 个 GEMM shape
python benchmarks/bench_w4a16.py --mode api --json benchmarks/results/api.json

# GPU Graph Replay；与 API 数据分开保存、解释
python benchmarks/bench_w4a16.py --mode graph --json benchmarks/results/graph.json

# 分开运行两套；也可用 --shapes 替换尺寸
python benchmarks/bench_w4a16.py --suite gemv --mode graph
python benchmarks/bench_w4a16.py --suite gemm --mode api
python benchmarks/bench_w4a16.py --suite gemm --shapes 17x23x65 128x256x1024 \
  --mode graph --warmup 50 --repeats 30 --graph-batch 32 --graph-warmup 5 --seed 42 --device 0

# 不依赖 profiler 的目标校验与执行
python benchmarks/profile_w4a16.py --operator gemv --shape 1x129x256
python benchmarks/profile_w4a16.py --operator gemm --shape 33x37x129
```

支持脚本执行与 `python -m benchmarks.bench_w4a16` / `python -m benchmarks.profile_w4a16`。
`--help` 列出参数。group_size 固定 64、activation/原始 weight/output 固定 FP16，
scales FP32、zeros/packed INT32，与当前正式加速路径一致。
`--suite gemv` 只接受确实走 GEMV 的公开接口尺寸；混合路由使用 `--suite all`。
profile 的原始 GEMV 仅要求 M=1、K%64=0，允许非 128 倍数 N。

### Nsight Compute

```bash
ncu --profile-from-start off --kernel-name regex:cute_w4a16_gemm_kernel \
  --launch-count 1 --set basic \
  python benchmarks/profile_w4a16.py --operator gemm --shape 33x37x129

ncu --profile-from-start off --kernel-name regex:w4a16_gemv_warp_vec \
  --launch-count 1 --set basic \
  python benchmarks/profile_w4a16.py --operator gemv --shape 1x129x256
```

`profile_w4a16.py` 的数据准备、正确性调用、50 次 warmup 和同步均在 profiler range 外。
`cudaProfilerStart/Stop` 仅包住 `--launches` 指定的目标调用，无需脆弱的 `--launch-skip 20`。
默认只请求一个逻辑目标 launch；Nsight 为收集计数器进行的内部 replay passes 由工具管理。
不要将 profiler 插桩后的 Duration 放入普通 latency 排名。

## 测量定义与公平性

### API 模式

每个实现用相同方式：先在计时外 `cuda.synchronize()`，随后 `perf_counter_ns()`、
一次 forward、完成同步，再停止计时。报告的是稳态、同步的单请求墙钟延迟（us），
包括 Python、验证/dispatch、输出分配和完成同步成本；不是纯 Kernel 时间，也不含权重量化或 CPU/GPU 传输。
每次 forward 的输出释放在停止计时后，warmup 已建立 caching allocator 和库句柄。

### Graph 模式

每个实现分别在 side stream warmup，建立 stream 依赖并完成同步后，捕获相同数量的调用。
默认一个 graph 含 32 次调用；捕获、Python 逻辑、首次分配、校验与 graph warmup 不计时。
静态输入、输出、graph storage 保持存活；重放前后都检查输出，避免空 graph 或错误捕获被当作成功。

使用已初始化的 CUDA Events 包住一次 replay，等待 end event 完成后读取时间，
毫秒转微秒，再除以 graph-batch。结果是摊销的 GPU Graph Replay 延迟，含图调度开销；
既不是 Python API 墙钟，也不宣称是无调度开销的单 Kernel 时间。
对于极短 kernel 可增加 graph-batch 检查摊销稳定性；大 reference graph 的内存和捕获成本也会增加。
如果某个未来路径不支持 capture，应明确失败并使用 API 模式，不自动改成另一种计时结果。

### 样本、语义与比值

- 默认 warmup=50、repeats=30、graph-warmup=5；seed=42，每个 shape 独立建 generator。
- 同一 shape 的三个实现按固定 seed 打乱、循环轮换测量顺序；每 3 轮各占每个位置一次。
- 输出 Median us；JSON 另存全部样本、P90、min/max、采样顺序、正确性最大绝对误差、峰值 tensor memory。
- 同一次运行、同一 shape、同一 timing mode 下计算 `median(reference)/median(W4A16)`。
  JSON 同时保存分子分母的原始样本；不把不同机器或不同模式混在一起排名。
- GPU 型号/SM/显存、Python/PyTorch/PyTorch build CUDA、扩展路径、参数、前后 `nvidia-smi` 状态写入 JSON。
  无法读取可选 telemetry 时会明确记录原因，不生成伪数据。

| 实现 | 定义 | 比较限制 |
| --- | --- | --- |
| `w4a16_api` | 公开 forward，每次调用融合解包/反量化与计算 | 正式目标 |
| `packed_reference` | 相同 packed state，每次用 PyTorch 解包、FP32 反量化、FP32 matmul、FP16 输出 | GEMV 使用项目 `linear_int4_reference`；CuTe 用相同量化工具并额外把反量化权重转 FP16 再转 FP32，以匹配 MMA 操作数舍入 |
| `predequant_fp16` | 提前解包/反量化为 FP16，计时 `x @ weight_half.T` | compute-only、存储格式不同、不含反量化；GEMV 还改变权重舍入。仅报告 `FP16/W4A16 context ratio`，不是严格等价替换的 speedup |

Reference 是可解释的未融合 PyTorch 基线，不代表最优低比特库。
CuTe 与未舍入的 FP32 反量化 reference 原本具有不同语义，不能靠放宽误差来混用。
校验先禁用 TF32 与 FP16 reduced-precision reduction，FP32 oracle 再转 FP16，
使用与 Tests 一致的 atol=2e-3、rtol=1e-3；不修改量化输入或数据类型来取巧。

## 默认 shape 的含义

| 路径 | MxNxK | 目的 |
| --- | --- | --- |
| GEMV | 1x128x256 | 小规模、公开 GEMV dispatch |
| GEMV | 1x1024x1024 | 中规模 decode |
| GEMV | 1x2048x4096 | 较大权重与长 K |
| CuTe | 1x1x1 | 最小边界、launch floor；不是吞吐代表 |
| CuTe | 17x23x65 | M/N/K 尾块、两组量化、尾 word |
| CuTe | 33x37x129 | M/N 尾块、三 K stage、部分尾组 |
| CuTe | 32x64x512 | 小规模对齐、多 stage |
| CuTe | 128x256x1024 | 中规模 |
| CuTe | 512x2048x2048 | 较大 prefill，仍可在当前 4 GiB 环境验证 |

当前正式 CuTe GEMM 使用 64x64x64 CTA tile、128 线程；Tests 另外覆盖 63/64/65 边界与跨 CTA 映射。

空矩阵正确性由 Tests 覆盖，不纳入性能排名。

## 验证与限制

使用上述命令重新运行；通过 `--json` 保存当前环境、原始样本和统计结果。
无 GPU、缺扩展、ABI/注册错误、数值断言失败均明确报错，不回退为伪性能结果。
CLI 检查负/零维度、错误路由、重复尺寸和无效迭代次数。

当前验证只代表这台 WSL RTX 3050 Ti Laptop GPU 和当前版本组合。
没有锁定频率/功耗或控制其他系统负载；短 kernel 和 laptop 动态频率会有波动。
输入/权重重复使用，属于 warm-cache steady state，不能当成冷缓存、生产并发服务或跨机器结论。
图内有多次连续 forward，graph 指标反映批量 replay；不要将 API 与 graph 差值解释为某一项开销的精确分解。
CUDA toolkit 版本与 `torch_build_cuda` 是不同概念，后者是 PyTorch 构建使用的 CUDA。
尚未做跨 GPU/版本矩阵、性能 CI 阈值、并发多 stream、能耗或带宽 roofline 分析。

CuTe launcher 的非当前 GPU device guard 风险仍由 Tests 中的双 GPU 用例保留；
本工具显式设置当前设备，与输入设备一致，没有修改或声称修复该生产实现问题。
