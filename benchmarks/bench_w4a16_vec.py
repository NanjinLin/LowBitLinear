"""Compatibility entry for Warp V1 vs Warp Vec, using the common graph harness."""
try:
    from .bench_w4a16_compare import main
except ImportError:
    from bench_w4a16_compare import main

if __name__ == "__main__":
    main(default_kernels=("warp", "vec"))
