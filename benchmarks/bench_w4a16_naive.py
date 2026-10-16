"""Compatibility entry for Naive; use --profile for untimed kernel profiling."""
try:
    from .bench_w4a16_compare import main
except ImportError:
    from bench_w4a16_compare import main

if __name__ == "__main__":
    main(default_kernels=("naive",), default_shapes=((1024, 1024),))
