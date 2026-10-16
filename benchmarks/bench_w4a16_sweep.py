"""Retain the earlier equal-weight-count shape sweep with the common harness."""
try:
    from .bench_w4a16_compare import main
except ImportError:
    from bench_w4a16_compare import main

if __name__ == "__main__":
    main(default_shapes=(
        (4096, 1024), (2048, 2048), (1024, 4096),
        (8192, 1024), (4096, 2048), (2048, 4096),
        (4096, 4096), (8192, 2048), (2048, 8192),
    ))
