# Benchmark Results — 2026-05-10

## Environment

- **Machine**: 小泽 (local server)
- **CPU**: Intel Pentium G4600 @ 3.60GHz (no AVX)
- **GPU**: NVIDIA GeForce GTX 1050 4GB (CUDA 12.2)
- **RAM**: 7.6 GB + 9 GB swap
- **Model**: Qwen3.5-35B-A3B Q4_K_M (19.71 GiB)
- **llama.cpp**: b8998 (2098fd616), built with g++ 11.4.0

## Test Protocol

All tests use prompt "写一首关于春天的五言绝句" with `max_tokens=30`,
`temperature=0.7`. Speed measured from curl wall time.

## Results

| Config | VRAM | Wall Time | Speed | Δ |
|--------|------|-----------|-------|---|
| `--cpu-moe` (all CPU) | 1,728 MiB | 46.4s | 0.65 tok/s | — |
| `--n-cpu-moe 36` (4 layers GPU) | 3,574 MiB | 40.6s | 0.74 tok/s | +14% |
| `--n-cpu-moe 35` (5 layers GPU) | 3,976 MiB | 36.5s | 0.82 tok/s | +26% |

## GPU Utilization Timeline

### Config A: Baseline
```
T1-T7:  0% (CPU prefill)
T8:     1% (GPU brief)
T9:     3% (GPU brief)
T10-T12: 0% (CPU expert compute)
T13:    8% (GPU brief)
T14:    5%
T15-T20: 0%
```

### Config B: --n-cpu-moe 36
```
T1-T11: 0% (CPU long prefill)
T12:    5%
T13:    0%
T14-T15: 0%
T16:    1%
T17-T18: 0%
T19:    5%
T20:   14%  ← GPU doing expert compute!
T21:    5%
T22-T23: 0%
T24:    3%
T25:    4%
```

### Config C: --n-cpu-moe 35
```
T1-T15: 0% (very long CPU prefill)
T16:    8%  ← GPU expert compute burst
T17:    1%
T18-T19: 0%
T20:    2%
```

## Key Observations

1. **GPU sits idle 90%+ of the time** — the CPU is the bottleneck
2. **More experts on GPU = more GPU activity**, but the benefit is sub-linear because:
   - Only 5/40 layers' experts fit in 4GB VRAM
   - The remaining 35 layers still compute on slow CPU (SSE4.2)
3. **VRAM at 98% is risky** — Config C nearly OOMs during warmup
4. **Dynamic expert caching** could use VRAM 3-4× more efficiently by caching individual hot experts instead of full layers
