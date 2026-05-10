# Benchmarks

## Test Environment

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Pentium G4600 @ 3.60GHz (4C/4T) |
| **CPU ISA** | SSE4.2 only (no AVX, no AVX2, no FMA) |
| **GPU** | NVIDIA GeForce GTX 1050 4GB (CUDA 12.2) |
| **RAM** | 7.6 GB DDR4 + 9 GB swap |
| **Model** | Qwen3.5-35B-A3B Q4_K_M (19.71 GiB GGUF) |
| **Framework** | llama.cpp b8998 (2098fd616) |
| **OS** | Ubuntu 22.04.5 LTS |

## Configuration Comparison

### Config A: Baseline (All Experts on CPU)

**Command:**
```bash
llama-server -ngl 18 --cpu-moe -c 2048 -t 4 -b 512 -ub 2048
```

| Metric | Value |
|--------|-------|
| GPU VRAM used | 1,728 MiB (43%) |
| CPU RAM (RSS) | ~5.5 GB |
| Generation speed | **0.65 tok/s** |
| GPU utilization (peak) | 8% |
| GPU utilization (avg) | <3% |

**GPU Timeline (1s intervals):**
```
T1-T7:  0%  (CPU prefill)
T8:     1%
T9:     3%  (GPU briefly active)
T10-T12: 0% (CPU computing experts)
T13:    8%  (GPU briefly active)
T14:    5%
T15-T20: 0% (CPU computing experts)
```

### Config B: Last 4 Layers' Experts on GPU

**Command:**
```bash
llama-server -ngl 20 --n-cpu-moe 36 -c 2048 -t 4 -b 512 -ub 2048
```

| Metric | Value |
|--------|-------|
| GPU VRAM used | 3,574 MiB (88%) |
| CPU RAM (RSS) | ~5.5 GB |
| Generation speed | **0.74 tok/s (+14%)** |
| GPU utilization (peak) | 14% |

### Config C: Last 5 Layers' Experts on GPU

**Command:**
```bash
llama-server -ngl 18 --n-cpu-moe 35 -c 2048 -t 4 -b 512 -ub 2048
```

| Metric | Value |
|--------|-------|
| GPU VRAM used | 3,976 MiB (98.4%) |
| CPU RAM (RSS) | ~5.5 GB |
| Generation speed | **0.82 tok/s (+26%)** |
| GPU utilization (peak) | 8% |

## Detailed Test Log

### Test 1: Prefill Only (Long Prompt → 1 token)

Prompt: ~80 tokens describing AI applications
```
Prefill time: 33.4 seconds
```

### Test 2: Generation (Short prompt → 30 tokens)

Prompt: "请用中文写一段关于人工智能的介绍，50字左右。"
```
Wall time:  46.4 seconds
Output tokens: 30 (13 prompt + 30 generated)
Generation rate: 0.65 tok/s
VRAM: 1,728 MiB, GPU util: 0%
```

### Test 3: Pure Generation (50 tokens)

Prompt: "你好"
```
Wall time:  43.7 seconds
Output tokens: 50
Generation rate: 1.14 tok/s (warm cache)
```

## Bottleneck Analysis

```
Per-token timeline (Config A):
┌── GPU attention (<200ms) ──┐   ┌── next token GPU ──┐
                             ↓   ↑
                     ┌── CPU: 8 expert FFNs (~1.5s) ──┐
                     ↓                                ↑
             GPU utilization: 3-8% → CPU is the bottleneck
```

**Primary bottleneck**: CPU lacks AVX2. Expert computation runs on SSE4.2 only, which is 5-10x slower than AVX2-optimized kernels.

**Secondary bottleneck**: Single-channel DDR4-2400 memory bandwidth (~19 GB/s) limits how fast expert weights can be read from RAM.

## Projected Improvements

| Upgrade | Est. Cost | Est. Speed | Improvement Factor |
|---------|-----------|-----------|-------------------|
| CPU → i3-8100 (AVX2) | ¥80-150 | 3-5 tok/s | 4-6× |
| CPU + dual-channel RAM | ¥150-250 | 5-8 tok/s | 6-10× |
| Dynamic expert cache (no HW upgrade) | Software | 1.5-3 tok/s | 2-4× |
