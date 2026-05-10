# Dynamic MoE Expert GPU Cache

## Problem

Current llama.cpp MoE offloading (`--n-cpu-moe`) places experts at the **layer level** — all 256 experts in a layer are either on CPU or GPU. This is inefficient because:

1. Only 8 out of 256 experts (~3%) are activated per token
2. Expert activation is **non-uniform**: some experts fire frequently, others rarely
3. Static placement wastes VRAM on cold experts while hot experts on CPU slow everything down

## Solution: Per-Expert GPU Cache

Allocate a fixed-size GPU buffer and use it as a **cache** for individually selected experts. Keep the full expert set in CPU RAM (mmap), and copy frequently-accessed experts to GPU on demand.

```
For each token → router selects 8 experts:
  For each activated expert:
    if in GPU cache → compute on GPU (fast)
    else → compute on CPU (slow), AND async-copy to GPU cache
           (evict coldest expert if cache full)
```

## Cache Design

### Data Structures

```cpp
struct ExpertKey {
    int layer_id;   // 0-39
    int expert_id;  // 0-255
};

struct ExpertCacheEntry {
    ExpertKey key;
    float* weight_data;           // Pointer into GPU buffer
    size_t weight_size_bytes;     // ~1.5 MB per expert
    int frequency;                // Total access count
    uint64_t last_access_token;   // Token ID of last access
    float eviction_score;         // Computed score for eviction
};
```

### Eviction Policy: Hybrid LRU + LFU

```python
score(expert) = frequency(expert) / (1 + k × age)

where:
- frequency = total times this expert has been activated
- age = tokens since last use (current_token - last_access_token)
- k = decay factor (tunable, default ~0.01)
```

The hybrid policy ensures:
- **Frequently used experts** stay cached (high frequency → high score)
- **Recently used experts** stay cached (low age → high score)
- **Cold experts** get evicted first (low frequency + high age → low score)

### Cache Sizing

On GTX 1050 4GB:
- VRAM after attention layers: ~2,300 MiB free
- Reserve 300 MiB for compute buffers + safety margin
- Available for expert cache: **~2,000 MiB**
- Expert weight size: ~1.5 MB each (Q4_K_M)
- Cache capacity: **~1,300 experts** (out of 10,240 total)

## Implementation Plan

### Phase 1: Core Cache (current phase)

Modify `ggml-cuda.cu` MoE computation path:

```cpp
// Before: static expert access
src0_slice.data = (char *) src0->data + i02 * nb02;
ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);

// After: with caching
ExpertKey key = {current_layer, i02};
if (expert_cache.contains(key)) {
    // Use cached GPU weights
    src0_slice.data = expert_cache.get(key);
} else {
    // Compute on CPU, trigger async copy to GPU
    // Evict if cache full
    expert_cache.miss(key, src0->data + i02 * nb02);
    // Fall back to CPU computation
    cpu_mul_mat(...);
}
// Update metadata
expert_cache.record_access(key);
expert_cache.log_activation(key);
```

### Phase 2: Activation Profiling

Log each expert activation to a structured file:

```json
{
  "timestamp": 1715328000,
  "token_global_id": 1042,
  "layer": 15,
  "activated": [5, 12, 38, 67, 98, 134, 201, 245],
  "cache_hits": [5, 12, 38, 67],
  "cache_misses": [98, 134, 201, 245],
  "evicted": [{"layer": 12, "expert": 200, "score": 0.31}]
}
```

### Phase 3: Offline Analysis

Python tool to analyze logged activations:

- **Frequency heatmap**: Which experts fire most often?
- **Co-activation graph**: Which experts tend to fire together?
- **Temporal patterns**: Do certain experts become hot/cold over time?
- **Optimal cache sizing**: How many cache slots needed for X% hit rate?

### Phase 4: Adaptive Placement

Use profiling data to:
1. **Static optimization**: At load time, place known-hot experts in GPU
2. **Dynamic adjustment**: Periodically rebalance cache contents
3. **Predictive preloading**: Pre-cache experts likely to be activated soon

## Files to Modify

| File | Change | Est. Lines |
|------|--------|-----------|
| `common/arg.cpp` | Add `--expert-cache-mb`, `--expert-cache-log` | +30 |
| `common/fit.cpp` | VRAM budget for cache | +50 |
| `ggml-cuda/expert-cache.cuh` | **New**: cache core | +400 |
| `ggml-cuda/ggml-cuda.cu` | Hook into MoE computation | +150 |
| `src/models/qwen35moe.cpp` | Tag expert tensors | +20 |
| **Total** | | **~650** |

## Expected Impact

| Metric | Before | After (estimate) |
|--------|--------|-----------------|
| GPU utilization | 3-8% | 30-60% |
| Expert cache hit rate | 0% (static) | 60-80% |
| Inference speed | 0.65 tok/s | 1.5-3.0 tok/s |
| VRAM efficiency | Layer-granular | Expert-granular |
