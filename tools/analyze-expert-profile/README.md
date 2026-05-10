# Expert Activation Profiler

## Purpose

Analyze which MoE experts are activated during inference. This data drives:
1. **Frequency-based cache placement**: Put hot experts in GPU cache
2. **Co-activation analysis**: Find experts that tend to fire together
3. **Optimal cache sizing**: Determine how many cache slots are needed

## Usage (Planned)

```bash
# Step 1: Run inference with logging enabled
llama-server --expert-cache-log /tmp/expert_log.jsonl ...

# Step 2: Analyze the log
python3 analyze.py --log /tmp/expert_log.jsonl \
                   --output /tmp/analysis_report.html

# Step 3: Generate optimized config
python3 analyze.py --log /tmp/expert_log.jsonl \
                   --gen-config \
                   --vram-budget 2000 \
                   --output config_recommended.sh
```

## Requirements

- Python 3.10+
- matplotlib (frequency heatmaps)
- networkx (co-activation graphs)
- numpy

## Output Example

```
=== Expert Activation Summary ===
Total tokens processed: 1,000
Layers: 40 (0-39)
Total experts: 10,240 (256/layer)

=== Top-10 Hottest Experts ===
Rank  Layer  Expert  Frequency  % of tokens
 1     15     42      892       89.2%
 2      3    201      856       85.6%
 3     27     78      834       83.4%
...

=== Cache Simulation ===
Cache size (MiB) | Hit rate | Est. speed (tok/s)
       500       |   42%    |     0.92
      1000       |   61%    |     1.15
      1500       |   73%    |     1.38
      2000       |   81%    |     1.65

=== Co-activation Patterns ===
Top 3 co-activation pairs:
  (15,42) + (15,78): co-occur in 72% of tokens
  (3,201) + (27,78): co-occur in 58% of tokens
```

## Status

🚧 Not yet implemented — requires expert activation logging from the C++ runtime.
