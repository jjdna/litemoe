# Patches

This directory will contain patches to llama.cpp for implementing the dynamic expert cache.

## Planned Patches

| Patch | Status | Description |
|-------|--------|-------------|
| `0001-expert-cache-core.patch` | 🚧 Not started | Core cache data structures + CLI args |
| `0002-expert-cache-moe-hook.patch` | 🚧 Not started | Hook into ggml-cuda MoE path |
| `0003-expert-cache-logging.patch` | 🚧 Not started | Expert activation logging |

## How to Apply

```bash
cd /path/to/llama.cpp
git am /path/to/patches/*.patch
# Or for a single patch:
git apply /path/to/patches/0001-expert-cache-core.patch
```

## Patch Development Environment

Tested against: llama.cpp commit `2098fd616` (b8998)
Base model: Qwen3.5-35B-A3B (qwen35moe architecture)
