# Kotoba maintenance boundary

This repository preserves the upstream BigMoeOnEdge history and carries the
runtime seam used by the Kotoba/Murakumo fleet. Upstream remains
`Helldez/BigMoeOnEdge`; Kotoba-specific changes should stay small enough to
rebase or submit upstream.

## Responsibility

- BigMoeOnEdge owns GGUF expert selection, asynchronous NVMe reads, buffer
  rebinding, and the llama.cpp execution seam.
- `num` owns backend-neutral residency and memory-budget policy.
- `torch` owns the expert-load-to-execution lifecycle contract.
- `inference` owns the Qwen4Exp model/profile contract.
- `murakumo` owns node selection, launch parameters, qualification evidence,
  and service restoration.

## Split Metal placement

`--gpu-layers N` permits non-expert tensors in the last `N` layers to use
the GPU. Streamed expert tensors and Qwen4Exp's large
`per_layer_token_embd.weight` PLE table remain CPU-backed, because the expert
streamer replaces their backing pointers for each routed token.

The default is zero. This is deliberate on 16 GiB unified-memory Macs:
resident Metal weights compete with file-backed expert pages. Murakumo must
qualify a nonzero value on each memory class rather than assuming that Metal
offload is faster.

The 2026-08-29 M4/16 GiB qualification found:

- 8/49 GPU layers: warm-up failed with Metal out-of-memory.
- 4/49 GPU layers, ubatch 64: 0.128 tok/s and 201,959 major faults/token.
- GPU offload is therefore rejected for the 16 GiB profile.

These values are evidence for that exact model, quantization, machine, and
runtime revision, not a general Metal benchmark.
