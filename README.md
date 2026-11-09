# Qwen weight-editing experiment

This repository records a structural pruning experiment on
[Empero AI's Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill).
It includes the bounded-memory pruning script, model configuration and
tokenizer files, evaluation results, and the experiment report. Large model
weight files are intentionally excluded; the source checkpoint must be
downloaded separately to reproduce the edit.

## Reproduce the structural edit

1. Download `model.safetensors` from the upstream repository at pinned revision
   `0934f3d2327ff2df2197495278c4c46ae5a56bd9` into `source/model.safetensors`.
2. Run `python edit.py > output/edit-result.json`.

The script checks the source SHA-256 against `source-metadata.json`, streams
the retained tensors to `output/pruned-28/`, and verifies the output tensor
hashes. It uses only the Python standard library.

## Recorded results

The experiment removed decoder layers 24–27 from the 32-layer source. In a
short, single-repetition CPU benchmark, the pruned Q3_K_M artifact generated
3.09 tokens/sec and the baseline Q4_K_M generated 2.18 tokens/sec. The pruned
model failed the arithmetic check (417 instead of 391); the baseline passed.
The small evaluation set does not establish either model as a reliable coding
agent. See [`output/REPORT.md`](output/REPORT.md) for methods, quality results,
limitations, and benchmark settings.

The source checkpoint and generated GGUF files were removed from the working
folder to reclaim disk space. Benchmark JSON and manifests remain as records;
they do not contain the model weights.

## Upstream attribution

The upstream model repository declares Apache-2.0 licensing and is based on
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). See
[`UPSTREAM.md`](UPSTREAM.md) for pinned provenance and asset details. Check the
upstream license terms before redistributing model assets.
