# Experimental structural pruning report

## Summary

The edit removes decoder layers 24–27 (zero-based) from a 32-layer Qwen3.8-9B
checkpoint, renumbers the remaining layers, and updates `num_hidden_layers`
and `layer_types`. It preserves the period-four hybrid attention pattern.
Layer selection was a conservative heuristic, without recovery training.

## Recorded runtime results

CPU llama.cpp commit: `e95426f16982054c379ca7bdc8186996dd22f0fe`.

| Artifact | Generation speed | Quality observations |
| --- | ---: | --- |
| Publisher baseline Q4_K_M | 2.18 tokens/sec | Passed arithmetic, exact-response, JSON, and counting checks |
| Pruned Q3_K_M | 3.09 tokens/sec | Passed exact-response, JSON, counting, and synthetic tool round-trip; failed arithmetic |

The speed comparison used four CPU threads, 32 generated tokens, one
repetition, batch 256/microbatch 128, CPU only, and repacking disabled. The
pruned Q3 result was about 42% faster in this short test. It is not a sustained
throughput guarantee. The pruned model answered 17×23 as 417 instead of 391;
the baseline answered correctly. Neither model was established as a reliable
coding agent. See the JSON files in this directory for recorded evaluations.

## Reproduction and validation

`edit.py` checks the source checkpoint SHA-256 against the pinned upstream
metadata, streams retained tensors to the output, and verifies their hashes.
It does not run inference. The source checkpoint and generated GGUF files have
been removed from this project; benchmark results are retained as records.

The GGUF artifacts used for the runtime results were quantized through Q8_0
and then requantized to Q4/Q3. These steps add quantization error, and the
publisher baseline conversion recipe was not controlled against the edited
artifact. Quality differences cannot be attributed solely to pruning.

Remaining limitations include no recovery training, broad agent benchmark,
executable code-patch test, vision inference test, controlled swap-growth
measurement, or long-context performance validation. The results are
experimental and the observed arithmetic failure is material.
