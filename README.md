# Tiny LM in 16MB – Learning Project

Goal: Build and train extremely small language models (eventually ≤16 MB on disk) while learning transformers, quantization, tokenization, etc.

## Current milestone (character-level transformer)

- Dataset: Tiny Shakespeare (~1 MB text)
- Model: 4-layer transformer, 96-dim, 4 heads (~470k parameters)
- Vocab: 65 characters
- Training: ~10k steps on CPU (~20–25 min)
- Result: Generates play-like dialogue with names, archaic words, basic structure

Example output:
