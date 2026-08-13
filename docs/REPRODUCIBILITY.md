# Reproducibility notes

## Levels of reproducibility

1. **Code inspection:** all method modules and parameter definitions are public.
2. **Functional reproduction:** the synthetic corpus permits end-to-end execution.
3. **Experimental protocol reproduction:** users with their own authorized corpus can run
   the same ablation and evaluation scripts.
4. **Exact-score reproduction:** unavailable publicly because the study corpus and expert
   question-answer set cannot legally be redistributed.

## Important controls

- Record the Python and package versions used in each run.
- Record the exact Ollama model tag and model digest.
- Use fixed random seeds where supported by the selected model backend.
- Preserve the thresholds in `config/settings.py` and the switches in
  `config/ablation_modes.json`.
- Report GPU type, available VRAM, quantization, and inference settings.
- Build each ablation configuration from the same corpus snapshot.

Large-language-model generation may not be bitwise deterministic across hardware and
runtime versions. Retrieval metrics should therefore be reported separately from judged
generation metrics.

