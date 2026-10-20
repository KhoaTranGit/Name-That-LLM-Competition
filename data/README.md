# Data

The task: given a short text, predict which LLM wrote it (multi-class classification). All texts are short, single-paragraph answers (about 40–270 words) to prompts across science, philosophy, history, maths and creative writing.

```
data/
├── train/
│   ├── train.csv                # main labeled set
│   ├── additional_train_1.csv   # extra labeled set, rich metadata
│   └── additional_train_2.csv   # extra labeled set, 13 models
└── evaluation/
    ├── test.csv                 # unlabeled texts to predict
    └── sample_submission.parquet
```

## Training files

| File | Rows | Label column | Notes |
|---|---|---|---|
| `train.csv` | 2,699 | `model` | 18 models, 150 texts each (`claude-fable-5` has 149) over 150 prompts. Columns: `id`, `prompt_id`, `text`, `model`. |
| `additional_train_1.csv` | 4,926 | `model` | Same 18 models, 245–288 texts each, 288 prompts. 42 columns, see below. |
| `additional_train_2.csv` | 10,772 | `label` | 13 models, 20–1,000 texts each (heavily imbalanced), 1,000 prompts. 19 columns, see below. |

Label names are consistent across files (e.g. `gpt-5.5`, `claude-opus-5`, `gemini-3.1-pro-preview`, `kimi-k3`, `glm-5.3`, `qwen3.8-max`). `additional_train_2.csv` covers only 13 of the 18 models: it has none of the five Claude models (`claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`). Texts do not overlap across the three files.

**`additional_train_1.csv` metadata**
- Generation: `prompt_sent`, `domain`, `task_type`, `topic`, `style`, `target_words`, `word_min`/`word_max`, `attempt`.
- Provenance: `requested_model_id`, `served_model_id`, `served_provider`, `reasoning_setting`, `generated_at_utc`, `split` (`new` / `proxy_v1` / `train_regen`).
- Cost and tokens: `input_tokens`, `output_tokens`, `reasoning_tokens`, `cost_usd`.
- Quality flags: `is_refusal`, `is_truncated`, `is_empty`, `has_markdown`, `in_word_range`, `passes_funnel` (4,913 pass), `funnel_fail_step`, `max_shared_8grams_comp`.
- Missing values: `opening_phrase` (mostly empty), `source_prompt_id`, `reasoning_tokens`.

**`additional_train_2.csv` metadata**
- `requested_model`/`served_model`, `served_provider`, `setting` (`default`/`low`/`high`), `prompt`, `domain`, `task_type`, `topic`, `target_words`, `n_words`, `finish_reason` (always `stop`), token counts and `cost`.

Domains are mostly physics, philosophy, mathematics, creative prose and history. Task types are mainly "explain concept" and "describe/narrate".

## Evaluation files

- `test.csv`: columns `id`, `text`. Only 3 rows in this repo, so it looks like a placeholder or sample, not the full test set.
- `sample_submission.parquet`: columns `id`, `model`, one row per test `id`, with a placeholder prediction (`gpt-5.5`). Submissions should keep this format, with `model` set to a label name from the training data.

## Notes

- The training files differ in schema: `model` vs `label`, and `prompt_id` formats (`p081`, `n001`, `g0000`). Normalize column names before concatenating.
- Prompt ids are not shared between files, so split by `prompt_id` within a file if you want validation that doesn't leak prompts.
- Class balance differs a lot between `train.csv`/`additional_train_1.csv` (near uniform) and `additional_train_2.csv` (skewed toward the OpenAI models). Consider capping per-class counts when combining.
- `.parquet` needs `pyarrow` or `fastparquet` to read.
