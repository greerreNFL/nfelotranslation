# Key Model Configs

Serialized `KeyModel` state files produced by the seasonal trainer in `training/Distribution/Key/`.

## Naming convention

`key_model_{season}.json` — trained on all data through `season - 1`, valid for predictions in `season`.

For example, `key_model_2026.json` was trained on 2006–2025 data and is the file loaded for 2026 predictions.

## File structure

Each file is a config envelope produced by `Utilities/JsonIo.write_config_envelope`:

```
{
    "metadata": { "pipeline_id": "...", "generated_at": "..." },
    "params": {
        "1":  { ...NumberOutcome state... },
        "2":  { ...NumberOutcome state... },
        ...
        "40": { ...NumberOutcome state... }
    }
}
```

Each entry under `params` is the serialized state of one `NumberOutcome` tracker:

| Field | Type | Description |
|-------|------|-------------|
| `number` | int | The margin integer being tracked (1–40). |
| `eff_hits` | float | Exponentially decayed cumulative count of games landing on ±k. |
| `exp_eff_hits` | float | Exponentially decayed cumulative count of expected hits at ±k from the trainer's baseline. |
| `eff_games` | float | Exponentially decayed cumulative count of total games observed. |
| `trained_through` | int | Most recent season ingested into this tracker. |

## How the ratio is derived at runtime

`NumberOutcome.get_ratio` blends the raw observed-to-expected ratio toward `1.0` by a credibility weight:

```
raw_ratio   = eff_hits / exp_eff_hits
credibility = min(1, exp_eff_hits / threshold)
ratio       = 1 + (raw_ratio - 1) * credibility
```

`threshold` is supplied via the `params` dict injected at load time (shipped value: `25`). A `ratio > 1` is a key number, `ratio < 1` is a dead zone, and `ratio = 1` is no adjustment.

## How the ratio is applied at runtime

The ratio is applied multiplicatively against the discretized baseline PMF passed in by `MarginDistributionModel`. For each tracked integer `k`:

```
excess_at_+k = (ratio - 1) * baseline_pmf[+k]
excess_at_-k = (ratio - 1) * baseline_pmf[-k]
```

`baseline_pmf[k]` is small at integers far from the spread, so the absolute size of the per-bin correction shrinks naturally with distance. There is no separate distance-decay parameter.
