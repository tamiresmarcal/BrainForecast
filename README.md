# brain_forecast

Forecasting future brain activity from movie stimuli and brain history, with multi-model benchmarking and traceability.

## Hypothesis

```
future_brain(t + H) = f(past_brain(t-k..t), stimulus(t-k..t))
```

Tested across multiple forecast horizons `H` (0 to 60 minutes) using subject-out cross-validation. Compares a Temporal Fusion Transformer against linear and naive baselines.

## Targets supported

Choose one per experiment via the YAML config:

| `task_type` | Target | Example column(s) |
|---|---|---|
| `regression` | DFC connections | `UDFC_1`, `UDFC_2`, … |
| `regression` | ROI activations | `roi_frontal_pole`, … |
| `classification` | Brain state | single column with integer state IDs |

The harness automatically selects regression metrics (R², Pearson r) or classification metrics (F1, accuracy) based on `task_type`.

## Predictors

| Name | Description | Adapter | Reference |
|---|---|---|---|
| `persistence` | ŷ(t+H) = y(t) | tabular | naive baseline |
| `moving_average` | ŷ(t+H) = mean(y[t-k..t]) | tabular | smoothed baseline |
| `ar` | ŷ(t+H) = β·y_past | tabular | classical AR(p) |
| `banded_ridge` | linear with per-feature-family α | tabular | Dupré la Tour et al. 2022 |
| `tft` | Temporal Fusion Transformer | sequence | Lim et al. 2021 |

Temporal alignment:
- TFT learns it implicitly from the input window
- Banded ridge uses HRF convolution on movie features (Glover canonical, default)
- AR/MA/Persistence use only past targets, no alignment needed

## Cross-validation

**Stratified subject-out:**
- No subject appears in both train and test of any fold
- Each fold's test set contains subjects from every movie (proportional to that movie's subject count)
- Cohorts with ≥ `loso_threshold` subjects use k-fold; smaller cohorts use leave-one-subject-out

This is the scheme was designed for when subjects do not all watch the same stimuli and the hypothesis is about a general functional relationship that should generalize to new brains.

## Quickstart

Install:

```bash
cd brain_forecast
pip install -e .
```

Prepare your data as a parquet or CSV file with these reserved columns:
- `sub` — subject ID
- `start` — time in seconds within each subject's recording
- `cohort` — movie identifier (used for stratification)

Plus any number of feature and target columns.

### Typed feature roles

Following the TFT paper (Lim et al. 2021, Eq. 1), features are typed into
three categories that the package routes to different parts of each model:

| Role | Symbol | Meaning | Example |
|---|---|---|---|
| `static` | s | time-invariant per subject | age, sex |
| `known_dynamic` | x | known across past **and** future | the stimulus (whole movie known in advance) |
| `observed_dynamic` | z | known in the past only | brain history (cannot know future brain) |

This matters: the stimulus is *known*, so the TFT may attend to it at and
beyond the forecast time (the encoding-model signal at H=0, the leading-stimulus
signal at H>0). Brain history is *observed* (past only). Static covariates go
through dedicated static encoders, not the temporal path. Banded ridge uses
the same typing as its regularization bands (static / stimulus / brain /
history), which is exactly the feature-exclusion benchmark structure.

The simple benchmarks (Persistence, MA, AR) ignore typing — they only use the
target's own past.

Write a config (see `configs/example.yaml`), then run:

```bash
python -m brain_forecast run --config configs/example.yaml
```

This produces:
- `runs/<output_dir>/scores.csv` — one row per (predictor, fold, cohort, horizon)
- `runs/<output_dir>/horizon_curves.png` — R² vs horizon, one line per predictor
- `runs/<output_dir>/horizon_curves_per_cohort.png` — per-movie breakdown
- `runs/<output_dir>/heatmap.png` — cohort × horizon heatmap for the best predictor

## Modular Python API

For interactive use or custom experiments:

```python
from brain_forecast.data import load_bundle
from brain_forecast.features import TabularAdapter, SequenceAdapter
from brain_forecast.cv import StratifiedSubjectOutCV
from brain_forecast.evaluation import run_experiment
from brain_forecast.reporting import plot_horizon_curves

bundle = load_bundle(
    path="data.parquet",
    target_cols=["UDFC_1", "UDFC_2"],
    task_type="regression",
    static=["age", "sex"],
    static_categorical=["sex"],
    known_dynamic=["mov_v1", "mov_audio_mel"],   # stimulus
    observed_dynamic=["umap0/5"],                # brain history
)

results = run_experiment(
    bundle=bundle,
    predictor_specs=[
        {"name": "persistence"},
        {"name": "tft", "kwargs": {"max_epochs": 30}},
    ],
    horizons_min=[0, 5, 15, 30, 60],
    cv=StratifiedSubjectOutCV(k_default=5, loso_threshold=10),
)

plot_horizon_curves(results, metric="r2_mean", output_path="curves.png")
```

## Package structure

```
brain_forecast/
├── data.py             # FeatureBundle + load_bundle
├── features.py         # TabularAdapter, SequenceAdapter, HRF utilities
├── predictors/
│   ├── __init__.py     # Predictor protocol + make_predictor factory
│   ├── persistence.py
│   ├── moving_average.py
│   ├── ar.py
│   ├── banded_ridge.py
│   └── tft.py
├── cv.py               # StratifiedSubjectOutCV
├── evaluation.py       # run_experiment
├── reporting.py        # aggregate_scores, plot_horizon_curves
└── cli.py              # YAML-driven CLI
```

## What's not in v0

- Hyperparameter tuning (uses fixed reasonable defaults)
- Explainability (SHAP, Integrated Gradients) — to be added
- Multi-dataset pooling — to be added when needed
- Experiment tracking (MLflow, etc.) — uses plain CSVs in `output_dir`

## References

- Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.
- Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., & Gallant, J. L. (2022). Feature-space selection with banded ridge regression. *NeuroImage*, 264, 119728.
- Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4), 416–429.
