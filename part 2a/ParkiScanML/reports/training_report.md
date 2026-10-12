# ParkiScanML — training report

Generated 2026-10-11T20:11:35.227940+00:00

> Research benchmark on a small public dataset with participant-grouped cross-validation. Not clinically validated; not for diagnosis or screening decisions.

## oxford_voice

- Task: classification — PD (1) vs healthy (0)
- Unit: sustained vowel recording
- Rows: 195, participants: 32 (24 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **logistic_regression**
- Usable for research scores: **no**
- ⚠ CI_INCLUDES_CHANCE: the 95% participant-bootstrap CI of ROC AUC includes 0.5
- ⚠ SMALL_SAMPLE: only 32 participants

Metrics at **participant level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.406 | 0.189–0.622 | 0.406±0.001 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.724 | 0.469–0.942 | 0.740±0.026 | 0.688 | 0.875 | 0.500 |
| random_forest | 0.677 | 0.460–0.867 | 0.744±0.022 | 0.542 | 0.833 | 0.250 |
| hist_gradient_boosting | 0.625 | 0.359–0.891 | 0.678±0.037 | 0.667 | 0.958 | 0.375 |

## telemonitoring

- Task: regression — motor UPDRS (all participants have early PD)
- Unit: home voice recording
- Rows: 5875, participants: 42
- CV: GroupKFold(5) (participant-grouped)
- Best non-baseline model: **ridge**
- Usable for research scores: **no**
- ⚠ NO_BETTER_THAN_BASELINE: best model does not beat predicting the training mean
- Note: age/sex/test_time removed so the model cannot shortcut via demographics or time

| model | MAE | R² |
|---|---|---|
| dummy_mean | 7.15 | -0.042 |
| ridge | 7.36 | -0.133 |
| random_forest | 7.59 | -0.251 |
| hist_gradient_boosting | 7.61 | -0.233 |

## speech_multi

- Task: classification — PD (1) vs healthy (0)
- Unit: one of 26 speech recordings per participant
- Rows: 1040, participants: 40 (20 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **logistic_regression**
- Usable for research scores: **no**
- ⚠ CI_INCLUDES_CHANCE: the 95% participant-bootstrap CI of ROC AUC includes 0.5
- ⚠ SMALL_SAMPLE: only 40 participants
- Note: UPDRS column removed (it is only defined for PD and would leak the label)

Metrics at **participant level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.500 | 0.500–0.500 | 0.500±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.670 | 0.486–0.821 | 0.565±0.047 | 0.525 | 0.550 | 0.500 |
| random_forest | 0.630 | 0.448–0.800 | 0.547±0.032 | 0.625 | 0.600 | 0.650 |
| hist_gradient_boosting | 0.648 | 0.470–0.807 | 0.552±0.027 | 0.600 | 0.550 | 0.650 |

## spiral

- Task: classification — PD (1) vs control (0)
- Unit: spiral test 0, one file per participant
- Rows: 40, participants: 40 (25 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **random_forest**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 40 participants
- Note: features computed by parkiscan.features.spiral (same code used for in-game traces)

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.500 | 0.500–0.500 | 0.500±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.744 | 0.573–0.883 | 0.748±0.019 | 0.667 | 0.600 | 0.733 |
| random_forest | 0.829 | 0.682–0.945 | 0.843±0.014 | 0.747 | 0.760 | 0.733 |
| hist_gradient_boosting | 0.757 | 0.586–0.898 | 0.782±0.014 | 0.700 | 0.800 | 0.600 |

## gaitndd

- Task: classification — PD (1) vs control (0)
- Unit: one ~5 min walk per participant (force-sensitive insoles)
- Rows: 31, participants: 31 (15 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **random_forest**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 31 participants

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.475 | 0.331–0.636 | 0.475±0.000 | 0.475 | 0.200 | 0.750 |
| logistic_regression | 0.850 | 0.671–1.000 | 0.858±0.012 | 0.869 | 0.800 | 0.938 |
| random_forest | 0.900 | 0.754–1.000 | 0.888±0.013 | 0.838 | 0.800 | 0.875 |
| hist_gradient_boosting | 0.892 | 0.729–1.000 | 0.893±0.016 | 0.902 | 0.867 | 0.938 |

## daphnet

- Task: classification (event detection) — freezing-of-gait window (1) vs walking/standing (0)
- Unit: 4.0s window, 1.0s hop, ankle accelerometer
- Rows: 17720, participants: 10 (8 positive)
- CV: leave-one-subject-out (participant-grouped)
- Best non-baseline model: **hist_gradient_boosting**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 10 participants
- Note: detects FoG episodes in people already diagnosed; not a diagnostic model

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.237 | 0.149–0.355 | 0.237±0.000 | 0.500 | 0.000 | 1.000 |
| logistic_regression | 0.850 | 0.805–0.901 | 0.850±0.000 | 0.778 | 0.729 | 0.827 |
| random_forest | 0.867 | 0.829–0.904 | 0.867±0.000 | 0.721 | 0.512 | 0.929 |
| hist_gradient_boosting | 0.884 | 0.846–0.919 | 0.884±0.000 | 0.797 | 0.759 | 0.834 |

## etpdvis

- Task: classification — PD (1) vs healthy control (0)
- Unit: one participant = 6 visualisation images (TMT A/B/C x heatmap + scanpath)
- Rows: 40, participants: 40 (28 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **logistic_regression**
- Usable for research scores: **no**
- ⚠ SUSPICIOUSLY_HIGH: near-perfect separation on a tiny sample usually means a confound or label leakage (recording setup, rendering, task duration); do not rely on this model
- ⚠ SMALL_SAMPLE: only 40 participants
- Note: dataset ships rendered images, not raw gaze samples; image statistics are a weak proxy and the sample is very small (12 HC / 28 PD). Ink coverage mostly tracks how long the participant looked (task duration), so high scores here are not evidence of a gaze biomarker

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.429 | 0.260–0.593 | 0.429±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 1.000 | 1.000–1.000 | 1.000±0.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 0.997 | 0.980–1.000 | 0.997±0.005 | 0.958 | 1.000 | 0.917 |
| hist_gradient_boosting | 0.979 | 0.924–1.000 | 0.950±0.027 | 0.899 | 0.964 | 0.833 |
