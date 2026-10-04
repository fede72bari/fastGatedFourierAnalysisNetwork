# Changelog

All notable changes to `fastGatedFourierAnalysisNetwork` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-04

First release.

### Added
- `backend = 'jax'`: the network runs on JAX too (the backend to use on TPUs); `compute_gradients` /
  `gradient_feature_importance` support it. Models are portable across TensorFlow, PyTorch and JAX.
- `fastGatedFourierAnalysisNetwork` class (alias `fastGatedFAN`): gated Fourier Analysis Network with the API, scaling
  and splitting rules, callbacks and saved-file layout of `fastANN` 2.2.0 (classification, regression, autoencoder;
  TensorFlow or PyTorch backend; best model saved during training; full run restored with `load_all`).
- `GatedFAN` Keras layer (`[g·cos(xWp), g·sin(xWp), (1 − mean(g))·act(xWp̄ + b)]`, one trainable gate per frequency),
  serializable and backend-independent, and the `FrequencyClip` constraint; both available from `fan_layers()`.
- Parameters `periodic_share`, `gated`, `frequency_init_std`, `min_frequency`, `max_frequency`, saved in the
  hyperparameters JSON.
- `periodic_components_report()`: gates and learned periods per layer.
- `time_index_feature()`: elapsed time as an input for forecasting with periodic components.
- `sample_weight`, `monitor_auc`, `monitor_auc_rows` (as in fastANN 2.3 and fastLSTM 2.4): weighted training rows and
  epoch selection on the AUC of a subset of test rows (`val_monitored_auc`).
- `frequency_init = 'log_periods'` with `init_periods`: deterministic first-layer frequencies on a log-spaced grid of
  periods (`initialize_log_periods()`).
- `periodic_share = 0` builds the same network without periodic part (plain MLP control).
- Tests for both backends (`tests/test_fastGatedFAN.py`) and an extrapolation benchmark (`examples/extrapolation_benchmark.py`).
