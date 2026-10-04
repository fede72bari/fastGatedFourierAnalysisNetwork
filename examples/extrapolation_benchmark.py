"""
Extrapolation benchmark: forecast a series with weekly and monthly cycles plus a trend far beyond the training period.

The only input is the elapsed time (time_index_feature). The network is trained on the first 70% of the days and
predicts the following 30% (300 days never seen). Same pipeline, same widths, same training: gated FAN
(periodic_share = 1/3) against a control network with periodic_share = 0 (the same layers without the periodic part).
Model selection does not look at the forecast period (early stopping and checkpoint on the training loss).

    python examples/extrapolation_benchmark.py          (writes examples/extrapolation_benchmark.png)
"""
import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastGatedFourierAnalysisNetwork import fastGatedFAN, time_index_feature

days = pd.date_range('2021-01-01', periods = 1000, freq = 'D')
t = time_index_feature(days, unit = 'D')
rng = np.random.default_rng(0)
signal = np.sin(2 * np.pi * t / 7) + 0.6 * np.sin(2 * np.pi * t / 30.4) + 0.002 * t
y = signal + rng.normal(0, 0.15, len(t))
X = pd.DataFrame({'t_days': t}); Y = pd.DataFrame({'y': y})
cut = int(len(t) * 0.7)
# frequencies of the first layer act on the SCALED time: a cycle of P days has angular frequency 2*pi*std(t)/P
t_std = X.t_days[:cut].std()
FREQ_STD = 2 * np.pi * t_std / 7 / 2          # initial frequencies spread up to about the weekly cycle
folder = tempfile.mkdtemp() + os.sep
EPOCHS = int(os.environ.get('EPOCHS', 400))

import json
WIDTHS = json.loads(os.environ.get('WIDTHS', '[64, 64]'))
SHARE = float(os.environ.get('SHARE', 1 / 3))
FREQ_STD = float(os.environ.get('FSTD', FREQ_STD))
SEEDS = json.loads(os.environ.get('SEEDS', '[1, 2, 3]'))
ONLY_FAN = os.environ.get('ONLY_FAN') == '1'
INIT = os.environ.get('INIT', 'log_periods')
common = dict(X_data = X, Y_data = Y, model_relative_width = WIDTHS, model_dropout = [0] * len(WIDTHS), learning_rate = 2e-3,
              last_layer_activation = 'linear', loss = 'mse', metrics = ['mae'], history_metrics = ['loss', 'val_loss'],
              early_stop_monitor_metric = 'loss', early_stop_mode = 'min', checkpoint_monitor_metric = 'loss', checkpoint_mode = 'min',
              early_stop_patience = 60, scale_targets = True, save_X_Y_data = False, data_storage_path = folder)
rows, curves = [], {}
FAN_INIT = dict(frequency_init = 'log_periods', init_periods = (2, 700)) if INIT == 'log_periods' else dict(frequency_init_std = FREQ_STD)
for name, extra in [('gated FAN', dict(periodic_share = SHARE, **FAN_INIT)),
                    ('control (periodic_share = 0)', dict(periodic_share = 0))][:1 if ONLY_FAN else 2]:
    for seed in SEEDS:
        import keras; keras.utils.set_random_seed(seed)
        m = fastGatedFAN(model_name = f'{name[:4]}_{seed}', **common, **extra)
        m.network_structure_set_compile()
        m.model.fit(m.X_train_s, m.Y_train_s, epochs = 1, verbose = 0)   # warm-up of the graph (keeps logs short)
        m.network_training(epochs = EPOCHS, batch_size = 32)
        p = m.model_predict(X, descale_result = True).ravel()
        rmse_train = np.sqrt(np.mean((p[:cut] - signal[:cut]) ** 2)); rmse_future = np.sqrt(np.mean((p[cut:] - signal[cut:]) ** 2))
        rows.append(dict(model = name, seed = seed, rmse_train = rmse_train, rmse_forecast = rmse_future))
        if seed == 1: curves[name] = p
res = pd.DataFrame(rows)
print('RESULTS'); print(res.round(4).to_string(index = False))
print(res.groupby('model')[['rmse_train', 'rmse_forecast']].agg(['mean', 'std']).round(4).to_string())

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (10, 3.6))
ax.plot(days, y, color = '#bbbbbb', lw = 0.8, label = 'data (signal + noise)')
for (name, p), c in zip(curves.items(), ['#2a78d6', '#eb6834']): ax.plot(days, p, color = c, lw = 1.4, label = name)
ax.axvline(days[cut], color = '#444444', ls = ':', lw = 1); ax.text(days[cut], ax.get_ylim()[1], '  forecast (never seen) ->', va = 'top', fontsize = 9)
ax.legend(loc = 'lower left', fontsize = 8.5, frameon = False); ax.set_title('Extrapolation beyond the training period: input = elapsed time only', fontsize = 10.5)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), 'extrapolation_benchmark.png'), dpi = 150)
