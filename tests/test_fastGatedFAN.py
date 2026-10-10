"""
End-to-end checks of fastGatedFourierAnalysisNetwork. Run with one backend per process:

    KERAS_BACKEND=tensorflow python tests/test_fastGatedFAN.py
    KERAS_BACKEND=torch      python tests/test_fastGatedFAN.py
"""
import os, sys, tempfile
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastGatedFourierAnalysisNetwork import fastGatedFAN, fan_layers, periodic_split, time_index_feature, __version__
import keras

# fixed seed: without it the weight initialisation changes at every run and the accuracy threshold below can fail by chance
keras.utils.set_random_seed(0)

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame(rng.normal(size = (n, 6)), columns = [f'f{i}' for i in range(6)])
# periodic decision boundary along f0 plus a linear part on f1: a FAN can represent both
Y = pd.DataFrame({'up': ((np.sin(3 * X.f0) + 0.5 * X.f1) > 0).astype(int)})
folder = tempfile.mkdtemp() + os.sep

# 1. layer split
assert periodic_split(368, 1 / 3) == (61, 246)
assert periodic_split(10, 0) == (0, 10)

# 2. classification: build, train, evaluate, save
fan = fastGatedFAN(X_data = X, Y_data = Y, model_relative_width = [8, 4], model_dropout = [0.1, 0.1],
                   learning_rate = 3e-3, early_stop_patience = 5, data_storage_path = folder, model_name = 'test')
fan.network_structure_set_compile()
assert [l.__class__.__name__ for l in fan.model.layers] == ['GatedFAN', 'Dropout', 'GatedFAN', 'Dropout', 'Dense']
assert fan.model.layers[0].units == 48 and fan.model.layers[2].units == 24
fan.network_training(epochs = 15, batch_size = 64)
results, probabilities = fan.network_predictions_evaluation(0.5)
accuracy = (results['up'].values == fan.Y_test['up'].values).mean()
print('test accuracy', round(accuracy, 3))
assert accuracy > 0.8, accuracy
report = fan.periodic_components_report(plot = False)
print(report.to_string())
assert len(report) == 2 and report.frequencies.tolist() == [8, 4]

# 3. reload everything in a new instance: same predictions
p_before = fan.model_predict(X.tail(50))
fan2 = fastGatedFAN()
fan2.load_all(fan.hyperparameters_file_name, file_path_name = folder)
p_after = fan2.model_predict(X.tail(50))
assert np.allclose(p_before, p_after, atol = 1e-5)
assert fan2.periodic_share == fan.periodic_share and fan2.gated is True
assert sorted(os.listdir(folder)) and all(' - GATED FAN MODEL - ' in f or 'FOR GATED FAN MODEL' in f or 'OF GATED FAN MODEL' in f for f in os.listdir(folder))

# 4. control model (periodic_share = 0) and ungated FAN, regression with scaled targets
for share, gated in [(0, True), (1 / 3, False)]:
    m = fastGatedFAN(X_data = X, Y_data = pd.DataFrame({'y': np.sin(3 * X.f0) + X.f1}), model_relative_width = [4],
                     periodic_share = share, gated = gated, last_layer_activation = 'linear', loss = 'mse', metrics = ['mae'],
                     history_metrics = ['loss', 'val_loss'], early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
                     checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min', scale_targets = True,
                     save_X_Y_data = False, data_storage_path = folder, model_name = f'reg_{share:.2f}_{gated}')
    m.network_structure_set_compile()
    m.network_training(epochs = 3, batch_size = 128)
    assert m.model_predict(X.head(3)).shape == (3, 1)

# 5. frequency bounds are enforced after training
m = fastGatedFAN(X_data = X, Y_data = Y, model_relative_width = [4], min_frequency = 0.5, max_frequency = 2.0,
                 save_X_Y_data = False, data_storage_path = folder, model_name = 'clip')
m.network_structure_set_compile(); m.network_training(epochs = 2, batch_size = 64)
w = np.abs(np.asarray(m.model.layers[0].Wp))
assert w.min() >= 0.5 - 1e-6 and w.max() <= 2.0 + 1e-6, (w.min(), w.max())

# 5b. deterministic log-spaced periods: each frequency on one feature, period 2*pi*sigma/|w| in original units
m = fastGatedFAN(X_data = X, Y_data = Y, model_relative_width = [4], periodic_share = 0.5, frequency_init = 'log_periods',
                 init_periods = (2, 50), save_X_Y_data = False, data_storage_path = folder, model_name = 'logp')
m.network_structure_set_compile()
W = np.asarray(m.model.layers[0].Wp); assert (np.count_nonzero(W, axis = 0) == 1).all()
f = np.argmax(np.abs(W), axis = 0); periods = 2 * np.pi * m.scaler.scale_[f] / np.abs(W).max(axis = 0)
assert np.isclose(periods.min(), 2) and np.isclose(periods.max(), 50), periods
m.network_training(epochs = 1, batch_size = 64)

# 5c. weighted training and AUC monitor on a subset of the test rows
hard = (np.abs(X.f2) < 1).values
m = fastGatedFAN(X_data = X, Y_data = Y, model_relative_width = [4], sample_weight = 1 + hard, monitor_auc_rows = hard,
                 early_stop_monitor_metric = 'val_monitored_auc', checkpoint_monitor_metric = 'val_monitored_auc',
                 save_X_Y_data = False, data_storage_path = folder, model_name = 'auc')
m.network_structure_set_compile(); m.network_training(epochs = 3, batch_size = 64)
assert m.loss_df.val_monitored_auc.notna().all() and m.hyperparameters['monitor_auc_rows']

# 6. time feature
t = pd.date_range('2024-01-01', periods = 4, freq = '12h')
assert np.allclose(time_index_feature(t, unit = 'D'), [0, 0.5, 1, 1.5])

print(f'ALL TESTS PASSED (fastGatedFourierAnalysisNetwork {__version__}, backend {fan.backend})')
