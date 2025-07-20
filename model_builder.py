"""Training utilities for predictive and reinforcement learning models.

This module houses the :class:`ModelBuilder` used to train LSTM or RL models,
remote training helpers and a small REST API for integration tests.
"""

import numpy as np
import pandas as pd
import os
import time
import asyncio
from config import BotConfig
from collections import deque
import ray
from utils import logger, check_dataframe_empty, HistoricalDataCache, is_cuda_available
from dotenv import load_dotenv
try:  # prefer gymnasium if available
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError as e:  # pragma: no cover - gymnasium missing
    logger.error("gymnasium import failed: %s", e)
    raise ImportError("gymnasium package is required") from e
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import joblib
try:
    import mlflow
except ImportError as e:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore
    logger.warning("mlflow import failed: %s", e)

# Delay heavy SHAP import until needed to avoid CUDA warnings at startup
shap = None
from flask import Flask, request, jsonify
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError as e:  # pragma: no cover - optional dependency
    PPO = DQN = DummyVecEnv = None  # type: ignore
    SB3_AVAILABLE = False
    logger.warning("stable_baselines3 import failed: %s", e)


_torch_modules = None


def _get_torch_modules():
    """Lazy import torch and define neural network classes."""

    global _torch_modules
    if _torch_modules is not None:
        return _torch_modules

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class CNNLSTM(nn.Module):
        """Neural network that combines 1D convolution and LSTM layers."""

        def __init__(self, input_size, hidden_size, num_layers, dropout, conv_channels=32, kernel_size=3, l2_lambda=1e-5):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.l2_lambda = l2_lambda
            padding = kernel_size // 2
            self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attn = nn.Linear(hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 1)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            attn_weights = torch.softmax(self.attn(out), dim=1)
            context = (out * attn_weights).sum(dim=1)
            context = self.dropout(context)
            out = self.fc(context)
            return self.sigmoid(out)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    class CNNGRU(nn.Module):
        """Conv1D + GRU variant."""

        def __init__(self, input_size, hidden_size, num_layers, dropout, conv_channels=32, kernel_size=3, l2_lambda=1e-5):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.l2_lambda = l2_lambda
            padding = kernel_size // 2
            self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(conv_channels, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attn = nn.Linear(hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 1)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.gru(x, h0)
            attn_weights = torch.softmax(self.attn(out), dim=1)
            context = (out * attn_weights).sum(dim=1)
            context = self.dropout(context)
            out = self.fc(context)
            return self.sigmoid(out)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    class Net(nn.Module):
        """Simple multilayer perceptron."""

        def __init__(self, input_size, hidden_sizes=(128, 64), dropout=0.2, l2_lambda=1e-5):
            super().__init__()
            self.l2_lambda = l2_lambda
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return self.sigmoid(x)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    _torch_modules = {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "CNNLSTM": CNNLSTM,
        "CNNGRU": CNNGRU,
        "Net": Net,
    }
    return _torch_modules


# Reduce verbose TensorFlow logs before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _train_model_keras(X, y, batch_size, model_type):
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
    if model_type == "mlp":
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
    else:
        x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.Dropout(0.2)(x)
        if model_type == "gru":
            x = keras.layers.GRU(64, return_sequences=True)(x)
        else:
            x = keras.layers.LSTM(64, return_sequences=True)(x)
        attn = keras.layers.Dense(1, activation="softmax")(x)
        x = keras.layers.Multiply()([x, attn])
        x = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    val_split = 0.1
    model.fit(X, y, batch_size=batch_size, epochs=20, verbose=0, validation_split=val_split)
    val_start = int((1 - val_split) * len(X))
    val_preds = model.predict(X[val_start:]).reshape(-1)
    val_labels = y[val_start:]
    return model.get_weights(), val_preds.tolist(), val_labels.tolist()


def _train_model_lightning(X, y, batch_size, model_type):
    torch_mods = _get_torch_modules()
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    DataLoader = torch_mods["DataLoader"]
    TensorDataset = torch_mods["TensorDataset"]
    Net = torch_mods["Net"]
    CNNGRU = torch_mods["CNNGRU"]
    CNNLSTM = torch_mods["CNNLSTM"]
    import pytorch_lightning as pl
    device = torch.device("cuda" if is_cuda_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    val_size = max(1, int(0.1 * len(X_tensor)))
    train_ds = TensorDataset(X_tensor[:-val_size], y_tensor[:-val_size])
    val_ds = TensorDataset(X_tensor[-val_size:], y_tensor[-val_size:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if model_type == "mlp":
        input_dim = X.shape[1] * X.shape[2]
        net = Net(input_dim)
    elif model_type == "gru":
        net = CNNGRU(X.shape[2], 64, 2, 0.2)
    else:
        net = CNNLSTM(X.shape[2], 64, 2, 0.2)

    class LightningWrapper(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.criterion = nn.BCELoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            if model_type == "mlp":
                x = x.view(x.size(0), -1)
            y_hat = self(x).squeeze()
            loss = self.criterion(y_hat, y) + self.model.l2_regularization()
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            if model_type == "mlp":
                x = x.view(x.size(0), -1)
            y_hat = self(x).squeeze()
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    wrapper = LightningWrapper(net)
    trainer = pl.Trainer(
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
        devices=1 if is_cuda_available() else None,
    )
    trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wrapper.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            if model_type == "mlp":
                val_x = val_x.view(val_x.size(0), -1)
            out = wrapper(val_x).squeeze()
            preds.extend(out.cpu().numpy())
            labels.extend(val_y.numpy())
    return net.state_dict(), preds, labels


@ray.remote
def _train_model_remote(X, y, batch_size, model_type="cnn_lstm", framework="pytorch"):
    if framework in {"keras", "tensorflow"}:
        return _train_model_keras(X, y, batch_size, model_type)
    if framework == "lightning":
        return _train_model_lightning(X, y, batch_size, model_type)

    torch_mods = _get_torch_modules()
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    DataLoader = torch_mods["DataLoader"]
    TensorDataset = torch_mods["TensorDataset"]
    Net = torch_mods["Net"]
    CNNGRU = torch_mods["CNNGRU"]
    CNNLSTM = torch_mods["CNNLSTM"]

    device = torch.device('cuda' if is_cuda_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    val_size = max(1, int(0.1 * len(X_tensor)))
    train_dataset = TensorDataset(X_tensor[:-val_size], y_tensor[:-val_size])
    val_dataset = TensorDataset(X_tensor[-val_size:], y_tensor[-val_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if model_type == "mlp":
        input_dim = X.shape[1] * X.shape[2]
        model = Net(input_dim)
    elif model_type == "gru":
        model = CNNGRU(X.shape[2], 64, 2, 0.2)
    else:
        model = CNNLSTM(X.shape[2], 64, 2, 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 20
    patience = 3
    for _ in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            if model_type == "mlp":
                batch_X = batch_X.view(batch_X.size(0), -1)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y) + model.l2_regularization()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                if model_type == "mlp":
                    val_X = val_X.view(val_X.size(0), -1)
                val_y = val_y.to(device)
                outputs = model(val_X).squeeze()
                preds.extend(outputs.cpu().numpy())
                labels.extend(val_y.cpu().numpy())
                val_loss += criterion(outputs, val_y).item()
        val_loss /= len(val_loader)
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    return model.state_dict(), preds, labels


class ModelBuilder:
    """Simplified model builder used for training LSTM models."""

    def __init__(self, config: BotConfig, data_handler, trade_manager):
        self.config = config
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.model_type = config.get('model_type', 'cnn_lstm')
        self.nn_framework = config.get('nn_framework', 'pytorch').lower()
        self.lstm_models = {}
        if self.nn_framework in {'pytorch', 'lightning'}:
            torch_mods = _get_torch_modules()
            torch = torch_mods['torch']
            self.device = torch.device('cuda' if is_cuda_available() else 'cpu')
        else:
            self.device = 'cpu'
        logger.info(
            "Starting ModelBuilder initialization: model_type=%s, framework=%s, device=%s",
            self.model_type,
            self.nn_framework,
            self.device,
        )
        self.cache = HistoricalDataCache(config['cache_dir'])
        self.state_file = os.path.join(config['cache_dir'], 'model_builder_state.pkl')
        self.last_retrain_time = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.last_save_time = time.time()
        self.save_interval = 900
        self.scalers = {}
        self.prediction_history = {}
        self.calibrators = {}
        self.calibration_metrics = {}
        self.shap_cache_times = {}
        self.shap_cache_duration = config.get('shap_cache_duration', 86400)
        self.last_backtest_time = 0
        logger.info("ModelBuilder initialization complete")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self):
        if time.time() - self.last_save_time < self.save_interval:
            return
        try:
            if self.nn_framework == 'pytorch':
                models_state = {k: v.state_dict() for k, v in self.lstm_models.items()}
            else:
                models_state = {k: v.get_weights() for k, v in self.lstm_models.items()}
            state = {
                'lstm_models': models_state,
                'scalers': self.scalers,
                'last_retrain_time': self.last_retrain_time,
            }
            with open(self.state_file, 'wb') as f:
                joblib.dump(state, f)
            self.last_save_time = time.time()
            logger.info("Состояние ModelBuilder сохранено")
        except Exception as e:
            logger.exception("Ошибка сохранения состояния ModelBuilder: %s", e)
            raise

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state = joblib.load(f)
                self.scalers = state.get('scalers', {})
                if self.nn_framework == 'pytorch':
                    torch_mods = _get_torch_modules()
                    Net = torch_mods['Net']
                    CNNGRU = torch_mods['CNNGRU']
                    CNNLSTM = torch_mods['CNNLSTM']
                    for symbol, sd in state.get('lstm_models', {}).items():
                        scaler = self.scalers.get(symbol)
                        input_size = len(scaler.mean_) if scaler else self.config['lstm_timesteps']
                        mt = self.model_type
                        if mt == 'mlp':
                            model = Net(input_size * self.config['lstm_timesteps'])
                        elif mt == 'gru':
                            model = CNNGRU(input_size, 64, 2, 0.2)
                        else:
                            model = CNNLSTM(input_size, 64, 2, 0.2)
                        model.load_state_dict(sd)
                        model.to(self.device)
                        self.lstm_models[symbol] = model
                else:
                    from tensorflow import keras
                    for symbol, weights in state.get('lstm_models', {}).items():
                        scaler = self.scalers.get(symbol)
                        input_size = len(scaler.mean_) if scaler else self.config['lstm_timesteps']
                        if self.model_type == 'mlp':
                            inputs = keras.Input(shape=(input_size * self.config['lstm_timesteps'],))
                            x = keras.layers.Dense(128, activation='relu')(inputs)
                            x = keras.layers.Dropout(0.2)(x)
                            x = keras.layers.Dense(64, activation='relu')(x)
                        else:
                            inputs = keras.Input(shape=(self.config['lstm_timesteps'], input_size))
                            x = keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
                            x = keras.layers.Dropout(0.2)(x)
                            if self.model_type == 'gru':
                                x = keras.layers.GRU(64, return_sequences=True)(x)
                            else:
                                x = keras.layers.LSTM(64, return_sequences=True)(x)
                            attn = keras.layers.Dense(1, activation='softmax')(x)
                            x = keras.layers.Multiply()([x, attn])
                            x = keras.layers.Lambda(lambda t: keras.backend.sum(t, axis=1))(x)
                        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
                        model = keras.Model(inputs, outputs)
                        model.set_weights(weights)
                        self.lstm_models[symbol] = model
                self.last_retrain_time = state.get('last_retrain_time', self.last_retrain_time)
                logger.info("Состояние ModelBuilder загружено")
        except Exception as e:
            logger.exception("Ошибка загрузки состояния ModelBuilder: %s", e)
            raise

    # ------------------------------------------------------------------
    async def preprocess(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if check_dataframe_empty(df, f"preprocess {symbol}"):
            return pd.DataFrame()
        df = df.sort_index().interpolate(method='time', limit_direction='both')
        return df.tail(self.config['min_data_length'])

    async def prepare_lstm_features(self, symbol, indicators):
        ohlcv = self.data_handler.ohlcv
        if 'symbol' in ohlcv.index.names and symbol in ohlcv.index.get_level_values('symbol'):
            df = ohlcv.xs(symbol, level='symbol', drop_level=False)
        else:
            df = None
        if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
            return np.array([])
        df = await self.preprocess(df.droplevel('symbol'), symbol)
        if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
            return np.array([])
        features_df = df[['close', 'open', 'high', 'low', 'volume']].copy()
        features_df['funding'] = self.data_handler.funding_rates.get(symbol, 0.0)
        features_df['open_interest'] = self.data_handler.open_interest.get(symbol, 0.0)
        def _align(series: pd.Series) -> np.ndarray:
            """Return values aligned to ``df.index`` and forward filled."""
            if not isinstance(series, pd.Series):
                return np.full(len(df), 0.0, dtype=float)
            aligned = series.reindex(df.index).bfill().ffill()
            return aligned.to_numpy(dtype=float)

        features_df['ema30'] = _align(indicators.ema30)
        features_df['ema100'] = _align(indicators.ema100)
        features_df['ema200'] = _align(indicators.ema200)
        features_df['rsi'] = _align(indicators.rsi)
        features_df['adx'] = _align(indicators.adx)
        features_df['macd'] = _align(indicators.macd)
        features_df['atr'] = _align(indicators.atr)
        scaler = self.scalers.get(symbol)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features_df)
            self.scalers[symbol] = scaler
        features = scaler.transform(features_df)
        return features.astype(np.float32)

    async def retrain_symbol(self, symbol):
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            logger.warning("Нет индикаторов для %s", symbol)
            return
        features = await self.prepare_lstm_features(symbol, indicators)
        required_len = self.config['lstm_timesteps'] * 2
        if len(features) < required_len:
            history_limit = max(self.config.get('min_data_length', required_len), required_len)
            sym, df_add = await self.data_handler.fetch_ohlcv_history(
                symbol, self.config['timeframe'], history_limit
            )
            if not check_dataframe_empty(df_add, f"retrain_symbol fetch {symbol}"):
                df_add['symbol'] = sym
                df_add = df_add.set_index(['symbol', df_add.index])
                await self.data_handler.synchronize_and_update(
                    sym,
                    df_add,
                    self.data_handler.funding_rates.get(sym, 0.0),
                    self.data_handler.open_interest.get(sym, 0.0),
                    {"imbalance": 0.0, "timestamp": time.time()},
                )
                indicators = self.data_handler.indicators.get(sym)
                features = await self.prepare_lstm_features(sym, indicators)
        if len(features) < required_len:
            logger.warning(
                "Недостаточно данных для обучения %s",
                symbol,
            )
            return
        X = np.array([features[i:i + self.config['lstm_timesteps']] for i in range(len(features) - self.config['lstm_timesteps'])])
        price_now = features[:-self.config['lstm_timesteps'], 0]
        future_price = features[self.config['lstm_timesteps']:, 0]
        pct_change = (future_price - price_now) / np.clip(price_now, 1e-6, None)
        thr = self.config.get('target_change_threshold', 0.001)
        y = (pct_change > thr).astype(np.float32)
        train_task = _train_model_remote
        if self.nn_framework in {'pytorch', 'lightning'}:
            torch_mods = _get_torch_modules()
            torch = torch_mods['torch']
            train_task = _train_model_remote.options(num_gpus=1 if is_cuda_available() else 0)
        logger.debug("Dispatching _train_model_remote for %s", symbol)
        model_state, val_preds, val_labels = ray.get(
            train_task.remote(
                X,
                y,
                self.config['lstm_batch_size'],
                self.model_type,
                self.nn_framework,
            )
        )
        logger.debug("_train_model_remote completed for %s", symbol)
        if self.nn_framework in {'keras', 'tensorflow'}:
            import tensorflow as tf
            from tensorflow import keras

            def build_model():
                inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
                if self.model_type == 'mlp':
                    x = keras.layers.Flatten()(inputs)
                    x = keras.layers.Dense(128, activation='relu')(x)
                    x = keras.layers.Dropout(0.2)(x)
                    x = keras.layers.Dense(64, activation='relu')(x)
                else:
                    x = keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
                    x = keras.layers.Dropout(0.2)(x)
                    if self.model_type == 'gru':
                        x = keras.layers.GRU(64, return_sequences=True)(x)
                    else:
                        x = keras.layers.LSTM(64, return_sequences=True)(x)
                    attn = keras.layers.Dense(1, activation='softmax')(x)
                    x = keras.layers.Multiply()([x, attn])
                    x = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
                outputs = keras.layers.Dense(1, activation='sigmoid')(x)
                return keras.Model(inputs, outputs)

            model = build_model()
            model.set_weights(model_state)
        else:
            torch_mods = _get_torch_modules()
            Net = torch_mods['Net']
            CNNGRU = torch_mods['CNNGRU']
            CNNLSTM = torch_mods['CNNLSTM']
            if self.model_type == 'mlp':
                model = Net(X.shape[1] * X.shape[2])
            elif self.model_type == 'gru':
                model = CNNGRU(X.shape[2], 64, 2, 0.2)
            else:
                model = CNNLSTM(X.shape[2], 64, 2, 0.2)
            model.load_state_dict(model_state)
            model.to(self.device)
        calibrator = LogisticRegression()
        calibrator.fit(np.array(val_preds).reshape(-1, 1), np.array(val_labels))
        self.calibrators[symbol] = calibrator
        brier = brier_score_loss(val_labels, val_preds)
        prob_true, prob_pred = calibration_curve(val_labels, val_preds, n_bins=10)
        self.calibration_metrics[symbol] = {
            'brier_score': float(brier),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
        if self.config.get("mlflow_enabled", False) and mlflow is not None:
            mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "mlruns"))
            with mlflow.start_run(run_name=f"{symbol}_retrain"):
                mlflow.log_params({
                    "lstm_timesteps": self.config.get("lstm_timesteps"),
                    "lstm_batch_size": self.config.get("lstm_batch_size"),
                    "target_change_threshold": self.config.get("target_change_threshold", 0.001)
                })
                mlflow.log_metric("brier_score", float(brier))
                if self.nn_framework in {"keras", "tensorflow"}:
                    mlflow.tensorflow.log_model(model, "model")
                else:
                    mlflow.pytorch.log_model(model, "model")
        self.lstm_models[symbol] = model
        self.last_retrain_time[symbol] = time.time()
        self.save_state()
        await self.compute_shap_values(symbol, model, X)
        logger.info(
            "Модель %s обучена для %s, Brier=%.4f",
            self.model_type,
            symbol,
            brier,
        )
        await self.data_handler.telegram_logger.send_telegram_message(
            f"🎯 {symbol} обучен. Brier={brier:.4f}")

    async def train(self):
        self.load_state()
        while True:
            try:
                for symbol in self.data_handler.usdt_pairs:
                    if time.time() - self.last_retrain_time.get(symbol, 0) >= self.config['retrain_interval']:
                        await self.retrain_symbol(symbol)
                await asyncio.sleep(self.config['retrain_interval'])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Ошибка цикла обучения: %s", e)
                await asyncio.sleep(1)
                continue

    async def adjust_thresholds(self, symbol, prediction: float):
        base_long = self.config.get('base_probability_threshold', 0.6)
        adjust_step = self.config.get('threshold_adjustment', 0.05)
        loss_streak = await self.trade_manager.get_loss_streak(symbol)
        win_streak = await self.trade_manager.get_win_streak(symbol)
        if loss_streak >= self.config.get('loss_streak_threshold', 3):
            base_long = min(base_long + adjust_step, 0.9)
        elif win_streak >= self.config.get('win_streak_threshold', 3):
            base_long = max(base_long - adjust_step, 0.5)
        base_short = 1 - base_long
        history_size = self.config.get('prediction_history_size', 100)
        hist = self.prediction_history.setdefault(symbol, deque(maxlen=history_size))
        hist.append(float(prediction))
        if len(hist) < 10:
            return base_long, base_short
        mean_pred = float(np.mean(hist))
        std_pred = float(np.std(hist))
        sharpe = await self.trade_manager.get_sharpe_ratio(symbol)
        ohlcv = self.data_handler.ohlcv
        if 'symbol' in ohlcv.index.names and symbol in ohlcv.index.get_level_values('symbol'):
            df = ohlcv.xs(symbol, level='symbol', drop_level=False)
        else:
            df = None
        volatility = df['close'].pct_change().std() if df is not None and not df.empty else 0.02
        last_vol = self.trade_manager.last_volatility.get(symbol, volatility)
        vol_change = abs(volatility - last_vol) / max(last_vol, 0.01)
        adj = sharpe * 0.05 - vol_change * 0.05
        long_thr = np.clip(mean_pred + std_pred / 2 + adj, base_long, 0.9)
        short_thr = np.clip(mean_pred - std_pred / 2 - adj, 0.1, base_short)
        return long_thr, short_thr

    async def compute_shap_values(self, symbol, model, X):
        try:
            global shap
            if shap is None:
                try:
                    import shap  # type: ignore
                except ImportError as e:  # pragma: no cover - optional dependency
                    logger.warning("shap import failed: %s", e)
                    return
            if self.nn_framework != 'pytorch':
                return
            torch_mods = _get_torch_modules()
            torch = torch_mods['torch']
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            cache_file = os.path.join(self.cache.cache_dir, f"shap_{safe_symbol}.pkl")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            last_time = self.shap_cache_times.get(symbol, 0)
            if time.time() - last_time < self.shap_cache_duration:
                return
            sample = torch.tensor(X[:50], dtype=torch.float32, device=self.device)
            if self.model_type == 'mlp':
                sample = sample.view(sample.size(0), -1)
            was_training = model.training
            current_device = next(model.parameters()).device

            # Move model and sample to CPU for SHAP to avoid CuDNN RNN limitation
            model_cpu = model.to('cpu')
            sample_cpu = sample.to('cpu')
            model_cpu.train()
            # DeepExplainer does not fully support LSTM layers and may
            # produce inconsistent sums. GradientExplainer is more
            # reliable for sequence models, so use it instead.
            explainer = shap.GradientExplainer(model_cpu, sample_cpu)
            values = explainer.shap_values(sample_cpu)
            if not was_training:
                model_cpu.eval()
            model.to(current_device)
            joblib.dump(values, cache_file)
            mean_abs = np.mean(np.abs(values[0]), axis=(0, 1))
            feature_names = [
                'close', 'open', 'high', 'low', 'volume', 'funding',
                'open_interest', 'ema30', 'ema100', 'ema200', 'rsi',
                'adx', 'macd', 'atr'
            ]
            top_idx = np.argsort(mean_abs)[-3:][::-1]
            top_feats = {feature_names[i]: float(mean_abs[i]) for i in top_idx}
            self.shap_cache_times[symbol] = time.time()
            logger.info("SHAP значения для %s: %s", symbol, top_feats)
            await self.data_handler.telegram_logger.send_telegram_message(
                f"🔍 SHAP {symbol}: {top_feats}")
        except Exception as e:
            logger.exception("Ошибка вычисления SHAP для %s: %s", symbol, e)
            raise

    async def simple_backtest(self, symbol):
        try:
            model = self.lstm_models.get(symbol)
            indicators = self.data_handler.indicators.get(symbol)
            ohlcv = self.data_handler.ohlcv
            if not model or not indicators:
                return None
            if 'symbol' not in ohlcv.index.names or symbol not in ohlcv.index.get_level_values('symbol'):
                return None
            features = await self.prepare_lstm_features(symbol, indicators)
            if len(features) < self.config['lstm_timesteps'] * 2:
                return None
            X = np.array([features[i:i + self.config['lstm_timesteps']] for i in range(len(features) - self.config['lstm_timesteps'])])
            if self.nn_framework in {'keras', 'tensorflow'}:
                preds = model.predict(X).reshape(-1)
            else:
                torch_mods = _get_torch_modules()
                torch = torch_mods['torch']
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                model.eval()
                with torch.no_grad():
                    if self.model_type == 'mlp':
                        X_in = X_tensor.view(X_tensor.size(0), -1)
                    else:
                        X_in = X_tensor
                    preds = model(X_in).squeeze().cpu().numpy()
            thr = self.config.get('base_probability_threshold', 0.6)
            returns = []
            for i, p in enumerate(preds):
                price_now = features[i + self.config['lstm_timesteps'] - 1, 0]
                next_price = features[i + self.config['lstm_timesteps'], 0]
                ret = 0.0
                if p > thr:
                    ret = (next_price - price_now) / price_now
                elif p < 1 - thr:
                    ret = (price_now - next_price) / price_now
                if ret != 0.0:
                    returns.append(ret)
            if not returns:
                return None
            returns = np.array(returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(365 * 24 * 60 / pd.Timedelta(self.config['timeframe']).total_seconds())
            return float(sharpe)
        except Exception as e:
            logger.exception("Ошибка бектеста %s: %s", symbol, e)
            raise

    async def backtest_all(self):
        results = {}
        for symbol in self.data_handler.usdt_pairs:
            sharpe = await self.simple_backtest(symbol)
            if sharpe is not None:
                results[symbol] = sharpe
        self.last_backtest_time = time.time()
        return results


class TradingEnv(gym.Env if gym else object):
    """Simple trading environment for offline training."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # hold, buy, sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1],),
            dtype=np.float32,
        )

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return self.df.iloc[self.current_step].to_numpy(dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        if self.current_step < len(self.df) - 1:
            price_diff = (
                self.df["close"].iloc[self.current_step + 1]
                - self.df["close"].iloc[self.current_step]
            )
            if action == 1:  # buy
                reward = price_diff
            elif action == 2:  # sell
                reward = -price_diff
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        obs = self._get_obs()
        return obs, float(reward), done, {}


class RLAgent:
    def __init__(self, config: BotConfig, data_handler, model_builder):
        self.config = config
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.models = {}

    async def _prepare_features(self, symbol: str, indicators) -> pd.DataFrame:
        ohlcv = self.data_handler.ohlcv
        if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values("symbol"):
            df = ohlcv.xs(symbol, level="symbol", drop_level=False)
        else:
            df = None
        if check_dataframe_empty(df, f"rl_features {symbol}"):
            return pd.DataFrame()
        df = await self.model_builder.preprocess(df.droplevel("symbol"), symbol)
        if check_dataframe_empty(df, f"rl_features {symbol}"):
            return pd.DataFrame()
        features_df = df[["close", "open", "high", "low", "volume"]].copy()
        features_df["funding"] = self.data_handler.funding_rates.get(symbol, 0.0)
        features_df["open_interest"] = self.data_handler.open_interest.get(symbol, 0.0)
        def _align(series: pd.Series) -> np.ndarray:
            if not isinstance(series, pd.Series):
                return np.full(len(df), 0.0, dtype=float)
            return series.reindex(df.index).bfill().ffill().to_numpy(dtype=float)

        features_df["ema30"] = _align(indicators.ema30)
        features_df["ema100"] = _align(indicators.ema100)
        features_df["ema200"] = _align(indicators.ema200)
        features_df["rsi"] = _align(indicators.rsi)
        features_df["adx"] = _align(indicators.adx)
        features_df["macd"] = _align(indicators.macd)
        features_df["atr"] = _align(indicators.atr)
        return features_df.reset_index(drop=True)

    async def train_symbol(self, symbol: str):
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            logger.warning("Нет индикаторов для RL-обучения %s", symbol)
            return
        features_df = await self._prepare_features(symbol, indicators)
        if check_dataframe_empty(features_df, f"rl_train {symbol}") or len(features_df) < 2:
            return
        algo = self.config.get("rl_model", "PPO").upper()
        framework = self.config.get("rl_framework", "stable_baselines3").lower()
        timesteps = self.config.get("rl_timesteps", 10000)
        if framework == "rllib":
            try:
                if algo == "DQN":
                    from ray.rllib.algorithms.dqn import DQNConfig
                    cfg = (
                        DQNConfig()
                        .environment(lambda _: TradingEnv(features_df))
                        .rollouts(num_rollout_workers=0)
                    )
                else:
                    from ray.rllib.algorithms.ppo import PPOConfig
                    cfg = (
                        PPOConfig()
                        .environment(lambda _: TradingEnv(features_df))
                        .rollouts(num_rollout_workers=0)
                    )
                trainer = cfg.build()
                for _ in range(max(1, timesteps // 1000)):
                    trainer.train()
                self.models[symbol] = trainer
            except Exception as e:
                logger.exception("Ошибка RLlib-обучения %s: %s", symbol, e)
                raise
        elif framework == "catalyst":
            try:
                from catalyst import dl
                torch_mods = _get_torch_modules()
                torch = torch_mods['torch']
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(features_df.values, dtype=torch.float32)
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=32)

                model = torch.nn.Linear(features_df.shape[1], 1)

                class Runner(dl.Runner):
                    def predict_batch(self, batch):
                        x = batch[0]
                        return model(x)

                runner = Runner()
                runner.train(model=model, loaders={"train": loader}, num_epochs=max(1, timesteps // 1000))
                self.models[symbol] = model
            except Exception as e:
                logger.exception("Ошибка Catalyst-обучения %s: %s", symbol, e)
                raise
        else:
            if not SB3_AVAILABLE:
                logger.warning(
                    "stable_baselines3 not available, skipping RL training for %s",
                    symbol,
                )
                return
            env = DummyVecEnv([lambda: TradingEnv(features_df)])
            if algo == "DQN":
                model = DQN("MlpPolicy", env, verbose=0)
            else:
                model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=timesteps)
            self.models[symbol] = model
        logger.info("RL-модель обучена для %s", symbol)

    async def train_rl(self):
        for symbol in self.data_handler.usdt_pairs:
            await self.train_symbol(symbol)

    def predict(self, symbol: str, obs: np.ndarray):
        model = self.models.get(symbol)
        if model is None:
            return None
        framework = self.config.get("rl_framework", "stable_baselines3").lower()
        if framework == "rllib":
            action = model.compute_single_action(obs)[0]
        else:
            if not SB3_AVAILABLE:
                logger.warning(
                    "stable_baselines3 not available, cannot make RL prediction"
                )
                return None
            action, _ = model.predict(obs, deterministic=True)
        if int(action) == 1:
            return "buy"
        if int(action) == 2:
            return "sell"
        return None

# ----------------------------------------------------------------------
# REST API for minimal integration testing
# ----------------------------------------------------------------------

api_app = Flask(__name__)

MODEL_FILE = os.environ.get("MODEL_FILE", "model.pkl")
_model = None


def _load_model() -> None:
    global _model
    if os.path.exists(MODEL_FILE):
        try:
            _model = joblib.load(MODEL_FILE)
        except Exception as e:  # pragma: no cover - model may be corrupted
            logger.exception("Failed to load model: %s", e)
            _model = None
            raise


@api_app.route("/train", methods=["POST"])
def train_route():
    data = request.get_json(force=True)
    prices = np.array(data.get("prices", []), dtype=np.float32).reshape(-1, 1)
    labels = np.array(data.get("labels", []), dtype=np.float32)
    if len(prices) == 0 or len(prices) != len(labels):
        return jsonify({"error": "invalid training data"}), 400
    model = LogisticRegression()
    model.fit(prices, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({"status": "trained"})


@api_app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    price = float(data.get('price', 0))
    if _model is None:
        signal = 'buy' if price > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(_model.predict_proba([[price]])[0, 1])
        signal = 'buy' if prob >= 0.5 else 'sell'
    return jsonify({'signal': signal, 'prob': prob})


@api_app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


if __name__ == "__main__":
    load_dotenv()
    _load_model()
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting ModelBuilder service on %s:%s", host, port)
    api_app.run(host=host, port=port)
