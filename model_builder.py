"""Training utilities for predictive and reinforcement learning models.

This module houses the :class:`ModelBuilder` used to train LSTM or RL models,
remote training helpers and a small REST API for integration tests.
"""

import numpy as np
import pandas as pd
import os
import time
import asyncio
import sys
import re
from pathlib import Path
from bot.config import BotConfig
from collections import deque

if os.getenv("TEST_MODE") == "1":
    import types
    import sys

    ray = types.ModuleType("ray")

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)
            return wrapper
        return _RayRemoteFunction(func)

    ray.remote = _ray_remote
    ray.get = lambda x: x
    ray.init = lambda *a, **k: None
    ray.is_initialized = lambda: False
else:
    import ray
from bot.utils import logger, check_dataframe_empty, HistoricalDataCache, is_cuda_available
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

# Framework identifiers used for TensorFlow/Keras
KERAS_FRAMEWORKS = {"tensorflow", "keras"}

if os.getenv("TEST_MODE") == "1":
    import types

    sb3 = types.ModuleType("stable_baselines3")

    class DummyModel:
        def __init__(self, *a, **k):
            pass

        def learn(self, *a, **k):
            return self

        def predict(self, obs, deterministic=True):
            return np.array([1]), None

    sb3.PPO = DummyModel
    sb3.DQN = DummyModel
    common = types.ModuleType("stable_baselines3.common")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")

    class DummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

    vec_env.DummyVecEnv = DummyVecEnv
    common.vec_env = vec_env
    sb3.common = common
    sys.modules["stable_baselines3"] = sb3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.vec_env"] = vec_env

    rllib_base = types.ModuleType("ray.rllib")
    alg_mod = types.ModuleType("ray.rllib.algorithms")
    dqn_mod = types.ModuleType("ray.rllib.algorithms.dqn")
    ppo_mod = types.ModuleType("ray.rllib.algorithms.ppo")

    class _Cfg:
        def environment(self, *_a, **_k):
            return self

        def rollouts(self, *_a, **_k):
            return self

        def build(self):
            return self

        def train(self):
            return {}

    dqn_mod.DQNConfig = _Cfg
    ppo_mod.PPOConfig = _Cfg
    alg_mod.dqn = dqn_mod
    alg_mod.ppo = ppo_mod
    rllib_base.algorithms = alg_mod
    sys.modules.setdefault("ray.rllib", rllib_base)
    sys.modules.setdefault("ray.rllib.algorithms", alg_mod)
    sys.modules.setdefault("ray.rllib.algorithms.dqn", dqn_mod)
    sys.modules.setdefault("ray.rllib.algorithms.ppo", ppo_mod)

    catalyst_mod = types.ModuleType("catalyst")
    dl_mod = types.ModuleType("catalyst.dl")

    class _Runner:
        def train(self, *a, **k):
            pass

    dl_mod.Runner = _Runner
    catalyst_mod.dl = dl_mod
    sys.modules.setdefault("catalyst", catalyst_mod)
    sys.modules.setdefault("catalyst.dl", dl_mod)
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


    class CNNGRU(nn.Module):
        """Conv1D + GRU variant."""

        def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            conv_channels=32,
            kernel_size=3,
            l2_lambda=1e-5,
            regression=False,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.l2_lambda = l2_lambda
            padding = kernel_size // 2
            self.conv = nn.Conv1d(
                input_size, conv_channels, kernel_size=kernel_size, padding=padding
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(
                conv_channels,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.attn = nn.Linear(hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 1)
            self.act = nn.Identity() if regression else nn.Sigmoid()

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 1)
            h0 = torch.zeros(
                self.num_layers, x.size(0), self.hidden_size, device=x.device
            )
            out, _ = self.gru(x, h0)
            attn_weights = torch.softmax(self.attn(out), dim=1)
            context = (out * attn_weights).sum(dim=1)
            context = self.dropout(context)
            out = self.fc(context)
            return self.act(out)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    class Net(nn.Module):
        """Simple multilayer perceptron."""

        def __init__(
            self, input_size, hidden_sizes=(128, 64), dropout=0.2, l2_lambda=1e-5, regression=False
        ):
            super().__init__()
            self.l2_lambda = l2_lambda
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.act = nn.Identity() if regression else nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return self.act(x)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    class PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding."""

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-(np.log(10000.0) / d_model))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class TemporalFusionTransformer(nn.Module):
        """Transformer encoder with positional encoding."""

        def __init__(
            self,
            input_size,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            l2_lambda=1e-5,
            regression=False,
        ):
            super().__init__()
            self.l2_lambda = l2_lambda
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, 1)
            self.act = nn.Identity() if regression else nn.Sigmoid()

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return self.act(x)

        def l2_regularization(self):
            return self.l2_lambda * sum(p.pow(2.0).sum() for p in self.parameters())

    _torch_modules = {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "CNNGRU": CNNGRU,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "Net": Net,
    }
    return _torch_modules


def _freeze_torch_base_layers(model, model_type):
    """Freeze initial layers of a PyTorch model based on ``model_type``."""
    layers = []
    if model_type == "mlp" and hasattr(model, "fc1"):
        layers.append(model.fc1)
    elif model_type == "gru" and hasattr(model, "conv"):
        layers.append(model.conv)
    else:
        if hasattr(model, "input_proj"):
            layers.append(model.input_proj)
        if hasattr(model, "transformer") and getattr(model.transformer, "layers", None):
            first = model.transformer.layers[0]
            layers.append(first)
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False


def _freeze_keras_base_layers(model, model_type, framework):
    """Freeze initial layers of a Keras model based on ``model_type``."""
    if framework not in KERAS_FRAMEWORKS:
        return
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow import keras

    if model_type == "mlp":
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                layer.trainable = False
                break
    else:
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv1D):
                layer.trainable = False
                break


def generate_time_series_splits(X, y, n_splits):
    """Yield train/validation indices for time series cross-validation."""
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X):
        yield train_idx, val_idx




def _train_model_keras(
    X,
    y,
    batch_size,
    model_type,
    framework,
    initial_state=None,
    epochs=20,
    n_splits=3,
    early_stopping_patience=3,
    freeze_base_layers=False,
    prediction_target="direction",
):
    if framework not in KERAS_FRAMEWORKS:
        raise ValueError("Keras framework required")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
            attn = keras.layers.Dense(1, activation="softmax")(x)
            x = keras.layers.Multiply()([x, attn])
            x = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
        elif model_type in {"tft", "transformer"}:
            attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = keras.layers.GlobalAveragePooling1D()(attn)
        else:
            attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = keras.layers.GlobalAveragePooling1D()(attn)
    activation = None if prediction_target == "pnl" else "sigmoid"
    outputs = keras.layers.Dense(1, activation=activation)(x)
    model = keras.Model(inputs, outputs)
    if freeze_base_layers:
        _freeze_keras_base_layers(model, model_type, framework)
    loss = "mse" if prediction_target == "pnl" else "binary_crossentropy"
    model.compile(optimizer="adam", loss=loss)
    if initial_state is not None:
        model.set_weights(initial_state)
    preds: list[float] = []
    labels: list[float] = []
    for train_idx, val_idx in generate_time_series_splits(X, y, n_splits):
        fold_model = keras.models.clone_model(model)
        if freeze_base_layers:
            _freeze_keras_base_layers(fold_model, model_type, framework)
        fold_model.compile(optimizer="adam", loss=loss)
        if initial_state is not None:
            fold_model.set_weights(initial_state)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
        fold_model.fit(
            X[train_idx],
            y[train_idx],
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(X[val_idx], y[val_idx]),
            callbacks=[early_stop],
        )
        fold_preds = fold_model.predict(X[val_idx]).reshape(-1)
        preds.extend(fold_preds)
        labels.extend(y[val_idx])
        model = fold_model
    return model.get_weights(), preds, labels


def _train_model_lightning(
    X,
    y,
    batch_size,
    model_type,
    initial_state=None,
    epochs=20,
    n_splits=3,
    early_stopping_patience=3,
    freeze_base_layers=False,
    prediction_target="direction",
):
    torch_mods = _get_torch_modules()
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    DataLoader = torch_mods["DataLoader"]
    TensorDataset = torch_mods["TensorDataset"]
    Net = torch_mods["Net"]
    CNNGRU = torch_mods["CNNGRU"]
    TFT = torch_mods["TemporalFusionTransformer"]
    import pytorch_lightning as pl

    device = torch.device("cuda" if is_cuda_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    preds: list[float] = []
    labels: list[float] = []
    state = None

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = is_cuda_available()

    for train_idx, val_idx in generate_time_series_splits(X_tensor, y_tensor, n_splits):
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        pt = prediction_target
        if model_type == "mlp":
            input_dim = X.shape[1] * X.shape[2]
            net = Net(input_dim, regression=pt == "pnl")
        elif model_type == "gru":
            net = CNNGRU(X.shape[2], 64, 2, 0.2, regression=pt == "pnl")
        elif model_type in {"tft", "transformer"}:
            net = TFT(X.shape[2], regression=pt == "pnl")
        else:
            net = TFT(X.shape[2], regression=pt == "pnl")
        if freeze_base_layers:
            _freeze_torch_base_layers(net, model_type)

        class LightningWrapper(pl.LightningModule):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.criterion = nn.MSELoss() if prediction_target == "pnl" else nn.BCELoss()

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
        if initial_state is not None:
            net.load_state_dict(initial_state)

        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            devices=1 if is_cuda_available() else None,
            callbacks=[early_stop],
        )
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
        wrapper.eval()

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                if model_type == "mlp":
                    val_x = val_x.view(val_x.size(0), -1)
                out = wrapper(val_x).squeeze()
                preds.extend(out.cpu().numpy().reshape(-1))
                labels.extend(val_y.numpy().reshape(-1))

        state = net.state_dict()

    return state if state is not None else {}, preds, labels


@ray.remote
def _train_model_remote(
    X,
    y,
    batch_size,
    model_type="transformer",
    framework="pytorch",
    initial_state=None,
    epochs=20,
    n_splits=3,
    early_stopping_patience=3,
    freeze_base_layers=False,
    prediction_target="direction",
):
    if framework in KERAS_FRAMEWORKS:
        return _train_model_keras(
            X,
            y,
            batch_size,
            model_type,
            framework,
            initial_state,
            epochs,
            n_splits,
            early_stopping_patience,
            freeze_base_layers,
            prediction_target,
        )
    if framework == "lightning":
        return _train_model_lightning(
            X,
            y,
            batch_size,
            model_type,
            initial_state,
            epochs,
            n_splits,
            early_stopping_patience,
            freeze_base_layers,
            prediction_target,
        )

    torch_mods = _get_torch_modules()
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    DataLoader = torch_mods["DataLoader"]
    TensorDataset = torch_mods["TensorDataset"]
    Net = torch_mods["Net"]
    CNNGRU = torch_mods["CNNGRU"]
    TFT = torch_mods["TemporalFusionTransformer"]

    cuda_available = is_cuda_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_available)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    preds: list[float] = []
    labels: list[float] = []
    state = None

    # Use a single worker to avoid multiprocessing overhead and crashes in
    # environments where spawning subprocesses is restricted (e.g., CI)
    num_workers = 0
    pin_memory = cuda_available

    for train_idx, val_idx in generate_time_series_splits(X_tensor, y_tensor, n_splits):
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        pt = prediction_target
        if model_type == "mlp":
            input_dim = X.shape[1] * X.shape[2]
            model = Net(input_dim, regression=pt == "pnl")
        elif model_type == "gru":
            model = CNNGRU(X.shape[2], 64, 2, 0.2, regression=pt == "pnl")
        elif model_type in {"tft", "transformer"}:
            model = TFT(X.shape[2], regression=pt == "pnl")
        else:
            model = TFT(X.shape[2], regression=pt == "pnl")
        if initial_state is not None:
            model.load_state_dict(initial_state)
        if freeze_base_layers:
            _freeze_torch_base_layers(model, model_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss() if pt == "pnl" else nn.BCELoss()
        model.to(device)
        best_loss = float("inf")
        epochs_no_improve = 0
        max_epochs = epochs
        patience = early_stopping_patience
        for _ in range(max_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                if model_type == "mlp":
                    batch_X = batch_X.view(batch_X.size(0), -1)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=cuda_available):
                    outputs = model(batch_X).view(-1)
                    loss = criterion(outputs, batch_y) + model.l2_regularization()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(device)
                    if model_type == "mlp":
                        val_X = val_X.view(val_X.size(0), -1)
                    val_y = val_y.to(device)
                    with torch.amp.autocast("cuda", enabled=cuda_available):
                        outputs = model(val_X).view(-1)
                    preds.extend(outputs.cpu().numpy().reshape(-1))
                    labels.extend(val_y.cpu().numpy().reshape(-1))
                    val_loss += criterion(outputs, val_y).item()
            val_loss /= len(val_loader)
            if val_loss + 1e-4 < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        state = model.state_dict()

    return state if state is not None else {}, preds, labels


class ModelBuilder:
    """Simplified model builder used for training LSTM models."""

    def __init__(self, config: BotConfig, data_handler, trade_manager):
        self.config = config
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.model_type = config.get("model_type", "transformer")
        self.nn_framework = config.get("nn_framework", "pytorch").lower()
        # Predictive models for each trading symbol
        self.predictive_models = {}
        # Backwards compatibility alias
        self.lstm_models = self.predictive_models
        if self.nn_framework in {"pytorch", "lightning"}:
            torch_mods = _get_torch_modules()
            torch = torch_mods["torch"]
            self.device = torch.device("cuda" if is_cuda_available() else "cpu")
        else:
            self.device = "cpu"
        logger.info(
            "Starting ModelBuilder initialization: model_type=%s, framework=%s, device=%s",
            self.model_type,
            self.nn_framework,
            self.device,
        )
        self.cache = HistoricalDataCache(config["cache_dir"])
        self.state_file = os.path.join(config["cache_dir"], "model_builder_state.pkl")
        self.last_retrain_time = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.last_save_time = time.time()
        self.save_interval = 900
        self.scalers = {}
        self.prediction_history = {}
        self.threshold_offset = {symbol: 0.0 for symbol in data_handler.usdt_pairs}
        self.base_thresholds = {}
        self.calibrators = {}
        self.calibration_metrics = {}
        self.performance_metrics = {}
        self.feature_cache = {}
        self.shap_cache_times = {}
        self.shap_cache_duration = config.get("shap_cache_duration", 86400)
        self.last_backtest_time = 0
        self.backtest_results = {}
        logger.info("ModelBuilder initialization complete")

    def compute_prediction_metrics(self, symbol: str):
        """Return accuracy and Brier score over recent predictions."""
        window = self.config.get("performance_window", 100)
        hist = list(self.prediction_history.get(symbol, []))[-window:]
        if not hist:
            return None
        if not isinstance(hist[0], tuple):
            return None
        pairs = [(float(p), int(l)) for p, l in hist if l is not None]
        if not pairs:
            return None
        preds, labels = zip(*pairs)
        acc = float(np.mean([(p >= 0.5) == l for p, l in zip(preds, labels)]))
        self.base_thresholds[symbol] = max(0.5, min(acc, 0.9))
        try:
            brier = float(brier_score_loss(labels, preds))
        except ValueError:
            brier = float(np.mean((np.array(labels) - np.array(preds)) ** 2))
        self.performance_metrics[symbol] = {"accuracy": acc, "brier_score": brier}
        return self.performance_metrics[symbol]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self):
        if time.time() - self.last_save_time < self.save_interval:
            return
        try:
            if self.nn_framework == "pytorch":
                models_state = {k: v.state_dict() for k, v in self.predictive_models.items()}
            else:
                models_state = {k: v.get_weights() for k, v in self.predictive_models.items()}
            state = {
                "lstm_models": models_state,
                "scalers": self.scalers,
                "last_retrain_time": self.last_retrain_time,
                "threshold_offset": self.threshold_offset,
                "base_thresholds": self.base_thresholds,
            }
            with open(self.state_file, "wb") as f:
                joblib.dump(state, f)
            self.last_save_time = time.time()
            logger.info("Состояние ModelBuilder сохранено")
        except (OSError, ValueError) as e:
            logger.exception("Ошибка сохранения состояния ModelBuilder: %s", e)
            raise

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "rb") as f:
                    state = joblib.load(f)
                self.scalers = state.get("scalers", {})
                if self.nn_framework == "pytorch":
                    torch_mods = _get_torch_modules()
                    Net = torch_mods["Net"]
                    CNNGRU = torch_mods["CNNGRU"]
                    TFT = torch_mods["TemporalFusionTransformer"]
                    for symbol, sd in state.get("lstm_models", {}).items():
                        scaler = self.scalers.get(symbol)
                        input_size = (
                            len(scaler.mean_)
                            if scaler
                            else self.config["lstm_timesteps"]
                        )
                        mt = self.model_type
                        pt = self.config.get("prediction_target", "direction")
                        if mt == "mlp":
                            model = Net(
                                input_size * self.config["lstm_timesteps"],
                                regression=pt == "pnl",
                            )
                        elif mt == "gru":
                            model = CNNGRU(input_size, 64, 2, 0.2, regression=pt == "pnl")
                        elif mt in {"tft", "transformer"}:
                            model = TFT(input_size, regression=pt == "pnl")
                        else:
                            model = TFT(input_size, regression=pt == "pnl")
                        model.load_state_dict(sd)
                        model.to(self.device)
                        self.predictive_models[symbol] = model
                else:
                    if self.nn_framework in KERAS_FRAMEWORKS:
                        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                        from tensorflow import keras

                        for symbol, weights in state.get("lstm_models", {}).items():
                            scaler = self.scalers.get(symbol)
                            input_size = (
                                len(scaler.mean_)
                                if scaler
                                else self.config["lstm_timesteps"]
                            )
                            if self.model_type == "mlp":
                                inputs = keras.Input(
                                    shape=(input_size * self.config["lstm_timesteps"],)
                                )
                                x = keras.layers.Dense(128, activation="relu")(inputs)
                                x = keras.layers.Dropout(0.2)(x)
                                x = keras.layers.Dense(64, activation="relu")(x)
                            else:
                                inputs = keras.Input(
                                    shape=(self.config["lstm_timesteps"], input_size)
                                )
                                x = keras.layers.Conv1D(
                                    32, 3, padding="same", activation="relu"
                                )(inputs)
                                x = keras.layers.Dropout(0.2)(x)
                                if self.model_type == "gru":
                                    x = keras.layers.GRU(64, return_sequences=True)(x)
                                    attn = keras.layers.Dense(1, activation="softmax")(x)
                                    x = keras.layers.Multiply()([x, attn])
                                    x = keras.layers.Lambda(
                                        lambda t: keras.backend.sum(t, axis=1)
                                    )(x)
                                elif self.model_type in {"tft", "transformer"}:
                                    attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                                    x = keras.layers.GlobalAveragePooling1D()(attn)
                                else:
                                    attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                                    x = keras.layers.GlobalAveragePooling1D()(attn)
                            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
                            model = keras.Model(inputs, outputs)
                            model.set_weights(weights)
                            self.predictive_models[symbol] = model
                    else:
                        logger.warning(
                            "Skipping TensorFlow model load because framework is %s",
                            self.nn_framework,
                        )
                self.last_retrain_time = state.get(
                    "last_retrain_time", self.last_retrain_time
                )
                self.threshold_offset.update(state.get("threshold_offset", {}))
                self.base_thresholds.update(state.get("base_thresholds", {}))
                logger.info("Состояние ModelBuilder загружено")
        except (OSError, ValueError, KeyError, ImportError) as e:
            logger.exception("Ошибка загрузки состояния ModelBuilder: %s", e)
            raise

    # ------------------------------------------------------------------
    async def preprocess(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if check_dataframe_empty(df, f"preprocess {symbol}"):
            return pd.DataFrame()
        df = df.sort_index().interpolate(method="time", limit_direction="both")
        return df.tail(self.config["min_data_length"])

    async def prepare_lstm_features(self, symbol, indicators):
        ohlcv = self.data_handler.ohlcv
        if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values(
            "symbol"
        ):
            df = ohlcv.xs(symbol, level="symbol", drop_level=False)
        else:
            df = None
        if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
            return np.array([])
        df = await self.preprocess(df.droplevel("symbol"), symbol)
        if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
            return np.array([])
        features_df = df[["close", "open", "high", "low", "volume"]].copy()
        features_df["funding"] = self.data_handler.funding_rates.get(symbol, 0.0)
        features_df["open_interest"] = self.data_handler.open_interest.get(symbol, 0.0)
        features_df["oi_change"] = self.data_handler.open_interest_change.get(
            symbol, 0.0
        )

        def _align(series: pd.Series) -> np.ndarray:
            """Return values aligned to ``df.index`` and forward filled."""
            if not isinstance(series, pd.Series):
                return np.full(len(df), 0.0, dtype=float)
            if not series.index.equals(df.index):
                if len(series) > len(df.index):
                    series = pd.Series(series.values[: len(df.index)], index=df.index)
                else:
                    series = pd.Series(series.values, index=df.index[: len(series)])
            aligned = series.reindex(df.index).bfill().ffill()
            return aligned.to_numpy(dtype=float)

        def _maybe_add(name: str, series: pd.Series, window_key: str | None = None):
            """Add indicator ``name`` if sufficient history is available."""
            if not isinstance(series, pd.Series):
                return
            if window_key is not None:
                window = self.config.get(window_key, 0)
                if len(df) < window:
                    logger.debug(
                        "Skipping %s for %s: need %s rows, have %s",
                        name,
                        symbol,
                        window,
                        len(df),
                    )
                    return
            aligned = _align(series)
            if np.isnan(aligned).any():
                logger.debug("Skipping %s for %s due to NaNs", name, symbol)
                return
            features_df[name] = aligned

        _maybe_add("ema30", indicators.ema30, "ema30_period")
        _maybe_add("ema100", indicators.ema100, "ema100_period")
        _maybe_add("ema200", indicators.ema200, "ema200_period")
        _maybe_add("rsi", indicators.rsi, "rsi_window")
        _maybe_add("adx", indicators.adx, "adx_window")
        _maybe_add("macd", indicators.macd, "macd_window_slow")
        _maybe_add("atr", indicators.atr, "atr_period_default")
        min_len = self.config.get("min_data_length", 0)
        if len(features_df) < min_len:
            return np.array([])
        features_df = features_df.dropna()
        if features_df.empty:
            logger.warning("Нет валидных данных для %s", symbol)
            return np.array([])
        data_np = features_df.to_numpy(dtype=float, copy=True)
        scaler = self.scalers.get(symbol)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data_np)
            self.scalers[symbol] = scaler
        features = scaler.transform(data_np)
        return features.astype(np.float32)

    async def precompute_features(self, symbol):
        """Precompute and cache LSTM features for ``symbol``."""
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            return
        feats = await self.prepare_lstm_features(symbol, indicators)
        self.feature_cache[symbol] = feats

    def get_cached_features(self, symbol):
        """Return cached LSTM features for ``symbol`` if available."""
        return self.feature_cache.get(symbol)

    def clear_feature_cache(self, symbol: str) -> None:
        """Remove cached LSTM features for ``symbol`` to free memory."""
        self.feature_cache.pop(symbol, None)

    def prepare_dataset(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return training sequences and targets based on ``prediction_target``."""
        tsteps = self.config.get("lstm_timesteps", 60)
        X = np.array([features[i : i + tsteps] for i in range(len(features) - tsteps)])
        price_now = features[: -tsteps, 0]
        future_price = features[tsteps:, 0]
        returns = (future_price - price_now) / np.clip(price_now, 1e-6, None)
        if self.config.get("prediction_target", "direction") == "pnl":
            y = returns.astype(np.float32)
        else:
            thr = self.config.get("target_change_threshold", 0.001)
            y = (returns > thr).astype(np.float32)
        return X, y

    async def retrain_symbol(self, symbol):
        if self.config.get("use_transfer_learning") and symbol in self.predictive_models:
            await self.fine_tune_symbol(symbol, self.config.get("freeze_base_layers", False))
            return
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            logger.warning("Нет индикаторов для %s", symbol)
            return
        features = await self.prepare_lstm_features(symbol, indicators)
        required_len = self.config["lstm_timesteps"] * 2
        if len(features) < required_len:
            history_limit = max(
                self.config.get("min_data_length", required_len), required_len
            )
            sym, df_add = await self.data_handler.fetch_ohlcv_history(
                symbol, self.config["timeframe"], history_limit
            )
            if not check_dataframe_empty(df_add, f"retrain_symbol fetch {symbol}"):
                df_add["symbol"] = sym
                df_add = df_add.set_index(["symbol", df_add.index])
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
        X, y = self.prepare_dataset(features)
        train_task = _train_model_remote
        if self.nn_framework in {"pytorch", "lightning"}:
            torch_mods = _get_torch_modules()
            torch = torch_mods["torch"]
            train_task = _train_model_remote.options(
                num_gpus=1 if is_cuda_available() else 0
            )
        logger.debug("Dispatching _train_model_remote for %s", symbol)
        model_state, val_preds, val_labels = ray.get(
            train_task.remote(
                X,
                y,
                self.config["lstm_batch_size"],
                self.model_type,
                self.nn_framework,
                None,
                20,
                self.config.get("n_splits", 3),
                self.config.get("early_stopping_patience", 3),
                False,
                self.config.get("prediction_target", "direction"),
            )
        )
        logger.debug("_train_model_remote completed for %s", symbol)
        if self.nn_framework in KERAS_FRAMEWORKS:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            from tensorflow import keras

            def build_model():
                inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
                if self.model_type == "mlp":
                    x = keras.layers.Flatten()(inputs)
                    x = keras.layers.Dense(128, activation="relu")(x)
                    x = keras.layers.Dropout(0.2)(x)
                    x = keras.layers.Dense(64, activation="relu")(x)
                else:
                    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(
                        inputs
                    )
                    x = keras.layers.Dropout(0.2)(x)
                    if self.model_type == "gru":
                        x = keras.layers.GRU(64, return_sequences=True)(x)
                        attn = keras.layers.Dense(1, activation="softmax")(x)
                        x = keras.layers.Multiply()([x, attn])
                        x = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
                    elif self.model_type in {"tft", "transformer"}:
                        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                        x = keras.layers.GlobalAveragePooling1D()(attn)
                    else:
                        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                        x = keras.layers.GlobalAveragePooling1D()(attn)
                outputs = keras.layers.Dense(1, activation="sigmoid")(x)
                return keras.Model(inputs, outputs)

            model = build_model()
            model.set_weights(model_state)
        else:
            torch_mods = _get_torch_modules()
            Net = torch_mods["Net"]
            CNNGRU = torch_mods["CNNGRU"]
            TFT = torch_mods["TemporalFusionTransformer"]
            TFT = torch_mods["TemporalFusionTransformer"]
            pt = self.config.get("prediction_target", "direction")
            if self.model_type == "mlp":
                model = Net(X.shape[1] * X.shape[2], regression=pt == "pnl")
            elif self.model_type == "gru":
                model = CNNGRU(X.shape[2], 64, 2, 0.2, regression=pt == "pnl")
            elif self.model_type in {"tft", "transformer"}:
                model = TFT(X.shape[2], regression=pt == "pnl")
            else:
                model = TFT(X.shape[2], regression=pt == "pnl")
            model.load_state_dict(model_state)
            model.to(self.device)
        if self.config.get("prediction_target", "direction") == "pnl":
            mse = float(np.mean((np.array(val_labels) - np.array(val_preds)) ** 2))
            self.calibrators[symbol] = None
            self.calibration_metrics[symbol] = {"mse": mse}
        else:
            unique_labels = np.unique(val_labels)
            brier = brier_score_loss(val_labels, val_preds)
            if unique_labels.size < 2:
                logger.warning(
                    "Калибровка пропущена для %s: валидационные метки содержат только класс %s",
                    symbol,
                    unique_labels[0] if unique_labels.size == 1 else "unknown",
                )
                self.calibrators[symbol] = None
                self.calibration_metrics[symbol] = {"brier_score": float(brier)}
            else:
                calibrator = LogisticRegression()
                calibrator.fit(np.array(val_preds).reshape(-1, 1), np.array(val_labels))
                self.calibrators[symbol] = calibrator
                prob_true, prob_pred = calibration_curve(val_labels, val_preds, n_bins=10)
                self.calibration_metrics[symbol] = {
                    "brier_score": float(brier),
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                }
        if self.config.get("mlflow_enabled", False) and mlflow is not None:
            mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "mlruns"))
            with mlflow.start_run(run_name=f"{symbol}_retrain"):
                mlflow.log_params(
                    {
                        "lstm_timesteps": self.config.get("lstm_timesteps"),
                        "lstm_batch_size": self.config.get("lstm_batch_size"),
                        "target_change_threshold": self.config.get(
                            "target_change_threshold", 0.001
                        ),
                    }
                )
                mlflow.log_metric("brier_score", float(brier))
                if self.nn_framework in KERAS_FRAMEWORKS:
                    mlflow.tensorflow.log_model(model, "model")
                else:
                    mlflow.pytorch.log_model(model, "model")
        self.predictive_models[symbol] = model
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
            f"🎯 {symbol} обучен. Brier={brier:.4f}"
        )
        self.clear_feature_cache(symbol)

    async def fine_tune_symbol(self, symbol, freeze_base_layers=False):
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            logger.warning("Нет индикаторов для %s", symbol)
            return
        features = await self.prepare_lstm_features(symbol, indicators)
        required_len = self.config["lstm_timesteps"] * 2
        if len(features) < required_len:
            history_limit = max(
                self.config.get("min_data_length", required_len), required_len
            )
            sym, df_add = await self.data_handler.fetch_ohlcv_history(
                symbol, self.config["timeframe"], history_limit
            )
            if not check_dataframe_empty(df_add, f"fine_tune_symbol fetch {symbol}"):
                df_add["symbol"] = sym
                df_add = df_add.set_index(["symbol", df_add.index])
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
            logger.warning("Недостаточно данных для дообучения %s", symbol)
            return
        X, y = self.prepare_dataset(features)
        existing = self.predictive_models.get(symbol)
        init_state = None
        if existing is not None:
            if self.nn_framework in KERAS_FRAMEWORKS:
                init_state = existing.get_weights()
            else:
                init_state = existing.state_dict()
        train_task = _train_model_remote
        if self.nn_framework in {"pytorch", "lightning"}:
            train_task = _train_model_remote.options(
                num_gpus=1 if is_cuda_available() else 0
            )
        logger.debug("Dispatching _train_model_remote for %s (fine-tune)", symbol)
        model_state, val_preds, val_labels = ray.get(
            train_task.remote(
                X,
                y,
                self.config["lstm_batch_size"],
                self.model_type,
                self.nn_framework,
                init_state,
                self.config.get("fine_tune_epochs", 5),
                self.config.get("n_splits", 3),
                self.config.get("early_stopping_patience", 3),
                freeze_base_layers,
                self.config.get("prediction_target", "direction"),
            )
        )
        logger.debug("_train_model_remote completed for %s (fine-tune)", symbol)
        if self.nn_framework in KERAS_FRAMEWORKS:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            from tensorflow import keras

            def build_model():
                inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
                if self.model_type == "mlp":
                    x = keras.layers.Flatten()(inputs)
                    x = keras.layers.Dense(128, activation="relu")(x)
                    x = keras.layers.Dropout(0.2)(x)
                    x = keras.layers.Dense(64, activation="relu")(x)
                else:
                    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(
                        inputs
                    )
                    x = keras.layers.Dropout(0.2)(x)
                    if self.model_type == "gru":
                        x = keras.layers.GRU(64, return_sequences=True)(x)
                        attn = keras.layers.Dense(1, activation="softmax")(x)
                        x = keras.layers.Multiply()([x, attn])
                        x = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
                    elif self.model_type in {"tft", "transformer"}:
                        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                        x = keras.layers.GlobalAveragePooling1D()(attn)
                    else:
                        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                        x = keras.layers.GlobalAveragePooling1D()(attn)
                outputs = keras.layers.Dense(1, activation="sigmoid")(x)
                return keras.Model(inputs, outputs)

            model = build_model()
            model.set_weights(model_state)
        else:
            torch_mods = _get_torch_modules()
            Net = torch_mods["Net"]
            CNNGRU = torch_mods["CNNGRU"]
            TFT = torch_mods["TemporalFusionTransformer"]
            pt = self.config.get("prediction_target", "direction")
            if self.model_type == "mlp":
                model = Net(X.shape[1] * X.shape[2], regression=pt == "pnl")
            elif self.model_type == "gru":
                model = CNNGRU(X.shape[2], 64, 2, 0.2, regression=pt == "pnl")
            elif self.model_type in {"tft", "transformer"}:
                model = TFT(X.shape[2], regression=pt == "pnl")
            else:
                model = TFT(X.shape[2], regression=pt == "pnl")
            model.load_state_dict(model_state)
            model.to(self.device)
        if self.config.get("prediction_target", "direction") == "pnl":
            mse = float(np.mean((np.array(val_labels) - np.array(val_preds)) ** 2))
            self.calibrators[symbol] = None
            self.calibration_metrics[symbol] = {"mse": mse}
        else:
            unique_labels = np.unique(val_labels)
            brier = brier_score_loss(val_labels, val_preds)
            if unique_labels.size < 2:
                logger.warning(
                    "Калибровка пропущена для %s: валидационные метки содержат только класс %s",
                    symbol,
                    unique_labels[0] if unique_labels.size == 1 else "unknown",
                )
                self.calibrators[symbol] = None
                self.calibration_metrics[symbol] = {"brier_score": float(brier)}
            else:
                calibrator = LogisticRegression()
                calibrator.fit(np.array(val_preds).reshape(-1, 1), np.array(val_labels))
                self.calibrators[symbol] = calibrator
                prob_true, prob_pred = calibration_curve(val_labels, val_preds, n_bins=10)
                self.calibration_metrics[symbol] = {
                    "brier_score": float(brier),
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                }
        if self.config.get("mlflow_enabled", False) and mlflow is not None:
            mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "mlruns"))
            with mlflow.start_run(run_name=f"{symbol}_fine_tune"):
                mlflow.log_params(
                    {
                        "lstm_timesteps": self.config.get("lstm_timesteps"),
                        "lstm_batch_size": self.config.get("lstm_batch_size"),
                        "target_change_threshold": self.config.get(
                            "target_change_threshold", 0.001
                        ),
                    }
                )
                mlflow.log_metric("brier_score", float(brier))
                if self.nn_framework in KERAS_FRAMEWORKS:
                    mlflow.tensorflow.log_model(model, "model")
                else:
                    mlflow.pytorch.log_model(model, "model")
        self.predictive_models[symbol] = model
        self.last_retrain_time[symbol] = time.time()
        self.save_state()
        await self.compute_shap_values(symbol, model, X)
        logger.info(
            "Модель %s дообучена для %s, Brier=%.4f",
            self.model_type,
            symbol,
            brier,
        )
        await self.data_handler.telegram_logger.send_telegram_message(
            f"🔄 {symbol} дообучен. Brier={brier:.4f}"
        )
        self.clear_feature_cache(symbol)

    async def train(self):
        self.load_state()
        while True:
            try:
                for symbol in self.data_handler.usdt_pairs:
                    metrics = self.compute_prediction_metrics(symbol)
                    if metrics and metrics.get("accuracy", 1.0) < self.config.get("retrain_threshold", 0.1):
                        await self.retrain_symbol(symbol)
                        continue
                    if (
                        time.time() - self.last_retrain_time.get(symbol, 0)
                        >= self.config["retrain_interval"]
                    ):
                        await self.retrain_symbol(symbol)
                await asyncio.sleep(self.config["retrain_interval"])
            except asyncio.CancelledError:
                raise
            except (RuntimeError, ValueError, KeyError) as e:
                logger.exception("Ошибка цикла обучения: %s", e)
                await asyncio.sleep(1)
                continue

    async def adjust_thresholds(self, symbol, prediction: float):
        base_long = self.base_thresholds.get(
            symbol, self.config.get("base_probability_threshold", 0.6)
        )
        adjust_step = self.config.get("threshold_adjustment", 0.05)
        decay = self.config.get("threshold_decay_rate", 0.1)
        loss_streak = await self.trade_manager.get_loss_streak(symbol)
        win_streak = await self.trade_manager.get_win_streak(symbol)
        offset = self.threshold_offset.get(symbol, 0.0)
        if loss_streak >= self.config.get("loss_streak_threshold", 3):
            offset += adjust_step
        elif win_streak >= self.config.get("win_streak_threshold", 3):
            offset -= adjust_step
        offset *= 1 - decay
        self.threshold_offset[symbol] = offset
        base_long = float(np.clip(base_long + offset, 0.5, 0.9))
        base_short = 1 - base_long
        history_size = self.config.get("prediction_history_size", 100)
        hist = self.prediction_history.setdefault(symbol, deque(maxlen=history_size))
        hist.append((float(prediction), None))
        self.compute_prediction_metrics(symbol)
        if len(hist) < 10:
            return base_long, base_short
        mean_pred = float(np.mean(hist))
        std_pred = float(np.std(hist))
        sharpe = await self.trade_manager.get_sharpe_ratio(symbol)
        ohlcv = self.data_handler.ohlcv
        if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values(
            "symbol"
        ):
            df = ohlcv.xs(symbol, level="symbol", drop_level=False)
        else:
            df = None
        volatility = (
            df["close"].pct_change().std() if df is not None and not df.empty else 0.02
        )
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
            if self.nn_framework != "pytorch":
                return
            torch_mods = _get_torch_modules()
            torch = torch_mods["torch"]
            safe_symbol = re.sub(r"[^A-Za-z0-9_-]", "_", symbol)
            cache_dir = getattr(self.cache, "cache_dir", None)
            if not cache_dir:
                logger.error("Не задана директория кэша SHAP")
                return
            cache_file = Path(cache_dir) / "shap" / f"shap_{safe_symbol}.pkl"
            last_time = self.shap_cache_times.get(symbol, 0)
            if time.time() - last_time < self.shap_cache_duration:
                return
            sample = torch.tensor(X[:50], dtype=torch.float32, device=self.device)
            if self.model_type == "mlp":
                sample = sample.view(sample.size(0), -1)
            was_training = model.training
            current_device = next(model.parameters()).device

            # Move model and sample to CPU for SHAP to avoid CuDNN RNN limitation
            model_cpu = model.to("cpu")
            sample_cpu = sample.to("cpu")
            model_cpu.train()
            # DeepExplainer does not fully support LSTM layers and may
            # produce inconsistent sums. GradientExplainer is more
            # reliable for sequence models, so use it instead.
            explainer = shap.GradientExplainer(model_cpu, sample_cpu)
            values = explainer.shap_values(sample_cpu)
            if not was_training:
                model_cpu.eval()
            model.to(current_device)
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(values, str(cache_file))
                if not cache_file.exists():
                    raise RuntimeError(
                        f"Файл {cache_file} не создан после сохранения SHAP"
                    )
                logger.info("SHAP значения сохранены в %s", cache_file)
            except Exception as e:
                logger.error("Ошибка записи SHAP в %s: %s", cache_file, e)
                raise RuntimeError(
                    f"Ошибка записи SHAP в {cache_file}"
                ) from e
            mean_abs = np.mean(np.abs(values[0]), axis=(0, 1))
            feature_names = [
                "close",
                "open",
                "high",
                "low",
                "volume",
                "funding",
                "open_interest",
                "oi_change",
                "ema30",
                "ema100",
                "ema200",
                "rsi",
                "adx",
                "macd",
                "atr",
            ]
            top_idx = np.argsort(mean_abs)[-3:][::-1]
            top_feats = {feature_names[i]: float(mean_abs[i]) for i in top_idx}
            self.shap_cache_times[symbol] = time.time()
            logger.info("SHAP значения для %s: %s", symbol, top_feats)
            await self.data_handler.telegram_logger.send_telegram_message(
                f"🔍 SHAP {symbol}: {top_feats}"
            )
        except (ValueError, RuntimeError, ImportError) as e:
            logger.exception("Ошибка вычисления SHAP для %s: %s", symbol, e)
            raise

    async def simple_backtest(self, symbol):
        try:
            model = self.predictive_models.get(symbol)
            indicators = self.data_handler.indicators.get(symbol)
            ohlcv = self.data_handler.ohlcv
            if not model or not indicators:
                return None
            if (
                "symbol" not in ohlcv.index.names
                or symbol not in ohlcv.index.get_level_values("symbol")
            ):
                return None
            features = await self.prepare_lstm_features(symbol, indicators)
            if len(features) < self.config["lstm_timesteps"] * 2:
                return None
            X = np.array(
                [
                    features[i : i + self.config["lstm_timesteps"]]
                    for i in range(len(features) - self.config["lstm_timesteps"])
                ]
            )
            if self.nn_framework in KERAS_FRAMEWORKS:
                preds = model.predict(X).reshape(-1)
            else:
                torch_mods = _get_torch_modules()
                torch = torch_mods["torch"]
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                model.eval()
                with torch.no_grad():
                    if self.model_type == "mlp":
                        X_in = X_tensor.view(X_tensor.size(0), -1)
                    else:
                        X_in = X_tensor
                    preds = model(X_in).squeeze().cpu().numpy()
            thr = self.config.get("base_probability_threshold", 0.6)
            returns = []
            for i, p in enumerate(preds):
                price_now = features[i + self.config["lstm_timesteps"] - 1, 0]
                next_price = features[i + self.config["lstm_timesteps"], 0]
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
            sharpe = (
                np.mean(returns)
                / (np.std(returns) + 1e-6)
                * np.sqrt(
                    365
                    * 24
                    * 60
                    / pd.Timedelta(self.config["timeframe"]).total_seconds()
                )
            )
            return float(sharpe)
        except (ValueError, ZeroDivisionError, KeyError) as e:
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

    async def backtest_loop(self):
        """Periodically backtest all symbols and emit warnings."""
        interval = self.config.get("backtest_interval", 3600)
        threshold = self.config.get("min_sharpe_ratio", 0.5)
        while True:
            try:
                self.backtest_results = await self.backtest_all()
                for sym, ratio in self.backtest_results.items():
                    if ratio < threshold:
                        msg = (
                            f"⚠️ Sharpe ratio for {sym} below {threshold}: {ratio:.2f}"
                        )
                        logger.warning(msg)
                        await self.data_handler.telegram_logger.send_telegram_message(
                            msg
                        )
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, ValueError, KeyError) as e:
                logger.exception("Backtest loop error: %s", e)
                await asyncio.sleep(1)


class TradingEnv(gym.Env if gym else object):
    """Simple trading environment for offline training."""

    def __init__(self, df: pd.DataFrame, config: BotConfig | None = None):
        self.df = df.reset_index(drop=True)
        self.config = config or BotConfig()
        self.current_step = 0
        self.balance = 0.0
        self.max_balance = 0.0
        self.position = 0  # 1 for long, -1 for short
        self.drawdown_penalty = self.config.get("drawdown_penalty", 0.0)
        # hold, open long, open short, close
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1],),
            dtype=np.float32,
        )

    def reset(self):
        self.current_step = 0
        self.position = 0
        return self._get_obs()

    def _get_obs(self):
        return self.df.iloc[self.current_step].to_numpy(dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        if action == 1:  # open long
            self.position = 1
        elif action == 2:  # open short
            self.position = -1
        elif action == 3:  # close
            self.position = 0
        if self.current_step < len(self.df) - 1:
            price_diff = (
                self.df["close"].iloc[self.current_step + 1]
                - self.df["close"].iloc[self.current_step]
            )
            reward = price_diff * self.position
            self.balance += reward
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            drawdown = (
                (self.max_balance - self.balance) / self.max_balance
                if self.max_balance > 0
                else 0.0
            )
            reward -= self.drawdown_penalty * drawdown
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
        if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values(
            "symbol"
        ):
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
            if not series.index.equals(df.index):
                if len(series) > len(df.index):
                    series = pd.Series(series.values[: len(df.index)], index=df.index)
                else:
                    series = pd.Series(series.values, index=df.index[: len(series)])
            return series.reindex(df.index).bfill().ffill().to_numpy(dtype=float)

        features_df["ema30"] = _align(indicators.ema30)
        features_df["ema100"] = _align(indicators.ema100)
        features_df["ema200"] = _align(indicators.ema200)
        features_df["rsi"] = _align(indicators.rsi)
        features_df["adx"] = _align(indicators.adx)
        features_df["macd"] = _align(indicators.macd)
        features_df["atr"] = _align(indicators.atr)
        features_df["model_pred"] = 0.0
        features_df["exposure"] = 0.0
        return features_df.reset_index(drop=True)

    async def train_symbol(self, symbol: str):
        indicators = self.data_handler.indicators.get(symbol)
        if not indicators:
            logger.warning("Нет индикаторов для RL-обучения %s", symbol)
            return
        features_df = await self._prepare_features(symbol, indicators)
        if (
            check_dataframe_empty(features_df, f"rl_train {symbol}")
            or len(features_df) < 2
        ):
            return
        bc_dataset = None
        if self.config.get("rl_use_imitation", False):
            base_model = self.model_builder.predictive_models.get(symbol)
            if base_model is not None:
                lstm_feats = await self.model_builder.prepare_lstm_features(
                    symbol, indicators
                )
                tsteps = self.config.get("lstm_timesteps", 60)
                if len(lstm_feats) > tsteps:
                    seqs = np.array(
                        [
                            lstm_feats[i : i + tsteps]
                            for i in range(len(lstm_feats) - tsteps)
                        ]
                    )
                    states = features_df.iloc[tsteps - 1 :].to_numpy(dtype=np.float32)
                    torch_mods = _get_torch_modules()
                    torch = torch_mods["torch"]
                    device = self.model_builder.device
                    base_model.eval()
                    with torch.no_grad():
                        preds = (
                            base_model(
                                torch.tensor(seqs, dtype=torch.float32, device=device)
                            )
                            .squeeze()
                            .float()
                            .cpu()
                            .numpy()
                        )
                    thr = self.config.get("base_probability_threshold", 0.6)
                    actions = np.where(preds > thr, 1, np.where(preds < 1 - thr, 2, 0))
                    actions = actions.astype(np.int64)
                    states = states[: len(actions)]
                    bc_dataset = (states, actions)
        algo = self.config.get("rl_model", "PPO").upper()
        framework = self.config.get("rl_framework", "stable_baselines3").lower()
        timesteps = self.config.get("rl_timesteps", 10000)
        if framework == "rllib":
            try:
                if algo == "DQN":
                    from ray.rllib.algorithms.dqn import DQNConfig

                    cfg = (
                        DQNConfig()
                        .environment(lambda _: TradingEnv(features_df, self.config))
                        .rollouts(num_rollout_workers=0)
                    )
                else:
                    from ray.rllib.algorithms.ppo import PPOConfig

                    cfg = (
                        PPOConfig()
                        .environment(lambda _: TradingEnv(features_df, self.config))
                        .rollouts(num_rollout_workers=0)
                    )
                trainer = cfg.build()
                for _ in range(max(1, timesteps // 1000)):
                    trainer.train()
                self.models[symbol] = trainer
            except (ImportError, RuntimeError, ValueError) as e:
                logger.exception("Ошибка RLlib-обучения %s: %s", symbol, e)
                raise
        elif framework == "catalyst":
            try:
                from catalyst import dl

                torch_mods = _get_torch_modules()
                torch = torch_mods["torch"]
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(features_df.values, dtype=torch.float32)
                )
                num_workers = min(4, os.cpu_count() or 1)
                pin_memory = is_cuda_available()
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )

                model = torch.nn.Linear(features_df.shape[1], 1)

                class Runner(dl.Runner):
                    def predict_batch(self, batch):
                        x = batch[0]
                        return model(x)

                runner = Runner()
                runner.train(
                    model=model,
                    loaders={"train": loader},
                    num_epochs=max(1, timesteps // 1000),
                )
                self.models[symbol] = model
            except (ImportError, RuntimeError, ValueError) as e:
                logger.exception("Ошибка Catalyst-обучения %s: %s", symbol, e)
                raise
        else:
            if not SB3_AVAILABLE:
                logger.warning(
                    "stable_baselines3 not available, skipping RL training for %s",
                    symbol,
                )
                return
            env = DummyVecEnv([lambda: TradingEnv(features_df, self.config)])
            if algo == "DQN":
                model = DQN("MlpPolicy", env, verbose=0)
            else:
                model = PPO("MlpPolicy", env, verbose=0)
            if bc_dataset is not None:
                states, actions = bc_dataset
                torch_mods = _get_torch_modules()
                torch = torch_mods["torch"]
                DataLoader = torch_mods["DataLoader"]
                TensorDataset = torch_mods["TensorDataset"]
                nn = torch_mods["nn"]
                dataset = TensorDataset(
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.long),
                )
                num_workers = min(4, os.cpu_count() or 1)
                pin_memory = is_cuda_available()
                loader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                policy = model.policy
                device = policy.device
                optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                policy.train()
                for _ in range(5):
                    for b_states, b_actions in loader:
                        b_states = b_states.to(device)
                        b_actions = b_actions.to(device)
                        if algo == "DQN":
                            logits = policy.q_net(b_states)
                        else:
                            features = policy.extract_features(
                                b_states, policy.pi_features_extractor
                            )
                            latent_pi = policy.mlp_extractor.forward_actor(features)
                            logits = policy.action_net(latent_pi)
                        loss = criterion(logits, b_actions)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                policy.eval()
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
        action = action.item()
        action = int(action)
        if action == 1:
            return "open_long"
        if action == 2:
            return "open_short"
        if action == 3:
            return "close"
        return "hold"


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
        except (OSError, ValueError) as e:  # pragma: no cover - model may be corrupted
            logger.exception("Failed to load model: %s", e)
            _model = None


@api_app.route("/train", methods=["POST"])
def train_route():
    data = request.get_json(force=True)
    features = np.array(data.get("features", []), dtype=np.float32)
    labels = np.array(data.get("labels", []), dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    else:
        features = features.reshape(len(features), -1)
    if features.size == 0 or len(features) != len(labels):
        return jsonify({"error": "invalid training data"}), 400
    if len(np.unique(labels)) < 2:
        return jsonify({"error": "labels must contain at least two classes"}), 400
    model = LogisticRegression(multi_class="auto")
    model.fit(features, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({"status": "trained"})


@api_app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    features = data.get("features")
    if features is None:
        # Backwards compatibility for a single price input
        price_val = float(data.get("price", 0.0))
        features = [price_val]
    features = np.array(features, dtype=np.float32)
    if features.ndim == 0:
        features = np.array([[features]], dtype=np.float32)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    else:
        features = features.reshape(1, -1)
    price = float(features[0, 0]) if features.size else 0.0
    if _model is None:
        signal = "buy" if price > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(_model.predict_proba(features)[0, 1])
        signal = "buy" if prob >= 0.5 else "sell"
    return jsonify({"signal": signal, "prob": prob})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    from bot.utils import configure_logging

    configure_logging()
    load_dotenv()
    _load_model()
    port = int(os.environ.get("PORT", "8001"))
    # По умолчанию слушаем только локальный интерфейс.
    host = os.environ.get("HOST", "127.0.0.1")
    # Prevent binding to all interfaces.
    if host.strip() == "0.0.0.0":  # nosec B104
        raise ValueError("HOST=0.0.0.0 запрещён из соображений безопасности")
    if host != "127.0.0.1":
        logger.warning(
            "Используется не локальный хост %s; убедитесь, что это намеренно",
            host,
        )
    else:
        logger.info("HOST не установлен, используется %s", host)
    logger.info("Запуск сервиса ModelBuilder на %s:%s", host, port)
    api_app.run(host=host, port=port)  # nosec B104  # хост проверен выше
