import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import joblib
import os
import time
import asyncio
import shap
from utils import logger, check_dataframe_empty, HistoricalDataCache
from collections import deque
import ray


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
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
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


@ray.remote(num_gpus=1)
def _train_lstm_remote(X, y, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    val_size = max(1, int(0.1 * len(X_tensor)))
    train_dataset = TensorDataset(X_tensor[:-val_size], y_tensor[:-val_size])
    val_dataset = TensorDataset(X_tensor[-val_size:], y_tensor[-val_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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

    def __init__(self, config, data_handler, trade_manager):
        self.config = config
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.lstm_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self):
        if time.time() - self.last_save_time < self.save_interval:
            return
        try:
            state = {
                'lstm_models': {k: v.state_dict() for k, v in self.lstm_models.items()},
                'scalers': self.scalers,
                'last_retrain_time': self.last_retrain_time,
            }
            with open(self.state_file, 'wb') as f:
                joblib.dump(state, f)
            self.last_save_time = time.time()
            logger.info("Состояние ModelBuilder сохранено")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния ModelBuilder: {e}")

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state = joblib.load(f)
                self.scalers = state.get('scalers', {})
                for symbol, sd in state.get('lstm_models', {}).items():
                    scaler = self.scalers.get(symbol)
                    input_size = len(scaler.mean_) if scaler else self.config['lstm_timesteps']
                    model = CNNLSTM(input_size, 64, 2, 0.2)
                    model.load_state_dict(sd)
                    model.to(self.device)
                    self.lstm_models[symbol] = model
                self.last_retrain_time = state.get('last_retrain_time', self.last_retrain_time)
                logger.info("Состояние ModelBuilder загружено")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния ModelBuilder: {e}")

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
        features_df['ema30'] = indicators.ema30[-len(df):].values
        features_df['ema100'] = indicators.ema100[-len(df):].values
        features_df['ema200'] = indicators.ema200[-len(df):].values
        features_df['rsi'] = indicators.rsi[-len(df):].values
        features_df['adx'] = indicators.adx[-len(df):].values
        features_df['macd'] = indicators.macd[-len(df):].values
        features_df['atr'] = indicators.atr[-len(df):].values
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
            logger.warning(f"Нет индикаторов для {symbol}")
            return
        features = await self.prepare_lstm_features(symbol, indicators)
        if len(features) < self.config['lstm_timesteps'] * 2:
            logger.warning(f"Недостаточно данных для обучения {symbol}")
            return
        X = np.array([features[i:i + self.config['lstm_timesteps']] for i in range(len(features) - self.config['lstm_timesteps'])])
        price_now = features[:-self.config['lstm_timesteps'], 0]
        future_price = features[self.config['lstm_timesteps']:, 0]
        pct_change = (future_price - price_now) / np.clip(price_now, 1e-6, None)
        thr = self.config.get('target_change_threshold', 0.001)
        y = (pct_change > thr).astype(np.float32)
        model_state, val_preds, val_labels = await _train_lstm_remote.remote(X, y, self.config['lstm_batch_size'])
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
        self.lstm_models[symbol] = model
        self.last_retrain_time[symbol] = time.time()
        self.save_state()
        await self.compute_shap_values(symbol, model, X)
        logger.info(f"Модель CNN-LSTM обучена для {symbol}, Brier={brier:.4f}")
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
            except Exception as e:
                logger.error(f"Ошибка цикла обучения: {e}")
                await asyncio.sleep(60)

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
            cache_file = os.path.join(self.cache.cache_dir, f"shap_{symbol}.pkl")
            last_time = self.shap_cache_times.get(symbol, 0)
            if time.time() - last_time < self.shap_cache_duration:
                return
            sample = torch.tensor(X[:50], dtype=torch.float32, device=self.device)
            explainer = shap.DeepExplainer(model, sample)
            values = explainer.shap_values(sample)
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
            logger.info(f"SHAP значения для {symbol}: {top_feats}")
            await self.data_handler.telegram_logger.send_telegram_message(
                f"🔍 SHAP {symbol}: {top_feats}")
        except Exception as e:
            logger.error(f"Ошибка вычисления SHAP для {symbol}: {e}")

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
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            model.eval()
            with torch.no_grad():
                preds = model(X_tensor).squeeze().cpu().numpy()
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
            logger.error(f"Ошибка бектеста {symbol}: {e}")
            return None

    async def backtest_all(self):
        results = {}
        for symbol in self.data_handler.usdt_pairs:
            sharpe = await self.simple_backtest(symbol)
            if sharpe is not None:
                results[symbol] = sharpe
        self.last_backtest_time = time.time()
        return results
