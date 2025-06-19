import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
import optuna
import joblib
import os
import time
from utils import logger, check_dataframe_empty, HistoricalDataCache
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import psutil
import shutil

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, l2_lambda=1e-5):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def l2_regularization(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_lambda * l2_norm

class RiskxLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, l2_lambda=1e-5):
        super(RiskxLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

    def l2_regularization(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_lambda * l2_norm

def create_xlstm_model(input_shape, units, layers, dropout_rate, learning_rate):
    try:
        model = xLSTM(input_shape[-1], units, layers, dropout_rate)
        model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        return model, optimizer
    except Exception as e:
        logger.error(f"Ошибка создания xLSTM модели: {e}")
        return None, None

def create_risk_xlstm_model(input_shape, units, layers, dropout_rate, learning_rate):
    try:
        model = RiskxLSTM(input_shape[-1], units, layers, dropout_rate)
        model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        return model, optimizer
    except Exception as e:
        logger.error(f"Ошибка создания RiskxLSTM модели: {e}")
        return None, None

class ModelBuilder:
    def __init__(self, config, data_handler, trade_manager):
        self.config = config
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.lstm_models = {}
        self.risk_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.cache = HistoricalDataCache(config['cache_dir'])
        self.cache_duration = config.get('shap_cache_duration', 86400 * 7)
        self.scalers = {}
        self.feature_selectors = {}
        self.last_volatility = {symbol: 0.0 for symbol in data_handler.usdt_pairs}
        self.state_file = os.path.join(config['cache_dir'], 'model_builder_state.pkl')
        self.pretrained_model = None
        self.pretrained_optimizer = None
        self.last_retrain_time = {symbol: 0 for symbol in data_handler.usdt_pairs}
        self.last_save_time = time.time()
        self.save_interval = 900
        self.feature_data_limit = 500

    def save_state(self):
        if time.time() - self.last_save_time < self.save_interval:
            return
        try:
            disk_usage = shutil.disk_usage(self.config['cache_dir'])
            if disk_usage.free / (1024 ** 3) < 0.5:
                logger.warning(f"Недостаточно места для сохранения состояния: {disk_usage.free / (1024 ** 3):.2f} ГБ")
                return
            state = {
                'lstm_models': {k: v.state_dict() for k, v in self.lstm_models.items()},
                'risk_models': {k: v.state_dict() for k, v in self.risk_models.items()},
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'last_volatility': self.last_volatility,
                'pretrained_model': self.pretrained_model.state_dict() if self.pretrained_model else None,
                'last_retrain_time': self.last_retrain_time
            }
            with open(self.state_file, 'wb') as f:
                joblib.dump(state, f)
            self.last_save_time = time.time()
            logger.info("Состояние ModelBuilder сохранено")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state = joblib.load(f)
                self.scalers = state['scalers']
                self.feature_selectors = state.get('feature_selectors', {})
                self.last_volatility = state['last_volatility']
                self.last_retrain_time = state.get('last_retrain_time', {symbol: 0 for symbol in self.data_handler.usdt_pairs})
                if state.get('pretrained_model'):
                    self.pretrained_model = xLSTM(self.config['lstm_timesteps'], 64, 2, 0.2)
                    self.pretrained_model.load_state_dict(state['pretrained_model'])
                    self.pretrained_model.to(self.device)
                logger.info("Состояние ModelBuilder загружено")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

    async def preprocess(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        try:
            if check_dataframe_empty(df, f"preprocess {symbol}"):
                return pd.DataFrame()
            df = df.sort_index()
            if df.isna().sum().sum() / df.size > 0.1:
                logger.warning(f"Слишком много пропусков в данных для {symbol}, интерполяция отменена")
                return pd.DataFrame()
            df = df.interpolate(method='time', limit_direction='both')
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            max_gap = pd.Timedelta(self.config['timeframe']).total_seconds() * 2
            if time_diffs.max() > max_gap:
                logger.warning(f"Обнаружен разрыв в данных для {symbol}: {time_diffs.max()/60:.2f} минут")
                await self.data_handler.telegram_logger.send_telegram_message(
                    f"⚠️ Разрыв в данных для {symbol}: {time_diffs.max()/60:.2f} минут"
                )
                return pd.DataFrame()
            return df.tail(self.config['min_data_length'])
        except Exception as e:
            logger.error(f"Ошибка предобработки данных для {symbol}: {e}")
            return pd.DataFrame()

    async def prepare_lstm_features(self, symbol, indicators):
        try:
            df = self.data_handler.ohlcv.xs(symbol, level='symbol', drop_level=False) if symbol in self.data_handler.ohlcv.index.get_level_values('symbol') else None
            if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
                return np.array([])
            df = await self.preprocess(df.droplevel('symbol'), symbol)
            if check_dataframe_empty(df, f"prepare_lstm_features {symbol}"):
                return np.array([])
            cache_key = f"{symbol}_features"
            cached_features = self.cache.load_cached_data(symbol, cache_key)
            cache_file = os.path.join(self.cache.cache_dir, f"{cache_key}.pkl.gz")
            if cached_features is not None and os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file) < self.cache_duration):
                logger.info(f"Features loaded from cache for {symbol}")
                return cached_features
            volatility = df['close'].pct_change().std() if not df.empty else 0.02
            cache = indicators
            if cache.ema30 is None or len(cache.ema30) == 0:
                logger.warning(f"Ошибка расчета индикаторов для {symbol}")
                return np.array([])
            funding = np.float32(self.data_handler.funding_rates.get(symbol, 0.0))
            oi = np.float64(self.data_handler.open_interest.get(symbol, 0.0))
            orderbook_rows = self.data_handler.orderbook[self.data_handler.orderbook['symbol'] == symbol]
            orderbook_imbalance = orderbook_rows['imbalance'].iloc[-1] if not orderbook_rows.empty else 0.0
            volume_profile = cache.volume_profile.iloc[-1] if cache.volume_profile is not None else 0.0
            btc_eth_ratio = 0.0
            try:
                btc_df = self.data_handler.ohlcv.xs('BTC/USDT', level='symbol', drop_level=False) if 'BTC/USDT' in self.data_handler.ohlcv.index.get_level_values('symbol') else None
                eth_df = self.data_handler.ohlcv.xs('ETH/USDT', level='symbol', drop_level=False) if 'ETH/USDT' in self.data_handler.ohlcv.index.get_level_values('symbol') else None
                if btc_df is not None and eth_df is not None and not btc_df.empty and not eth_df.empty:
                    btc_price = btc_df['close'].tail(len(df))
                    eth_price = eth_df['close'].tail(len(df))
                    if len(btc_price) == len(eth_price) and len(btc_price) > 0:
                        btc_eth_ratio = btc_price / eth_price
                        btc_eth_ratio = np.float32(btc_eth_ratio.mean() if not np.isnan(btc_eth_ratio).all() else 0.0)
            except Exception as e:
                logger.warning(f"Ошибка расчета BTC/ETH ratio for {symbol}: {e}")
            all_features = {
                'ema30': cache.ema30,
                'ema100': cache.ema100,
                'ema200': cache.ema200,
                'atr': cache.atr,
                'rsi': cache.rsi,
                'adx': cache.adx,
                'macd': cache.macd,
                'volume': np.log1p(df['volume']).astype(np.float32),
                'funding_rate': np.full(len(df), funding, dtype=np.float32),
                'open_interest': np.full(len(df), np.log1p(oi), dtype=np.float32),
                'orderbook_imbalance': np.full(len(df), np.float32(orderbook_imbalance), dtype=np.float32),
                'volatility': np.full(len(df), np.float32(volatility), dtype=np.float32),
                'volume_profile': np.full(len(df), volume_profile, dtype=np.float32),
                'btc_eth_ratio': np.full(len(df), btc_eth_ratio, dtype=np.float32),
            }
            features = pd.DataFrame(all_features, index=df.index)
            features = features.interpolate(method='time', limit_direction='both')
            if check_dataframe_empty(features, f"prepare_lstm_features {symbol} post-fill"):
                return np.array([])
            dynamic_features = [col for col in features.columns if not (features[col] == features[col].iloc[0]).all()]
            if len(dynamic_features) < 2:
                logger.warning(f"Too few dynamic features for {symbol}: {dynamic_features}")
                return np.array([])
            corr_matrix = features[dynamic_features].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
            features = features.drop(columns=to_drop)
            dynamic_features = [col for col in features.columns if not (features[col] == features[col].iloc[0]).all()]
            if dynamic_features:
                vif_data = pd.DataFrame()
                vif_data['feature'] = dynamic_features
                vif_data['VIF'] = [variance_inflation_factor(features[dynamic_features].values, i) for i in range(len(dynamic_features))]
                high_vif = vif_data[vif_data['VIF'] > 5]['feature'].tolist()
                features = features.drop(columns=high_vif)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalerscalers[symbol] = scaler
            memory = {'available': psutil.virtual_memory().available / (1024 ** 3)}
            if memory['available'] < 2:
                logger.warning(f"Insufficient memory для {symbol}, используем все признаки")
                return scaled_features
            try:
                feature_data = scaled_features[-min(self.feature_data_limit, len(scaled_features)):]
                feature_data_y = (feature_data[1:, 0] > feature_data[:-1, 0]).astype(np.int32)
                if len(feature_data_y) < len(feature_data):
                    feature_data = feature_data[:-1]
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(feature_data, feature_data_y)
                importances = rf.feature_importances_
                top_indices = np.argsort(importances)[-int(0.8 * len(importances)):]
                selected_features = scaled_features[:, top_indices]
                self.feature_selectors[symbol] = top_indices
                self.cache.save_cached_data(symbol, cache_key, selected_features)
                logger.info(f"Feature importance computed для {symbol}")
                return selected_features
            except Exception as e:
                logger.warning(f"Ошибка в RandomForest для {symbol}: {e}, используем все признаки")
                return scaled_features
        except Exception as e:
            logger.error(f"Ошибка подготовки признаков для {symbol}: {e}")
            return np.array([])

    def optimize_xlstm_params(self, trial, X, labels, input_shape):
        units = trial.suggest_int('units', 32, 128)
        layers = trial.suggest_int('layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-4, log=True)
        batch_size = self.config['lstm_batch_size']
        if torch.cuda.memory_allocated() > 0:
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory() / (1024 ** 3)
            if vram_used / vram_total > 0.8:
                batch_size = max(16, batch_size // 2)
                logger.warning(f"Высокая загрузка VRAM ({vram_used:.2f}/{vram_total:.2f} GB), уменьшен batch_size до {batch_size}")
')
        tscv = TimeSeriesSplit(n_splits=5)
        val_losses = []
        for train_idx, val_idx in tscv.split(X):
            X_train = X[train_idx], X[val_idx]
            y_train = labels[train_idx], labels[val_idx]
            model, optimizer = create_xlstm_model((X_train.shape[1], X_train.shape[2]), units, layers, dropout_rate, learning_rate])
            if not model:
                return None
            model.to(self.device)
            criterion = nn.BCELoss()(')
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
            best_val_loss = float('inf')
            patience_counter = 0
            scaler = torch.cuda.amp.GradScaler()
            for epoch in range(30):
                model.train()()
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()()
                    with torch.cuda.amp.autocast()():
                        outputs = model(batch_X_batch).squeeze()()
                        loss = criterion(outputs, batch_y) + (model.l2_regular())
                    scaler.scale(loss)().backward()
                    scaler.step(optimizer)
                    scaler.update()()
                    total_loss += += loss.item()()
                model.eval()()
                with torch.no_grad():
                    with torch.cuda.amp.autocast()():
                        val_outputs = model(X_val_tensor()).squeeze().detach()()
                        val_loss = criterion(val_outputs, y_val_tensor).item()()
                scheduler.step(val_loss)()
                if val_loss < best_val_loss:
                    best_val_loss = = val_loss
                    patience_counter = = 0
                else:
                    patience_counter += += 1
                if patience_counter >= >= patience:
                    break
            val_losses.append(best_val_loss)
            torch.cuda.empty_cache()()
        return -np.mean(val_loss_losses)

    def optimize_risk_xlstm_params(self, trial, X, labels):
        units = trial.suggest_int('units', 32, 128)
        layers = 64
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-6, lambda1e-4, log=True)

        batch_size = self._config['lstm_batch_size']
        if torch.cuda.memory_allocated() > 0:
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory() / (1024 ** 3)
            if vram_used / vram_total > 0.8:
                batch_size = max(16, batch_size // 2)
                logger.warning(f"Высокая загрузка VRAM ({vram_used:.2f}/{vram_total:.2f} GB), уменьшен batch_size до {batch_size}")
')
        tscv = TimeSeriesSplit(n_splits=5)
        val_accuracies = []
        for train_idx, val_idx in tscv.items.split(X):
            X_train = X[train_idx], X[val_idx]
            y_train = labels[train_idx], labels[val_idx]
            model, optimizer = create_risk_xlstm_model((X_train.shape[1], X_train.shape[2]), units, layers, dropout_rate, learning_rate)
            if model is None:
                return 0.0
            model.to(self.device)
            criterion = nn.CrossEntropyLoss()()
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(y_train).cuda(self.device)
            X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val = torch.tensor(y_val, dtype=torch.long, device=self.device)
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=batch_size], shuffle=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5)
            best_val_accuracy = 0.0
            patience_counter = 0
            scaler = torch.cuda.amp.GradScaler()()
            for epoch in range(30):
                model.train()()
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()()
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch) + model.l2_regular()
                    scaler.scale(loss)().backward()()
                    scaler.step(optimizer)
                    scaler.update()()
                    total_loss += += loss.item()()
                model.eval()()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        outputs = model(X_val_tensor)
                        _, predicted = outputs.max(1).max(1)
                        accuracy = (predicted == y_val_tensor).float().mean().item()()
                scheduler.step(accuracy)
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += += 1
                if patience_counter >= >= patience:
                    break
            val_accuracies.append(best_val_accuracy)
            torch.cuda.empty_cache()()
        return np.mean(val_accuracies)

    async def pretrain_base_model(self):
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        try:
            all_features = []
            all_labels = []
            for symbol in self.symbols:
                indicators = self.data_handler.indicators.get(symbol)
                if not indicators:
                    logger.warning(f"Нет индикаторов для {symbol}, пропуск")
                    continue
                features = await self.features.prepare_lstm_features(symbols, indicators)
                if len(features) < self.config['lstm_timesteps'] * 2:
                    logger.warning(f"Недостаточно данных для {symbol}")
                    continue
                X = np.array([features[i:i + self.config['lstm_timesteps']] for i in range(len(features) - self.config['lstm_timesteps'])])
                labels = (features[self.config['lstm_timesteps']:, 0] > features[:-self.config['lstm_timesteps], 0]).astype(np.float32)
                if len(X) < 20:
                    logger.warning(f"Недостаточно данных для обучения xLSTM для {symbol}")
                    continue
                all_features.append(X)
                all_labels.append(labels)
                if not all_features:
                    logger.warning("Нет данных для предварительного обучения")
                    return
                X = np.concatenate(all_features, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
                study.optimize(lambda trial: self.optimize_xlstm_params(trial, X, labels, X.shape), n_trials=50)
                best_params = study.best_params()
                model, optimizer = create_xlstm_model((X.shape[1], X.shape[2]), best_params['units'], best_params['layers'], best_params['dropout_rate'], best_params['learning_rate'])
                if model is not None:
                    self.pretrained_model = model.to(self.device)
                    self.pretrained_optimizer = optimizer
                    logger.info(f"Предобученная модель xLSTM создана для {symbols}")
                    torch.cuda.empty_cache()()
        except Exception:
 as e:
            logger.error(f"Ошибка при предобучении модели: {e}")
            torch.cuda.empty_cache()

    async def online_update(self, symbol, str, X_new, labels_new):
        try:
            try:
                if not symbol in self._lstm_models:
                    logger.warning(f"Модель для {symbol} не найдена, пропускаем обновление")
                    return
                model = self.lstm_models[symbol]
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.stepLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
                criterion = nn.BCELoss()()
                X_tensor = torch.tensor(X_new, dtype=torch.float32, device=self.device)
                y_tensor = torch.tensor(labels_new, dtype=torch.float32, device=self.device)
                dataset = TensorDataset(X_train_tensor, y_train_tensor)
                loader = DataLoader(dataset, batch_size=self._config['lstm_batch_size'], shuffle=True)
                indicators = self.data_handler.indicators.get(symbol)
                volatility = indicators.volatility if indicators else 0.02
                num_epochs = max(3, min(10, int(3 / (1 + volatility / 0.02))))
                model.train()()
                scaler = torch.cuda.amp.GradScaler()()
                for epoch in range(num_epochs):
                    total_loss = 0
                    for batch_X, batch_y in loader:
                        optimizer.zero_grad()()
                        with torch.cuda.amp.autocast():
                            outputs = model(X_batch).squeeze()
                            loss = criterion(outputs, y_batch) + model.l2_regular()()
                        scaler.scale(loss)().backward()()
                        scaler.step(optimizer)
                        scaler.update()()
                        total_loss += += loss.item()()
                    scheduler.step(total_loss / len(loader))
                logger.info(f"Онлайн обновление для {symbol} завершено с {num_epochs} эпохами")
                self.lstm_models[symbol] = = model
                self.save_state()
                torch.cuda.empty_cache()()
        except Exception as e:
            logger.error(f"Ошибка онлайн обновления для {symbol}: {e}")
            torch.cuda.empty_cache()()

    async def retrain_symbol(self, symbol):
        try:
            indicators = self.data_handler.indicators.get(symbol)
            if not indicators or check_dataframe_empty(indicators.df, f"retrain {symbol}"):
                logger.warning(f"No indicators for retraining on {symbol}")
                return None
            features = await self.prepare_lstm_features(symbol, indicators)
            if len(features) < self.config['lstm_timesteps'] * 2:
                logger.warning(f"Insufficient features for training on {symbol}")
                return None
            current_volatility = indicators.volatility
            self.last_volatility[symbol] = current_volatility
            X = np.array([features[i:i+self.config['lstm_timesteps']] for i in range(len(features) - self.config['lstm_timesteps'])])
            y = (features[self.config['lstm_timesteps']:, 0] > features[:-self.config['lstm_timesteps'], 0]).astype(np.float32)
            if len(X) < 20 and self.pretrained_model:
                logger.warning(f"Insufficient data for {symbol} ({len(X)} samples), using transfer learning")
                model = xLSTM(self.config['lstm_timesteps'], 64, 2, 0.2)
                model.load_state_dict(self.pretrained_model.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
                model.to(self.device)
                criterion = nn.BCELoss()()
                dataset = TensorDataset(TensorDataset(torch.tensor(X), torch.tensor(y)), dtype=torch.float32, device=self.device)
                loader = DataLoader(dataset, batch_size=self.config['lstm_batch_size'], shuffle=False)
                scaler = torch.cuda.amp.GradScaler()()
                for epoch in range(30):10:
                    model.train()()
                    for batch_X, batch_y in loader:
                        optimizer.zero_grad()()
                        with torch.cuda.amp.autocast()():
                            outputs = model(X_batch).squeeze()()
                            loss = criterion(outputs), y_batch) + (model.l2_regular())()
                        scaler.scale(loss)().backward()()
                        scaler.step(optimizer)()
                            scaler.step(optimizer)
                        scaler.update()()
                    self.lstm_models[symbol] = model
                logger.info(f"Transfer learning completed for {symbol}")
            else:
                n_splits = 5 if len(X) >= else 50 else 2
                train_size = int(0.6 * len(X))
                test_size = int(0.2 * len(X))
                best_model = None
                best_optimizer = None
                best_val_loss = float('inf')
                tscv = TimeSeriesSplit(n_splits=n_splits)
                for train_idx, val_idx in tscv.split(X):
                    X_train = X[train_idx[start_idx:start_idx + train_size]
                    y_train = y[train_idx[start_idx:start_idx + train_size]
                    X_val = X[val_idx + train_size:end_idx]
                    y_val = y[val_idx + end_idx]
                    if len(X_val) < 2:
                        logger.warning(f"Too small validation data for {symbol} ({len(X_val)} samples), skipping fold")
                        continue
                    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
                    study.optimize(lambda trial: self.optimize_xlstm_params(trial, X_train, y_train, X_train.shape), n_trials=50)
                    best_params = study.best_params()
                    model, optimizer = create_xlstm_model(
                        (X_train.shape[1], X_train.shape[2]),
                        best_params['units'],
                        best_params['layers'],
                        best_params['dropout_rate'],
                        best_params['learning_rate']
                    )
                    if model is None:
                        continue
                    model.to(self.device)
                    criterion = nn.BCELoss()
                    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32, device=self.device), torch.tensor(y_train, dtype=torch.float32, device=self.device))
                    loader = DataLoader(dataset, batch_size=self.config['lstm_batch_size'], shuffle=False)
                    patience = 5
                    patience_counter = 0
                    scaler = torch.cuda.amp.GradScaler()
                    for epoch in range(30):
                        total_loss = 0
                        for X_batch, y_batch in loader:
                            optimizer.zero_grad()
                            with torch.cuda.amp.autocast():
                                outputs = model(X_batch).squeeze()
                                loss = criterion(outputs, y_batch) + model.l2_regularization()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            total_loss += loss.item()
                        model.eval()
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                val_outputs = model(torch.tensor(X_val, dtype=torch.float32, device=self.device)).squeeze()
                                val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.float32, device=self.device)).item()
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model = model
                                best_optimizer = optimizer
                                patience_counter = 0
                            else:
                                patience_counter += 1
                            if patience_counter >= patience:
                                break
                    torch.cuda.empty_cache()
                if best_model:
                    self.lstm_models[symbol] = best_model
                    logger.info(f"xLSTM model retrained for {symbol} with validation loss {best_val_loss:.2f}")
                    recent_features = features[-self._config['lstm_timesteps'] * 2:]
                    if len(recent_features) >= self._config['lstm_timesteps']:
                        X_new = np.array([recent_features[i:i + self.config['lstm_timesteps']] for i in range(len(recent_features) - self.config['lstm_timesteps'])]))
                        y_new = (recent_features[self._config['lstm_timesteps']:, 0] > recent_features[:-self.config['lstm_timesteps'], 0]).astype(np.float32)
                        if len(X_new) > 0:
                            await self.online_update(symbol, X_new, y_new)
            y_risk = np.zeros(len(y_new), dtype=np.int64)
            y_risk[y_new > 0.5] = 1
            study_risk = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study_risk.optimize(lambda trial: self.optimize_risk_xlstm_params(trial, X_new, y_risk), n_trials=50)
            best_risk_params = study_risk.best_params()
            risk_model, risk_optimizer = create_risk_xlstm_model(
                (X.shape[1], X.shape[2]),
                best_risk_params['units'],
                best_risk_params['layers'],
                best_risk_params['dropout_rate'],
                best_risk_params['learning_rate']
            })
            if risk_model:
                risk_model.to(self.device)
                criterion_risk = nn.CrossEntropyLoss()
                dataset_risk = TensorDataset(
                    torch.tensor(X, dtype=torch.float32, device=self.device),
                    torch.tensor(y_risk, dtype=torch.long, device=self.device)
                )
                loader_risk = DataLoader(dataset_risk, batch_size=self.config['lstm_batch_size'], shuffle=False)
                scaler = torch.cuda.amp.GradScaler()
                for epoch in range(30):
                    risk_model.train()
                    for X_batch, y_batch in loader_risk:
                        risk_optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            outputs = risk_model(X_batch)
                            loss = criterion_risk(outputs, y_batch) + (risk_model.l2_regularization())
                        scaler.scale(loss).backward()
                        scaler.step(risk_optimizer)
                        scaler.update()
                self.risk_models[symbol] = risk_model
                logger.info(f"RiskxLSTM model retrained for {symbol}")
            self.last_retrain_time[symbol] = time.time()
            self.save_state()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error retraining {symbol}: {e}")
            torch.cuda.empty_cache()

    async def train(self):
        self.load_state()
        await self.pretrain_base_model()
        while True:
            try:
                tasks = []
                for symbol in self.data_handler.usdt_pairs:
                    if time.time() - self.last_retrain_time[symbol] < self.config['retrain_interval']:
                        logger.debug(f"Skipping retrain for {symbol}, next in {self.config['retrain_interval'] - (time.time() - self.last_retrain_time[symbol]):.2f} seconds")
                        continue
                    await self.retrain_symbol(symbol)
                await asyncio.sleep(self.config['retrain_interval'] // len(self.data_handler.usdt_pairs))
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await self.data_handler.asyncio.sleep(60)
                torch.cuda.empty_cache()

    async def adjust_thresholds(self, symbol):
        try:
            if symbol not in self.risk_models:
                logger.warning(f"RiskxLSTM model not found for {symbol}, default thresholds used")
                return 0.6, 0.4
            indicators = self.data_handler.indicators.get(symbol)
            if not indicators or check_dataframe_empty(indicators.df, f"adjust_thresholds {symbol}"):
                logger.warning(f"No indicators for {symbol}, default thresholds used")
                return 0.6, 0.4
            volatility = indicators.volatility
            risk_features = await self.prepare_lstm_features(symbol, indicators)
            if len(risk_features) < self.config['lstm_timesteps']:
                logger.warning(f"Insufficient features for threshold adjustment for {symbol}")
                return 0.6, 0.4
            X_risk = np.array([risk_features[-self.config['lstm_timesteps']:]])
            X_tensor = torch.tensor(X_risk, dtype=torch.float32, device=self.device)
            model = self.risk_models[symbol]
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor).squeeze().cpu().numpy()
            long_threshold = 0.6 + 0.2 * (1 - outputs[1]) * (1 + volatility)
            short_threshold = 0.4 - 0.2 * outputs[0] * (1 + volatility)
            torch.cuda.empty_cache()
            return min(long_threshold, 0.9), max(short_threshold, 0.1)
        except Exception as e:
            logger.error(f"Error adjusting thresholds for {symbol}: {e}")
            return 0.6, 0.4