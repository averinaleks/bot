"""Model architecture definitions for PyTorch and Keras."""

from __future__ import annotations

import os
import numpy as np

KERAS_FRAMEWORKS = {"tensorflow", "keras"}
PYTORCH_FRAMEWORKS = {"pytorch", "lightning"}

MLP_PARAMS = {"hidden_sizes": (128, 64), "dropout": 0.2, "l2_lambda": 1e-5}
GRU_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "conv_channels": 32,
    "kernel_size": 3,
    "l2_lambda": 1e-5,
}
TRANSFORMER_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "l2_lambda": 1e-5,
}


def _torch_architectures():
    import torch
    import torch.nn as nn

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
            self,
            input_size,
            hidden_sizes=(128, 64),
            dropout=0.2,
            l2_lambda=1e-5,
            regression=False,
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

    return Net, CNNGRU, TemporalFusionTransformer


def create_model(model_type: str, framework: str, input_size: int, regression: bool = False):
    """Create a model instance for the given framework and type."""
    fw = framework.lower()
    if fw in PYTORCH_FRAMEWORKS:
        Net, CNNGRU, TFT = _torch_architectures()
        if model_type == "mlp":
            return Net(
                input_size,
                hidden_sizes=MLP_PARAMS["hidden_sizes"],
                dropout=MLP_PARAMS["dropout"],
                l2_lambda=MLP_PARAMS["l2_lambda"],
                regression=regression,
            )
        if model_type == "gru":
            return CNNGRU(
                input_size,
                GRU_PARAMS["hidden_size"],
                GRU_PARAMS["num_layers"],
                GRU_PARAMS["dropout"],
                conv_channels=GRU_PARAMS["conv_channels"],
                kernel_size=GRU_PARAMS["kernel_size"],
                l2_lambda=GRU_PARAMS["l2_lambda"],
                regression=regression,
            )
        return TFT(
            input_size,
            d_model=TRANSFORMER_PARAMS["d_model"],
            nhead=TRANSFORMER_PARAMS["nhead"],
            num_layers=TRANSFORMER_PARAMS["num_layers"],
            dropout=TRANSFORMER_PARAMS["dropout"],
            l2_lambda=TRANSFORMER_PARAMS["l2_lambda"],
            regression=regression,
        )

    if fw in KERAS_FRAMEWORKS:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow import keras

        if model_type == "mlp":
            params = MLP_PARAMS
            inputs = keras.Input(shape=(input_size,))
            x = inputs
            for hs in params["hidden_sizes"]:
                x = keras.layers.Dense(hs, activation="relu")(x)
                x = keras.layers.Dropout(params["dropout"])(x)
            activation = "linear" if regression else "sigmoid"
            outputs = keras.layers.Dense(1, activation=activation)(x)
            return keras.Model(inputs, outputs)

        inputs = keras.Input(shape=(None, input_size))
        x = keras.layers.Conv1D(
            GRU_PARAMS["conv_channels"],
            GRU_PARAMS["kernel_size"],
            padding="same",
            activation="relu",
        )(inputs)
        x = keras.layers.Dropout(GRU_PARAMS["dropout"])(x)
        if model_type == "gru":
            x = keras.layers.GRU(GRU_PARAMS["hidden_size"], return_sequences=True)(x)
            attn = keras.layers.Dense(1, activation="softmax")(x)
            x = keras.layers.Multiply()([x, attn])
            x = keras.layers.Lambda(lambda t: keras.backend.sum(t, axis=1))(x)
        else:
            attn = keras.layers.MultiHeadAttention(
                num_heads=TRANSFORMER_PARAMS["nhead"],
                key_dim=TRANSFORMER_PARAMS["d_model"] // TRANSFORMER_PARAMS["nhead"],
            )(x, x)
            x = keras.layers.GlobalAveragePooling1D()(attn)
        activation = "linear" if regression else "sigmoid"
        outputs = keras.layers.Dense(1, activation=activation)(x)
        return keras.Model(inputs, outputs)

    raise ValueError(f"Unsupported framework: {framework}")
