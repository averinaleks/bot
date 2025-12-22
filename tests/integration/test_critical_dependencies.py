"""Integration smoke tests for critical third-party dependencies.

These tests execute lightweight scenarios with the libraries that are most
likely to break our trading stack when they release new major versions.  By
keeping the scenarios small we can run them as part of the integration suite
and surface incompatibilities long before packaging a release.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_ccxt_smoke_handles_basic_exchange_flow() -> None:
    ccxt = pytest.importorskip("ccxt")

    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        assert exchange.id == "binance"
        # ``has`` should expose capability flags; verifying a well-known key
        # confirms the version still exposes the expected structure.
        assert "fetchTicker" in exchange.has
    finally:
        if hasattr(exchange, "close"):
            exchange.close()


@pytest.mark.integration
def test_stable_baselines3_can_train_and_predict() -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    gymnasium = pytest.importorskip("gymnasium")

    if getattr(sb3, "__model_builder_stub__", False) or getattr(
        gymnasium, "__model_builder_stub__", False
    ):
        pytest.skip("stable-baselines3 or gymnasium is stubbed in the test environment")

    if not hasattr(gymnasium, "make"):
        pytest.skip("gymnasium stub without make() detected; skipping dependency check")

    env = gymnasium.make("CartPole-v1")
    try:
        model = sb3.PPO(
            "MlpPolicy",
            env,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            learning_rate=3e-4,
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=64)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=False)

        # Stable-Baselines always returns a numpy array.  Normalising it lets us
        # use the environment's validation helper without relying on the exact
        # shape representation across releases.
        action_arr = np.asarray(action)
        if action_arr.shape == ():
            action_value = action_arr.item()
        else:
            action_value = action_arr.squeeze().item()
        assert env.action_space.contains(int(action_value))
    finally:
        env.close()


@pytest.mark.integration
def test_transformers_forward_pass_remains_compatible() -> None:
    torch = pytest.importorskip("torch")
    bert_config_mod = pytest.importorskip("transformers.models.bert.configuration_bert")
    bert_model_mod = pytest.importorskip("transformers.models.bert.modeling_bert")

    config = bert_config_mod.BertConfig(
        vocab_size=30522,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_labels=2,
    )

    model = bert_model_mod.BertForSequenceClassification(config)

    input_ids = torch.ones((2, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (2, config.num_labels)
