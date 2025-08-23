from config import BotConfig


def test_ws_batch_size_defaults_to_max_subscriptions():
    cfg = BotConfig(max_subscriptions_per_connection=7, ws_subscription_batch_size=None)
    assert cfg.ws_subscription_batch_size == 7


def test_ws_batch_size_respects_explicit_value():
    cfg = BotConfig(max_subscriptions_per_connection=7, ws_subscription_batch_size=3)
    assert cfg.ws_subscription_batch_size == 3

