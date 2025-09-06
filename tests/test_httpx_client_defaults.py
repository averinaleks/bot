from http_client import get_httpx_client


def test_get_httpx_client_trust_env_false():
    with get_httpx_client() as client:
        assert client.trust_env is False
