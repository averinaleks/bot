from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return {"ok": True}, 200

    return app


def test_health():
    app = create_app()
    client = app.test_client()
    rv = client.get("/health")
    assert rv.status_code == 200
    assert rv.get_json()["ok"] is True
