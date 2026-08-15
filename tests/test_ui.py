from ui.app_ui import build_app


def test_gradio_builds():
    app = build_app()
    assert app is not None
