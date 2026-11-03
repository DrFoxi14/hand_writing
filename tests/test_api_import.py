def test_api_module_imports():
    from hie.api.main import app
    assert app.title == "Handwriting Identity Engine"
