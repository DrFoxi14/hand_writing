from hie.device import detect_device

def test_device_report_has_valid_fallback():
    result=detect_device()
    assert result["device"] in {"cpu","cuda","mps"}
    assert isinstance(result["cuda"],bool) and isinstance(result["mps"],bool)
