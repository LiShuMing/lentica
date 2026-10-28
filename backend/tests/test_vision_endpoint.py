import pytest

from backend.app.models.vision_service import validate_local_endpoint


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8100", "http://127.0.0.2:8100/", "http://[::1]:8100", "http://localhost:8100"])
def test_local_vision_endpoint(endpoint):
    assert validate_local_endpoint(endpoint) == endpoint.rstrip("/")


@pytest.mark.parametrize("endpoint", ["https://example.com", "http://192.168.1.2:8100", "file:///tmp/model", "http://user:secret@localhost:8100"])
def test_remote_or_credentialed_vision_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        validate_local_endpoint(endpoint)
