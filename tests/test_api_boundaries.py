from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE_DIRS = [PROJECT_ROOT / "src", PROJECT_ROOT / "pages"]
PYTHON_SOURCE_FILES = [PROJECT_ROOT / "app.py", PROJECT_ROOT / "data" / "generate_ticket_data.py"]
LIVE_INTEGRATIONS_PATH = PROJECT_ROOT / "src" / "live_market_integrations.py"
GENERATIVE_PATH = PROJECT_ROOT / "src" / "generative_explainer.py"


def _python_source_text() -> str:
    files = list(PYTHON_SOURCE_FILES)
    for directory in PYTHON_SOURCE_DIRS:
        files.extend(directory.glob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in files if path.exists())


def test_live_market_api_clients_are_isolated_to_integration_layer():
    live_source = LIVE_INTEGRATIONS_PATH.read_text(encoding="utf-8")
    assert "import requests" in live_source
    assert "app.ticketmaster.com" in live_source
    assert "api.seatgeek.com" in live_source
    assert "api.open-meteo.com" in live_source

    for path in list((PROJECT_ROOT / "src").glob("*.py")) + list((PROJECT_ROOT / "pages").glob("*.py")):
        if path == LIVE_INTEGRATIONS_PATH:
            continue
        source = path.read_text(encoding="utf-8")
        assert "import requests" not in source
        assert "from requests" not in source


def test_openai_is_explanation_only_boundary():
    generative_source = GENERATIVE_PATH.read_text(encoding="utf-8")
    live_source = LIVE_INTEGRATIONS_PATH.read_text(encoding="utf-8")

    assert "from openai import OpenAI" in generative_source
    assert "responses.create" in generative_source
    assert "OpenAI" not in live_source
    assert "responses.create" not in live_source
