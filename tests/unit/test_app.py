"""Unit tests for app.py (CLI entry point)."""

from unittest.mock import MagicMock, patch

import app as app_module


def _run_main(args: list[str]):
    """Helper: run main() with given CLI args, mocking load_config and Assistant."""
    with (
        patch("app.load_config") as mock_load_config,
        patch("app.Assistant") as MockAssistant,
        patch("sys.argv", ["app"] + args),
    ):
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        mock_assistant_instance = MagicMock()
        MockAssistant.return_value = mock_assistant_instance

        app_module.main()

    return mock_load_config, MockAssistant, mock_config, mock_assistant_instance


class TestMainCLI:
    def test_default_config_path(self):
        mock_load_config, _, _, _ = _run_main([])
        mock_load_config.assert_called_once_with("config/default.yaml")

    def test_custom_config_path(self):
        mock_load_config, _, _, _ = _run_main(["--config", "custom.yaml"])
        mock_load_config.assert_called_once_with("custom.yaml")

    def test_voice_override(self):
        _, _, mock_config, _ = _run_main(["--voice", "path/to/voice.wav"])
        assert mock_config.tts.voice_sample == "path/to/voice.wav"

    def test_model_override(self):
        _, _, mock_config, _ = _run_main(["--model", "llama3"])
        assert mock_config.llm.model == "llama3"

    def test_exaggeration_override(self):
        _, _, mock_config, _ = _run_main(["--exaggeration", "0.7"])
        assert mock_config.personality.emotion.base_exaggeration == 0.7

    def test_save_voice_flag(self):
        _, _, mock_config, _ = _run_main(["--save-voice"])
        assert mock_config.tts.save_responses is True

    def test_assistant_run_called(self):
        _, _, _, mock_assistant = _run_main([])
        mock_assistant.run.assert_called_once()

    def test_no_voice_override_when_not_specified(self):
        _, _, mock_config, _ = _run_main([])
        # voice_sample should not have been explicitly set
        assert mock_config.tts.voice_sample != "path/to/voice.wav"
