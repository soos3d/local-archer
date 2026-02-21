"""Shared test fixtures for Archer test suite."""

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_data():
    """Ensure NLTK punkt_tab is available for tokenization tests."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@pytest.fixture
def default_config():
    """ArcherConfig with all defaults."""
    from archer.core.config import ArcherConfig

    return ArcherConfig()


@pytest.fixture
def archer_config_with_emotions():
    """ArcherConfig with emotional keywords configured."""
    from archer.core.config import ArcherConfig, EmotionConfig, PersonalityConfig

    config = ArcherConfig()
    config.personality = PersonalityConfig(
        name="Archer",
        system_prompt="You are a helpful AI assistant.",
        emotional_keywords={
            "positive": ["amazing", "wonderful", "great"],
            "negative": ["terrible", "awful"],
            "emphasis": ["obviously", "absolutely"],
        },
        emotion=EmotionConfig(
            base_exaggeration=0.5,
            max_exaggeration=0.9,
            min_exaggeration=0.3,
            keyword_boost=0.1,
        ),
    )
    return config


@pytest.fixture
def sample_audio() -> np.ndarray:
    """2-second float32 audio at 16kHz with audible signal (440Hz tone)."""
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def silent_audio() -> np.ndarray:
    """2-second near-silent audio (below silence threshold)."""
    return np.zeros(32000, dtype=np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """0.2-second audio (below min_audio_duration of 0.5s)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(3200) * 0.5).astype(np.float32)


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with personalities/ subdirectory."""
    personalities_dir = tmp_path / "personalities"
    personalities_dir.mkdir()
    return tmp_path
