"""Main Archer Assistant orchestrator."""

import os
import re
import threading
import time

from rich.console import Console

_STAGE_DIRECTION = re.compile(r"\([^)]*\)")

from archer.audio.pipeline import StreamingPipeline
from archer.audio.player import AudioPlayer
from archer.audio.recorder import AudioRecorder
from archer.audio.vad import SileroVAD
from archer.core.config import ArcherConfig, STTConfig, TTSConfig, LLMConfig, PersonalityConfig
from archer.llm.base import BaseLLM
from archer.llm.ollama_llm import OllamaLLM
from archer.stt.base import BaseSTT
from archer.stt.whisper_stt import WhisperSTT
from archer.tts.base import BaseTTS
from archer.tts.chatterbox_tts import ChatterboxTTS
from archer.tts.pocket_tts import PocketTTS

console = Console()


def create_stt(config: STTConfig) -> BaseSTT:
    """
    Factory function to create STT provider.

    Args:
        config: STT configuration.

    Returns:
        STT provider instance.
    """
    if config.provider == "whisper":
        stt = WhisperSTT(config)
        stt.load_model()
        return stt
    else:
        raise ValueError(f"Unknown STT provider: {config.provider}")


def create_tts(config: TTSConfig) -> BaseTTS:
    """
    Factory function to create TTS provider.

    Args:
        config: TTS configuration.

    Returns:
        TTS provider instance.
    """
    if config.provider == "chatterbox":
        return ChatterboxTTS(config)
    elif config.provider == "pocket_tts":
        return PocketTTS(config)
    else:
        raise ValueError(f"Unknown TTS provider: {config.provider}")


def create_llm(config: LLMConfig, personality: PersonalityConfig) -> BaseLLM:
    """
    Factory function to create LLM provider.

    Args:
        config: LLM configuration.
        personality: Personality configuration.

    Returns:
        LLM provider instance.
    """
    if config.provider == "ollama":
        return OllamaLLM(config, personality)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


class Assistant:
    """Main Archer Voice Assistant orchestrator."""

    def __init__(self, config: ArcherConfig):
        """
        Initialize the Archer Assistant.

        Args:
            config: Full Archer configuration.
        """
        self.config = config
        self.session_id = "archer_session"
        self.response_count = 0

        console.print(f"[cyan]🤖 {config.assistant_name} Voice Assistant")
        console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Initialize components
        console.print("[dim]Loading STT model...[/dim]")
        self.stt = create_stt(config.stt)

        console.print("[dim]Loading TTS model...[/dim]")
        self.tts = create_tts(config.tts)

        console.print("[dim]Initializing LLM...[/dim]")
        self.llm = create_llm(config.llm, config.personality)

        # Audio components
        self.recorder = AudioRecorder(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self.player = AudioPlayer()

        # VAD (loaded lazily only when needed)
        self.vad: SileroVAD | None = None
        if config.listening.vad_enabled:
            console.print("[dim]Loading VAD model...[/dim]")
            self.vad = SileroVAD(config.listening)

        # Create output directory if saving responses
        if config.tts.save_responses:
            os.makedirs(config.tts.output_dir, exist_ok=True)

        self._print_config()

    def _print_config(self) -> None:
        """Print current configuration."""
        if self.config.tts.voice_sample:
            console.print(f"[green]Voice cloning: {self.config.tts.voice_sample}")
        else:
            console.print("[yellow]Using default voice (no cloning)")

        console.print(f"[blue]LLM model: {self.config.llm.model}")
        console.print(f"[blue]Personality: {self.config.personality.name}")
        console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        console.print("[cyan]Press Ctrl+C to exit.\n")

    def analyze_emotion(self, text: str) -> float:
        """
        Analyze text for emotional content to adjust TTS.

        Args:
            text: Text to analyze.

        Returns:
            Exaggeration value (0.3-0.9).
        """
        emotion_config = self.config.personality.emotion
        keywords = self.config.personality.emotional_keywords

        score = emotion_config.base_exaggeration

        text_lower = text.lower()

        # Check all keyword categories
        for category in ["positive", "negative", "emphasis"]:
            for keyword in keywords.get(category, []):
                if keyword in text_lower:
                    score += emotion_config.keyword_boost

        # Clamp to valid range
        return min(
            emotion_config.max_exaggeration,
            max(emotion_config.min_exaggeration, score),
        )

    def _process_input(self, audio) -> None:
        """
        Process a single voice input: transcribe then pass to _process_text.

        Args:
            audio: Audio data as numpy array.
        """
        if audio.size == 0:
            console.print("[red]No audio recorded. Please ensure your microphone is working.")
            return

        with console.status("Transcribing...", spinner="dots"):
            text = self.stt.transcribe(audio)

        if not text:
            return

        self._process_text(text)

    def _process_text(self, text: str) -> None:
        """
        Process pre-transcribed text through LLM and TTS.

        Args:
            text: Transcribed text to process.
        """
        console.print(f"[yellow]You: {text}")

        sentences: list[str] = []
        with console.status("Generating response...", spinner="dots"):
            for sentence in self.llm.stream(text, self.session_id):
                sentences.append(sentence)

        response = " ".join(sentences)

        exaggeration = self.analyze_emotion(response)
        cfg_weight = self.config.tts.cfg_weight
        if exaggeration > 0.6:
            cfg_weight *= 0.8

        console.print(f"[cyan]{self.config.assistant_name}: {response}")
        console.print(f"[dim](Emotion: {exaggeration:.2f}, CFG: {cfg_weight:.2f})[/dim]")

        # Strip stage directions (e.g. "(sighs dramatically)") before TTS
        tts_sentences = [
            cleaned
            for s in sentences
            if (cleaned := _STAGE_DIRECTION.sub("", s).strip())
        ]

        pipeline = StreamingPipeline(self.tts)
        pipeline.run(
            tts_sentences,
            voice_sample=self.config.tts.voice_sample,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if self.config.tts.save_responses:
            self.response_count += 1
            filename = f"{self.config.tts.output_dir}/response_{self.response_count:03d}.wav"
            self.tts.save_audio(response, filename, self.config.tts.voice_sample)
            console.print(f"[dim]Voice saved to: {filename}[/dim]")

    def _pause_briefly(self, paused: threading.Event, seconds: float) -> None:
        """Keep mic paused for a short duration to avoid echo pickup."""
        if seconds <= 0:
            paused.clear()
            return
        try:
            time.sleep(seconds)
        finally:
            paused.clear()

    def _run_vad_loop(self) -> None:
        """Run VAD-based listening loop (wake word or continuous mode)."""
        if self.vad is None:
            raise RuntimeError(
                "_run_vad_loop called but VAD was not initialised. "
                "Set listening.vad_enabled = True in config."
            )

        listening = self.config.listening
        paused = threading.Event()
        mode = listening.mode

        if mode not in ("wake_word", "continuous"):
            raise ValueError(
                f"Unknown listening mode '{mode}'. Expected 'wake_word' or 'continuous'."
            )

        wake_word = listening.wake_word.lower().strip()
        # Build a flexible pattern that allows punctuation/whitespace between words
        # e.g. "hey archer" matches "hey, archer", "hey archer!", "hey  archer"
        wake_word_tokens = wake_word.split()
        flexible_pattern = r"\b" + r"[,;:!?\s]+".join(re.escape(t) for t in wake_word_tokens) + r"\b"
        wake_word_pattern = re.compile(flexible_pattern)

        console.print(f"[green]Listening mode: {mode}")
        if mode == "wake_word":
            console.print(f'[green]Say "{listening.wake_word}" to activate.')

        # Conversation state for wake_word mode
        conversation_active = False
        last_interaction: float = 0.0

        for utterance in self.recorder.listen_continuous(self.vad, listening, paused):
            if mode == "continuous":
                paused.set()
                try:
                    self._process_input(utterance)
                finally:
                    self._pause_briefly(paused, listening.post_speech_delay_s)
                continue

            # --- wake_word mode ---

            # Check conversation timeout
            if conversation_active:
                if time.monotonic() - last_interaction > listening.conversation_timeout_s:
                    conversation_active = False
                    console.print(
                        f'[yellow]Conversation timed out. Say "{listening.wake_word}" to reactivate.'
                    )

            # If conversation is active, process directly (no wake word needed)
            if conversation_active:
                paused.set()
                last_interaction = time.monotonic()
                try:
                    self._process_input(utterance)
                finally:
                    self._pause_briefly(paused, listening.post_speech_delay_s)
                continue

            # Not active — transcribe and look for wake word
            with console.status("Transcribing...", spinner="dots"):
                text = self.stt.transcribe(utterance)

            if not text:
                continue

            text_lower = text.lower().strip()
            match = wake_word_pattern.search(text_lower)

            if not match:
                console.print(f'[dim]Heard: "{text}" (no wake word)[/dim]')
                continue

            # Activate conversation
            conversation_active = True
            last_interaction = time.monotonic()
            console.print("[green]Wake word detected! Conversation active.")

            # Strip wake word and process remaining text if any
            remaining = text[match.end():].strip()
            remaining = remaining.lstrip(",.!? ")

            if remaining:
                paused.set()
                try:
                    self._process_text(remaining)
                finally:
                    self._pause_briefly(paused, listening.post_speech_delay_s)

    def run(self) -> None:
        """Run the main conversation loop."""
        try:
            if self.config.listening.vad_enabled and self.vad is not None:
                self._run_vad_loop()
            else:
                while True:
                    audio = self.recorder.record_until_enter()
                    self._process_input(audio)
        except KeyboardInterrupt:
            console.print("\n[red]Exiting...")

        console.print(f"[blue]Session ended. Thank you for using {self.config.assistant_name}!")
