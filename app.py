#!/usr/bin/env python3
"""Archer Voice Assistant - Entry Point."""

import argparse
from dataclasses import replace

from archer.core.config import load_config
from archer.core.assistant import Assistant


def main():
    """Main entry point for Archer Voice Assistant."""
    parser = argparse.ArgumentParser(description="Archer Voice Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="Override voice sample path for cloning",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override LLM model",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        help="Override base emotion exaggeration (0.0-1.0)",
    )
    parser.add_argument(
        "--save-voice",
        action="store_true",
        help="Save generated voice responses",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable VAD-based listening (wake word mode by default)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous listening mode (implies --vad)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    if args.voice:
        config.tts.voice_sample = args.voice
    if args.model:
        config.llm.model = args.model
    if args.exaggeration is not None:
        config.personality.emotion.base_exaggeration = args.exaggeration
    if args.save_voice:
        config.tts.save_responses = True
    if args.continuous:
        config = replace(
            config,
            listening=replace(config.listening, vad_enabled=True, mode="continuous"),
        )
    elif args.vad:
        config = replace(
            config,
            listening=replace(config.listening, vad_enabled=True),
        )

    # Run assistant
    assistant = Assistant(config)
    assistant.run()


if __name__ == "__main__":
    main()
