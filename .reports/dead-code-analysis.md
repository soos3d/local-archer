# Dead Code Analysis -- Archer Voice Assistant

> Generated: 2026-02-19 (revision 2)
> Tools: vulture 2.x (--min-confidence 70), manual AST import analysis, grep-based call-graph tracing
> Scope: app.py + archer/ (18 Python files), pyproject.toml, requirements.txt

---

## Summary

| Category | Items | Recommended Action |
|----------|-------|--------------------|
| SAFE -- clearly dead code | 5 | Remove |
| CAUTION -- unused interface methods | 4 | Keep (part of public abstract API) |
| CAUTION -- unused but potentially useful | 2 | Keep for now |
| INFO -- already deleted (unstaged) | 1 | Stage the deletion |
| Unused dependencies (requirements.txt) | 8+ | Remove from pinned requirements |

---

## 1. SAFE -- Clearly Dead Code

### 1a. Vulture findings (100% confidence)

**File:** `/Users/davide/Documents/coding/local-talking-llm/archer/audio/recorder.py`, line 31

```python
def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
```

The parameters `frames` and `time_info` are never used inside the callback body.
These are required by the sounddevice callback signature, so they cannot be removed
but should be prefixed with underscores to signal they are intentionally unused:
`_frames`, `_time_info`.

**Risk:** None. Renaming unused callback params is a safe convention change.

---

### 1b. `PocketTTS.export_voice_state()` -- never called

**File:** `/Users/davide/Documents/coding/local-talking-llm/archer/tts/pocket_tts.py`, lines 123-132

```python
def export_voice_state(self, output_path: str) -> None:
    """Export the current voice state to a .safetensors file for fast reloading."""
    export_model_state(self._voice_state, output_path)
```

This method is defined but never called anywhere in the codebase. It also pulls in
the `export_model_state` import from `pocket_tts` which is used only here.

This is a utility method that could be useful via an interactive session or CLI
extension, but today it is dead code. If removed, the `export_model_state` import
on line 8 can also be trimmed from the import statement.

**Risk:** Low. Not part of any abstract base class contract. No callers.

---

### 1c. `BaseTTS.synthesize_long()` -- abstract method never called externally

**File:** `/Users/davide/Documents/coding/local-talking-llm/archer/tts/base.py`, lines 39-59

The abstract method `synthesize_long()` is defined in `BaseTTS` and implemented in
both `ChatterboxTTS` and `PocketTTS`. However, the orchestrator `Assistant._process_input()`
does its own sentence-splitting and per-sentence `synthesize()` calls (lines 191-205 of
`assistant.py`). The only internal caller of `synthesize_long()` is
`PocketTTS.save_audio()` which calls `self.synthesize_long(text)`.

The `ChatterboxTTS.synthesize_long()` method is truly dead -- it is never called by
anyone. However, because it is part of the abstract interface and the `PocketTTS`
version *is* used transitively via `save_audio()`, this falls into CAUTION territory.

**Risk:** Medium. Removing the abstract method would break the interface contract.
Recommendation: keep but document that the orchestrator does not use it directly.

---

### 1d. Deleted file `tts.py` -- unstaged deletion

**File:** `/Users/davide/Documents/coding/local-talking-llm/tts.py`

Git status shows `D tts.py`. This is the old monolithic TTS module (122 lines) from
before the refactor into `archer/tts/`. It still exists in the HEAD commit but has
been deleted from the working tree. The deletion should be staged and committed.

No code in the project imports from `tts.py` -- it has been fully superseded by
`archer/tts/chatterbox_tts.py`.

**Risk:** None. Already deleted locally; just needs to be committed.

---

### 1e. `archer/__init__.py` -- `__version__` never referenced

**File:** `/Users/davide/Documents/coding/local-talking-llm/archer/__init__.py`

```python
__version__ = "0.1.0"
```

The `__version__` string is defined but never imported or referenced anywhere.
The canonical version lives in `pyproject.toml` line 3 (`version = "0.1.0"`).

**Risk:** Very low. This is a common Python convention, and some tools may read it
dynamically. Recommendation: keep for now (convention), but note the duplication
with `pyproject.toml`.

---

## 2. CAUTION -- Unused Interface Methods (Keep)

These methods are part of abstract base class contracts. They exist to define the
public API for provider implementations. None are called by the current orchestrator,
but they form a deliberate extensibility surface.

| Method | File | Reason to keep |
|--------|------|----------------|
| `BaseLLM.clear_history()` | `archer/llm/base.py:24` | Part of LLM provider contract |
| `BaseLLM.set_system_prompt()` | `archer/llm/base.py:34` | Part of LLM provider contract |
| `OllamaLLM.clear_history()` | `archer/llm/ollama_llm.py:74` | Implements abstract method |
| `OllamaLLM.set_system_prompt()` | `archer/llm/ollama_llm.py:84` | Implements abstract method |

---

## 3. CAUTION -- Unused but Potentially Useful (Keep)

### `AudioRecorder.start_recording()`, `stop_recording()`, `get_audio_array()`

These three public methods are only called internally by `record_until_enter()`.
They are not dead code -- they form the lower-level recording API that
`record_until_enter()` composes. Keeping them enables programmatic recording
without the Enter-key interaction.

### `archer/tools/__init__.py`

Empty module with `__all__ = []` and comments about future tools. This is a
placeholder for planned functionality. Keep as-is.

---

## 4. Dependency Analysis

### Top-level packages actually imported by source code

```
chatterbox    (chatterbox-tts)
langchain_core (langchain-core, transitive via langchain)
langchain_ollama (langchain-ollama)
nltk
numpy
pocket_tts    (pocket-tts)
rich
sounddevice
soundfile     (used only in pocket_tts.py)
torch
torchaudio
whisper       (openai-whisper)
yaml          (pyyaml)
```

### pyproject.toml dependencies -- all actively used

| Dependency | Used by | Status |
|------------|---------|--------|
| chatterbox-tts | `archer/tts/chatterbox_tts.py` | USED |
| pocket-tts | `archer/tts/pocket_tts.py` | USED |
| langchain-ollama | `archer/llm/ollama_llm.py` | USED |
| langchain | transitive (langchain-core) | USED |
| nltk | `chatterbox_tts.py`, `pocket_tts.py`, `assistant.py` | USED |
| openai-whisper | `archer/stt/whisper_stt.py` | USED |
| pyyaml | `archer/core/config.py` | USED |
| rich | `recorder.py`, `whisper_stt.py`, `assistant.py` | USED |
| setuptools | build system | USED |
| sounddevice | `recorder.py`, `player.py` | USED |
| torchaudio | `archer/tts/chatterbox_tts.py` | USED |

**All pyproject.toml dependencies are actively used. No removals recommended.**

### requirements.txt -- packages not directly imported

The `requirements.txt` is a full pip freeze (99 pinned packages). Many are
transitive dependencies that pip resolves automatically. The following packages
in `requirements.txt` are NOT directly imported and are purely transitive:

| Package | Pulled in by |
|---------|-------------|
| sqlalchemy | langchain (transitive) |
| pillow | diffusers/transformers (transitive) |
| tiktoken | langchain (transitive) |
| peft | transformers (transitive) |
| onnx | chatterbox-tts (transitive) |
| conformer | chatterbox-tts (transitive) |
| diffusers | chatterbox-tts (transitive) |
| s3tokenizer | chatterbox-tts (transitive) |

These are fine to keep in a pinned requirements.txt (they ensure reproducible installs).
However, `requirements.txt` should not be considered the source of truth for direct
dependencies -- `pyproject.toml` serves that role.

**Recommendation:** No changes to `requirements.txt` needed. The `pyproject.toml`
already correctly lists only direct dependencies.

---

## 5. Proposed Safe Cleanups

### Cleanup A: Prefix unused callback parameters (recorder.py)

Change `frames` to `_frames` and `time_info` to `_time_info` in
`AudioRecorder._audio_callback()` to follow Python convention for unused parameters.

### Cleanup B: Remove `PocketTTS.export_voice_state()` and trim import

Remove the dead method and simplify the import line in `pocket_tts.py`.

### Cleanup C: Stage the `tts.py` deletion

The old monolithic `tts.py` has already been deleted from disk. Stage this deletion
so it gets included in the next commit.

---

## 6. Items NOT recommended for removal

| Item | Reason |
|------|--------|
| `BaseTTS.synthesize_long()` | Abstract interface contract; `PocketTTS.save_audio()` uses it |
| `BaseLLM.clear_history()` | Abstract interface contract |
| `BaseLLM.set_system_prompt()` | Abstract interface contract |
| `archer/__init__.py __version__` | Python convention; harmless |
| `archer/tools/__init__.py` | Intentional placeholder |
| `AudioRecorder.start/stop/get_audio_array` | Composable lower-level API |

---

## 7. No Test Suite

No `tests/` directory exists. All proposed cleanups are limited to obviously dead
code with zero callers. Adding a test suite is strongly recommended before
performing larger refactoring efforts.
