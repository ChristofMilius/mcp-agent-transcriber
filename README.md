# mcp-agent-transcriber

**Video transcription for agents**, as a local **MCP server**. Point an
MCP-aware harness at any common tube-platform link (YouTube, Vimeo,
Dailymotion, Twitch, TikTok, ...) and get an agent-friendly transcript —
either by grabbing the platform's own captions directly (fast, no download,
no model) or by downloading the audio at a preferred quality and transcribing
it locally with OpenAI Whisper (turbo, CUDA/CPU).

This is the standalone `whisper_transcriber` (a terminal-menu CLI) rebuilt as
an MCP tool for the agent-tools stack, plus the caption-grabbing workflow its
history documented for YT downloads. The interactive REPL became MCP tools:
probe a URL, download audio, or transcribe — callable by any harness over
stdio.

## Two paths

| Path | How | Cost | Use when |
|---|---|---|---|
| **direct** | `fetch_transcript` — YouTube via youtube-transcript-api, other platforms via the yt-dlp caption pass | no model, no download | captions exist and speed matters |
| **whisper** | `download_audio` (yt-dlp) → `transcribe_*` (local Whisper turbo) | ~1.6 GB model, downloads audio | captions missing, or fidelity matters |

`transcribe_video(url, method="auto")` runs the whole route: captions first,
Whisper fallback when the platform has none.

## Tool surface

| Domain | Tools |
|---|---|
| Info | `supported_platforms`, `video_info` |
| Transcript (direct) | `list_available_transcripts`, `fetch_transcript` |
| Download | `download_audio` |
| Transcribe (Whisper) | `transcriber_status`, `transcribe_file`, `transcribe_video` |
| Utility | `get_current_datetime` |

All tools return JSON. Unexpected errors are sanitized to the tool name +
exception class; the full traceback stays in the server log. Known conditions
(private video, disabled captions, unknown platform, sign-in required,
network) come back as structured `status: failed` payloads with a static,
path-free reason — exception *messages* are never forwarded because they can
leak local paths.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- **ffmpeg on PATH** (Whisper and audio postprocessing both need it)
- NVIDIA GPU + CUDA 12.8 driver for GPU transcription (CPU fallback automatic)

## Install

```powershell
git clone <repo-url> mcp_agent_transcriber
cd mcp_agent_transcriber
uv sync
```

`uv sync` installs the CUDA build of PyTorch automatically: `pyproject.toml`
pins torch to PyTorch's official CUDA 12.8 wheel index (`pytorch-cu128`) via
`[tool.uv.sources]`. To switch CUDA versions, change the index URL.

## Usage

### Run as an MCP server (stdio — what harnesses expect)

```powershell
uv run mcp-agent-transcriber
```

Override paths via environment variables (defaults resolve relative to the
project root, never the process CWD):

| Var | Default | Purpose |
|---|---|---|
| `TRANSCRIBER_OUTPUT_DIR` | `transcript_output` | Where transcript `.txt`/`.vtt` files go |
| `TRANSCRIBER_DOWNLOADS_DIR` | `<output>/downloads` | Where audio downloads land |
| `TRANSCRIBER_LOGS_DIR` | `logs` | Rotating DEBUG logs (5×5 MB) |
| `TRANSCRIBER_MODEL` | `turbo` | Whisper checkpoint (turbo / large-v3 / medium / ...) |
| `TRANSCRIBER_LANGUAGE` | `en` | Default caption language for direct grabs |

To serve over HTTP (streamable-http) instead:

```powershell
uv run mcp-agent-transcriber serve --http --port 8000
```

### Register in an MCP client

```json
{
  "mcpServers": {
    "mcp-agent-transcriber": {
      "command": "uv",
      "args": ["--project", "C:/path/to/mcp_agent_transcriber", "run", "mcp-agent-transcriber"]
    }
  }
}
```

### CLI commands

| Command | Purpose |
|---|---|
| `serve` *(default)* | Run the MCP server. `--http --port` for streamable-http |
| `doctor` | Offline diagnostics: paths, engine state, ffmpeg, yt-dlp |

## Workflow

```
1. supported_platforms()      → is the host a known tube platform?
2. video_info(url)            → platform, title, caption tracks, audio formats
3. fetch_transcript(url)      → direct transcript when captions exist
4. download_audio(url, "best")→ audio track for the whisper path
5. transcribe_video(url, method="auto") → one call, captions then Whisper
```

The Whisper checkpoint is **never loaded at server startup** — it loads on the
first `transcribe_*` call (~1.6 GB) and stays cached for the process
lifetime, so caption-only workflows never touch the model. Transcripts are
written as `{title-slug}_{lang}_{timestamp}.txt` (plus `.vtt` on request)
into `TRANSCRIBER_OUTPUT_DIR` and returned inline so the model gets the text
even without a filesystem view.

## Model weights

The turbo checkpoint (`large-v3-turbo.pt`, ~1.6 GB) resolves through Whisper's
standard cache (`~/.cache/whisper/` on Windows). It auto-downloads on first
use and is then reused. Offline machines can place the `.pt` file into that
cache by hand.

## Output quality

Download quality presets map to yt-dlp format selectors:

| Preset | Selector |
|---|---|
| `best` *(default)* | `bestaudio/best` |
| `high` | `bestaudio[abr>=128]/bestaudio/best` |
| `standard` | `bestaudio[abr<=128]/bestaudio/best` |
| `low` | `worstaudio/worst` |

`download_audio(output_format=...)` keeps the platform's container (`orig`) by
default, or re-encodes with ffmpeg to mp3/wav/m4a/opus/flac/aac.

Caption text is cleaned for agent consumption (ported from the original
workflow): inline `<...>` timing tags are stripped, cue timestamps dropped,
consecutive duplicate lines collapsed (the rolling word-by-word caption
artifact), and whitespace normalized.

## Development

```powershell
uv run ruff check .
uv run pyright
uv run pytest -q
```

Tests cover config path resolution, caption cleaning, platform discovery,
yt-dlp metadata/download (mocked), the transcript grabbers (mocked APIs), the
pipeline routes, and the full tool surface — no network, no model load.

### Type checking

`pyright` resolves imports against the project venv via `[tool.pyright]` in
`pyproject.toml`. That section is load-bearing: without `venvPath`/`venv` it
falls back to the system interpreter and reports every third-party import as
missing, which buries the real findings under hundreds of phantom errors. Keep
it at 0 errors — the last run surfaced two genuine bugs that the test suite
passed straight over.

Four call sites use `cast(Any, ...)` around `yt_dlp.YoutubeDL(...)` and
`ydl.prepare_filename(...)` — three in `download.py`, one in `transcript.py`.
**Do not "clean these up" by typing the options dict as
`yt_dlp.YoutubeDL._Params`.** That name exists only inside a typeshed-fallback
stub bundled with pyright, so the import satisfies the type checker and then
raises `ImportError` at server startup — the LSP goes green and the server
dies. The stub is wrong besides: it declares `skip_download` as `str | None`,
where yt-dlp tests it for truthiness and documents it as a boolean flag. The
casts are deliberate, and each carries its reason inline.

## Security advisories

Two Dependabot alerts have been triaged and dismissed, both `not_used`:

| Alert | CVE | Why not used |
|---|---|---|
| `torch` | CVE-2025-3000 | The vulnerability is in `torch.jit.script`. This project calls only `torch.cuda.is_available()` (`whisper_engine.py`), and openai-whisper does not route inference through TorchScript. No patched version published. |
| `setuptools` | CVE-2026-59890 | Transitive build-time dependency of torch. This project builds with `uv_build`, ships no sdists, and never runs setuptools' sdist builder. The CVE is macOS APFS manifest-matching only. |

The `torch` dismissal is the fragile one, because it is a **claim about the
code** rather than a fixed property of a pinned artifact: the day a TorchScript
path appears, "not used" is false and Dependabot stays silent, having been told
to stop looking. `tests/test_security_invariants.py` is what keeps that honest —
it fails the build if `torch.jit` ever enters `src/`, naming the file, line and
symbol, and telling you to re-open the alert. It parses the AST rather than
grepping, so a comment *about* the CVE cannot trip it. The second half of the
claim — that openai-whisper stays TorchScript-free — is covered by `uv.lock`
pinning the exact version, so an upgrade that changed it surfaces as a
reviewable diff.

## License

MIT © 2026 Christof Milius — this repository ships only hand-authored source
and configuration; nothing else.

- **No redistributed dependencies.** torch (BSD-3), openai-whisper (MIT),
  yt-dlp (Unlicense), youtube-transcript-api (MIT) and friends are declared
  in `pyproject.toml` and installed by uv — never vendored into this repo.
- **No model weights.** The Whisper checkpoint is *not* included; it is
  downloaded at runtime into `~/.cache/whisper` on first use and subject to
  its own license terms. The transcription machinery is MIT regardless.
- **No licensed media.** Transcribed caption/audio content is produced at
  runtime by the tool; no copyrighted media ships in the repository.