### Goal
Migrate the Python CLI transcriptor to a TypeScript CLI with an Ink-based TUI, preserving feature parity and improving distribution, type safety, and streaming ergonomics.

## Phase 0 — Requirements and acceptance criteria
- **Feature parity**: YouTube audio download, chunking, Whisper transcription, summaries (OpenAI/Anthropic), interactive Q&A, JSON caching, Markdown outputs, temp cleanup.
- **CLI flags**: Mirror Python options (`--api-key`, `--language`, `--detail`, `--output`, `--keep-audio`, `--transcript-only`, `--provider`, `--show-temp-dir`, `--qa`, `--ask`).
- **Caching schema**: Keep `transcripts/{VIDEO_ID}_{SANITIZED_TITLE}.json` structure and keys compatible.
- **TUI parity**: Progress indicators, panels, interactive Q&A with `/quit`, `/save`, `/clear`, `/help`.
- **Cross-platform**: Linux/macOS/Windows; minimize external deps via `ffmpeg-static` and `yt-dlp-exec`.
- **Acceptance tests**: Run both Python and TS on the same video; compare output sizes, key fields, and basic content structure.

## Phase 1 — Project bootstrap
- Initialize project with TypeScript and tooling:
  - `typescript`, `tsx` (or `ts-node`), `eslint`, `prettier`, `dotenv`, `rimraf`.
  - `openai`, `@anthropic-ai/sdk`, `fs-extra`, `commander` (or `yargs`).
  - `ink`, `react`, `ink-spinner`, `ink-box`, `ink-text-input`, `ink-markdown` (or `marked` + `marked-terminal`).
  - Media: `yt-dlp-exec`, `fluent-ffmpeg`, `ffmpeg-static`, `tmp`, `get-audio-duration` (optional).
- Create base files:
  - `tsconfig.json`, `eslint`/`prettier` configs.
  - `package.json` with `bin` entry (`"yt-transcriber": "dist/cli.js"`).
  - Scripts: `dev`, `build`, `start`, `lint`, `typecheck`.

## Phase 2 — Types, utils, and config
- `src/types.ts`
  - `VideoInfo`, `TranscriptCache`, `SummaryDetail = 'brief' | 'medium' | 'detailed'`, `Provider = 'openai' | 'anthropic'`, `QAEntry`.
- `src/utils.ts`
  - `sanitizeFilename()` — port Python logic and regex.
  - `extractVideoId()` — port all URL patterns and bare 11-char IDs.
  - `nowIso()` and small helpers.
- `src/config.ts`
  - Read env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TRANSCRIPTOR_*`.
  - Resolve models and defaults: `OPENAI_WHISPER_MODEL='whisper-1'`, `OPENAI_CHAT_MODEL`, `ANTHROPIC_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.

## Phase 3 — YouTube download module
- `src/youtube.ts`
  - Implement `downloadAudio(url, outTemplate?): Promise<{ audioPath: string, info: VideoInfo }>` using `yt-dlp-exec`.
  - Use `ffmpeg-static` path if needed by yt-dlp postprocessor.
  - Return metadata: title, duration, uploader, upload_date, view_count.
  - Respect a session temp directory; expose its path to the caller.

## Phase 4 — Audio chunking module
- `src/audio.ts`
  - Derive number of chunks from file size target (e.g., 24 MB to stay under 25 MB).
  - Use `fluent-ffmpeg` with `ffmpeg-static` to slice by duration computed from target size.
  - Ensure deterministic chunk naming and a chunk subfolder inside session temp.
  - Provide `splitIfNeeded(audioPath, maxSizeMB): Promise<string[]>`.

## Phase 5 — Transcription module
- `src/transcribe.ts`
  - `transcribeChunk(filePath, language?)` with OpenAI Node SDK (`client.audio.transcriptions.create`).
  - `transcribe(audioPath, language?, onProgress?)` that calls `splitIfNeeded` and merges chunk texts with spacing.
  - Persist chunk transcripts to temp for debugging (optional, as in Python).

## Phase 6 — Summaries and Q&A modules
- `src/summarize.ts`
  - Prompt templates for `brief`, `medium`, `detailed` identical in intent.
  - `summarize(text, detail, provider)` routes to `_openai` or `_anthropic` using configured models and temperature/max tokens.
- `src/qa.ts`
  - `answerQuestion(question, transcript, videoInfo, history, provider)`.
  - Mirror Python conversation handling for OpenAI/Anthropic with consistent message arrays.

## Phase 7 — Caching and outputs
- `src/cache.ts`
  - Read/write `transcripts/{VIDEO_ID}_{SANITIZED_TITLE}.json`.
  - Merge semantics: preserve existing fields, update `summaries[provider][detail]`, append `qa_history`, update `last_updated`.
- `src/output.ts`
  - Write Markdown summary/transcript files with the same headings/order as Python.

## Phase 8 — CLI command surface
- `src/cli.ts`
  - Parse flags; support shorthand where a second arg is the quick question (maps to `--ask`).
  - Implement modes: transcript-only, ask (quick answer), qa (interactive), or default summarize with cache reuse.
  - Log with `chalk` and minimal text output when not in TUI.

## Phase 9 — Ink TUI
- `src/tui/App.tsx`
  - State machine: `idle → extractingId → checkingCache → downloading → transcribing → summarizing → showingResult` or `→ qaMode`.
  - Use `ink-spinner` for progress; `ink-box`/styled panels for sections; `ink-markdown` to display answers/summaries.
  - `QAView`: text input for questions; handle `/quit`, `/save`, `/clear`, `/help`.
  - Provide a non-TUI fallback path when output is redirected or `--no-tui` (optional).
- Wire TUI into CLI:
  - If `--qa` or interactive terminal and not `--transcript-only`, render Ink; else minimal CLI panels.

## Phase 10 — Error handling and cleanup
- Standardize error classes for config, download, transcription, summarization.
- Always clean temp directories unless `--keep-audio` or `--show-temp-dir`.
- Clear, actionable messages for missing API keys, ffmpeg issues, network errors.

## Phase 11 — Testing
- Unit tests (Vitest/Jest):
  - `utils` (sanitize, extractVideoId), `cache` merge, prompt builders.
- Integration tests:
  - Mock OpenAI/Anthropic with `nock` or `msw`.
  - Simulate small audio files for chunking via `ffmpeg`.
- E2E smoke:
  - One public short video; ensure download → transcribe → summarize pipeline succeeds under CI (can be opt-in).

## Phase 12 — Packaging and distribution
- Build: `tsup` or `esbuild` to `dist/`.
- CLI bin:
  - `package.json: { "bin": { "yt-transcriber": "dist/cli.js" } }`
  - Add shebang and set executable flag.
- Optional single-binary:
  - `pkg`/`nexe` builds for linux/macos/win; confirm `ffmpeg-static` and `yt-dlp-exec` work inside bundle (or ship them side-by-side).
- Publish to npm and attach GitHub Release assets.

## Phase 13 — Migration strategy
- Coexist tools during migration.
- Golden test:
  - Run Python and TS on 3–5 sample videos (short, medium, long).
  - Compare: cache JSON structure/fields, transcript length similarity, summary generation presence per detail level, QA roundtrip persistence.
- Rollout:
  - Tag TS v0.x behind `--ts` flag (optional) or release as `yt-transcriber-js` first.
  - Gather feedback, then deprecate Python with a notice in README.

## Phase 14 — Timeline (estimate)
- Bootstrap + core modules (utils, config, types): 0.5 day
- Download + chunking: 0.5–1 day
- Transcription + summaries/QA: 0.5–1 day
- Caching + outputs: 0.25 day
- CLI + Ink TUI: 1–1.5 days
- Tests + packaging: 0.5–1 day
- Total: ~3–5 days for polished parity

## Phase 15 — Risks and mitigations
- **ffmpeg/yt-dlp availability**: Use `ffmpeg-static` and `yt-dlp-exec`; document fallbacks.
- **Large files**: Stream where possible; avoid buffering entire files in memory.
- **API rate/limits**: Retry with backoff; chunk intelligently; expose `--language`.
- **Windows quirks**: Validate paths and process signals for TUI exit/cleanup.
- **Terminal capabilities**: Detect non-TTY; downgrade UI gracefully.

## Suggested dependency install commands
```bash
# core
npm i openai @anthropic-ai/sdk dotenv commander fs-extra
# TUI
npm i react ink ink-spinner ink-box ink-markdown chalk
# media
npm i yt-dlp-exec fluent-ffmpeg ffmpeg-static tmp
# dev
npm i -D typescript tsx tsup eslint prettier vitest @types/node @types/react
```

## Directory layout
- `src/cli.ts`
- `src/tui/App.tsx`
- `src/youtube.ts`
- `src/audio.ts`
- `src/transcribe.ts`
- `src/summarize.ts`
- `src/qa.ts`
- `src/cache.ts`
- `src/output.ts`
- `src/utils.ts`
- `src/types.ts`

### Summary
- Clear 15-phase plan covering parity, TS modules, Ink TUI, caching, tests, packaging, rollout.
- Choices: `yt-dlp-exec` + `ffmpeg-static`, OpenAI/Anthropic Node SDKs, Ink for TUI, `commander` for CLI.
- Migration path ensures side-by-side validation against the Python tool before deprecation.

