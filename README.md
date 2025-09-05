# YouTube Transcriptor

A powerful Python CLI tool that downloads YouTube videos, transcribes their audio using OpenAI's Whisper API, and provides AI-powered summaries and Q&A capabilities.

## Features

- **Audio Download**: Downloads audio from YouTube videos using `yt-dlp`
- **Smart Transcription**: Uses OpenAI's Whisper API with automatic audio chunking for large files
- **AI Summaries**: Generate summaries with customizable detail levels using OpenAI or Anthropic models
- **Interactive Q&A**: Ask questions about video content with conversation context
- **Caching System**: Intelligent caching of transcripts and Q&A sessions to avoid re-processing
- **Multiple Output Formats**: Save results as Markdown files or JSON cache files
- **Language Support**: Specify transcription language for better accuracy

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd transcriptor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys by copying the example environment file:
```bash
cp .env.example .env
```

5. Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional, for Claude summaries
```

## Usage

### Basic Transcription and Summary

```bash
./yt_transcriber_chunked.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Quick Question Mode

Ask a single question about the video without entering interactive mode:

```bash
./yt_transcriber_chunked.py "https://www.youtube.com/watch?v=VIDEO_ID" "What is the main topic discussed?"
```

Or use the explicit flag:
```bash
./yt_transcriber_chunked.py --ask "What are the key points?" "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Interactive Q&A Mode

Enter an interactive session to ask multiple questions:

```bash
./yt_transcriber_chunked.py --qa "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Transcript Only

Get just the raw transcript without summarization:

```bash
./yt_transcriber_chunked.py --transcript-only "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Options

```bash
./yt_transcriber_chunked.py \
  --provider anthropic \
  --detail detailed \
  --language es \
  --output summary.md \
  --keep-audio \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | OpenAI API key (overrides .env file) | From .env |
| `--language` | Language code for transcription (en, es, fr, etc.) | Auto-detect |
| `--detail` | Summary detail level: `brief`, `medium`, `detailed` | `medium` |
| `--output, -o` | Output file path for saving results | Auto-generated |
| `--keep-audio` | Keep downloaded audio files | False |
| `--transcript-only` | Only transcribe, skip summarization | False |
| `--provider` | LLM provider: `openai` or `anthropic` | `openai` |
| `--show-temp-dir` | Show and preserve temporary directory | False |
| `--qa` | Start interactive Q&A session | False |
| `--ask` | Ask a single question and exit | None |

## Interactive Q&A Commands

When in interactive Q&A mode (`--qa`), you can use these commands:

- Type any question and press Enter
- `/quit`, `/exit`, or `/q` - End session and save
- `/save` - Save Q&A history immediately
- `/clear` - Clear conversation context
- `/help` - Show available commands

## File Organization

The tool creates and uses the following files:

- `transcripts/` - Directory for cached transcripts and Q&A sessions
- `{VIDEO_ID}_{TITLE}.json` - Unified cache files containing video info, transcript, and Q&A history
- `{VIDEO_ID}_{TITLE}_summary.md` - Generated summary files
- `{VIDEO_ID}_{TITLE}_transcript.md` - Raw transcript files

## Caching System

The tool implements intelligent caching to improve performance:

- **Transcript Caching**: Once a video is transcribed, the result is cached to avoid re-processing
- **Session Persistence**: Q&A sessions are automatically saved and can be resumed later
- **Unified Format**: All data (video info, transcript, Q&A history) is stored in single JSON files

## API Requirements

### OpenAI API
- Used for Whisper transcription (always required)
- Used for GPT-4 summaries and Q&A (when `--provider openai`)
- Requires valid `OPENAI_API_KEY`

### Anthropic API
- Used for Claude summaries and Q&A (when `--provider anthropic`)  
- Optional - only needed if using Anthropic models
- Requires valid `ANTHROPIC_API_KEY`

## Audio Processing

The tool handles large audio files automatically:

- **Chunking**: Files over 24MB are split into smaller chunks for Whisper API
- **Format Conversion**: All audio is converted to MP3 format
- **Quality**: Downloads at 192kbps for optimal balance of quality and file size
- **Cleanup**: Temporary files are automatically cleaned unless `--keep-audio` is used

## Examples

### Basic Usage
```bash
# Simple summary
./yt_transcriber_chunked.py "https://youtu.be/dQw4w9WgXcQ"

# Quick question
./yt_transcriber_chunked.py "https://youtu.be/dQw4w9WgXcQ" "What is this video about?"

# Interactive Q&A
./yt_transcriber_chunked.py --qa "https://youtu.be/dQw4w9WgXcQ"
```

### Advanced Usage
```bash
# Detailed summary with Anthropic, Spanish transcription
./yt_transcriber_chunked.py \
  --provider anthropic \
  --detail detailed \
  --language es \
  --output "detailed_summary.md" \
  "https://youtu.be/dQw4w9WgXcQ"

# Just get the transcript
./yt_transcriber_chunked.py --transcript-only "https://youtu.be/dQw4w9WgXcQ"
```

## Error Handling

The tool includes comprehensive error handling for:

- Invalid YouTube URLs or video IDs
- API rate limits and failures
- Large file processing
- Network connectivity issues
- Missing API keys or configuration

## Dependencies

- `yt-dlp` - YouTube video downloading
- `openai` - OpenAI API client for Whisper and GPT
- `anthropic` - Anthropic API client for Claude
- `pydub` - Audio processing and chunking
- `click` - Command line interface
- `rich` - Beautiful terminal output
- `python-dotenv` - Environment variable management

## License

This tool is for educational and personal use. Please respect YouTube's Terms of Service and content creators' rights when using this tool.