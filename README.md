# YouTube Transcriptor

A sophisticated Python-based command-line application that transforms YouTube video content into actionable intelligence through automated transcription, AI-powered summarization, and interactive question-answering capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Command Line Options](#command-line-options)
7. [Core Components](#core-components)
8. [Technical Implementation](#technical-implementation)
9. [API Integration](#api-integration)
10. [Advanced Features](#advanced-features)
11. [Configuration](#configuration)
12. [Performance & Optimization](#performance--optimization)
13. [Troubleshooting](#troubleshooting)
14. [Development](#development)
15. [License](#license)

---

## Overview

The YouTube Transcriptor is a production-ready system that leverages OpenAI's Whisper API for high-accuracy transcription and supports both OpenAI GPT and Anthropic Claude models for content analysis. Built with a modular architecture, it provides flexible, scalable, and intelligent video content processing with a focus on user experience and reliability.

### Key Capabilities

- **Automatic Transcription**: Convert YouTube videos to text with high accuracy
- **Intelligent Summarization**: Generate summaries at multiple detail levels
- **Interactive Q&A**: Ask questions about video content with context awareness
- **Smart Caching**: Avoid re-processing with intelligent caching system
- **Multi-Provider Support**: Choose between OpenAI and Anthropic models
- **Rich Terminal UI**: Beautiful, informative command-line interface

---

## Features

### Core Features

- **Audio Download**: Downloads audio from YouTube videos using `yt-dlp`
- **Smart Transcription**: Uses OpenAI's Whisper API with automatic audio chunking for large files
- **AI Summaries**: Generate summaries with customizable detail levels using OpenAI or Anthropic models
- **Interactive Q&A**: Ask questions about video content with conversation context
- **Caching System**: Intelligent caching of transcripts and Q&A sessions to avoid re-processing
- **Multiple Output Formats**: Save results as Markdown files or JSON cache files
- **Language Support**: Specify transcription language for better accuracy

### Advanced Features

- **Automatic Audio Chunking**: Handles videos of any length by splitting large files
- **Session Persistence**: Resume Q&A sessions across multiple runs
- **Quick Question Mode**: Get instant answers without entering interactive mode
- **Enhanced Display Formats**: Multiple visual formats for summary presentation
- **Provider Flexibility**: Switch between LLM providers seamlessly
- **Progress Tracking**: Real-time progress updates for long operations

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Command Line Interface                    │
│                    (Click Framework + Rich)                   │
├─────────────────────────────────────────────────────────────┤
│                     Orchestration Layer                       │
│                     (Main Entry Point)                        │
├──────────────┬────────────────┬────────────────┬───────────┤
│   Audio      │  Transcription  │  Summarization │    Q&A    │
│  Extraction  │     Engine      │     Engine     │  Handler  │
├──────────────┴────────────────┴────────────────┴───────────┤
│                      Cache Management                         │
│                   (Unified JSON Storage)                      │
├─────────────────────────────────────────────────────────────┤
│                     External APIs Layer                       │
│              (OpenAI, Anthropic, YouTube/yt-dlp)             │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each component handles a specific responsibility with well-defined interfaces
2. **Scalability**: Automatic chunking for large files, parallel processing capabilities
3. **Reliability**: Comprehensive error handling, automatic retries, intelligent caching
4. **Extensibility**: Provider-agnostic design for LLM integration
5. **User-Centric**: Progressive disclosure of complexity, intuitive command structure

---

## Installation

### Prerequisites

1. **Python 3.9+** - Required for running the application
2. **FFmpeg** - Required for audio processing
3. **uv** - Fast Python package manager (recommended)

### Step-by-Step Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd transcriptor
```

2. **Install system dependencies**:

**macOS:**
```bash
brew install ffmpeg
brew install uv
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html and add to PATH
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. **Install Python dependencies**:
```bash
uv sync
```

4. **Configure API keys**:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
```

---

## Usage Guide

### Basic Usage

#### Simple Transcription and Summary
```bash
uv run ./transcriptor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### Quick Question Mode
Ask a single question without entering interactive mode:
```bash
# Shorthand syntax
uv run ./transcriptor.py "https://youtu.be/VIDEO_ID" "What is the main topic?"

# Explicit flag
uv run ./transcriptor.py --ask "What are the key points?" "https://youtu.be/VIDEO_ID"
```

#### Interactive Q&A Mode
Start an interactive session to ask multiple questions:
```bash
uv run ./transcriptor.py --qa "https://youtu.be/VIDEO_ID"
```

#### Transcript Only
Get just the raw transcript without summarization:
```bash
uv run ./transcriptor.py --transcript-only "https://youtu.be/VIDEO_ID"
```

### Advanced Usage

#### Detailed Summary with Anthropic
```bash
uv run ./transcriptor.py \
  --provider anthropic \
  --detail detailed \
  --language es \
  --output "detailed_summary.md" \
  "https://youtu.be/VIDEO_ID"
```

#### Batch Processing Example
```bash
# Process multiple videos with rate limiting
for url in video_urls.txt; do
  uv run ./transcriptor.py "$url" -o "summaries/$(basename $url).md"
  sleep 60  # Rate limiting
done
```

### Interactive Q&A Commands

When in interactive Q&A mode (`--qa`), you can use these commands:

- Type any question and press Enter to get an answer
- `/quit`, `/exit`, or `/q` - End session and save
- `/save` - Save Q&A history immediately
- `/clear` - Clear conversation context
- `/help` - Show available commands

---

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
| `--display-format` | Display format: `standard`, `enhanced`, `cards` | `enhanced` |

### Detail Levels Explained

- **Brief**: 2-3 sentence executive summary focusing on main topic and conclusion
- **Medium**: Structured summary with main topic, 3-5 key points, important details, conclusion
- **Detailed**: Comprehensive analysis with themes, evidence, quotes, implications, action items

---

## Core Components

### 1. YouTubeAudioExtractor

Manages the complete lifecycle of audio extraction from YouTube videos.

**Key Features:**
- Video ID extraction from various YouTube URL formats
- Audio download orchestration via yt-dlp
- Temporary file management with session-based isolation
- Metadata extraction (title, duration, uploader, view count, upload date)
- Automatic format conversion to MP3 (192kbps quality)

### 2. ChunkedWhisperTranscriber

Handles intelligent audio transcription with automatic chunking for large files.

**Key Features:**
- Automatic audio file chunking (24MB threshold)
- Progress tracking with real-time updates
- Language-specific transcription support
- Intermediate result preservation
- Automatic transcript merging with continuity preservation

### 3. Summarizer

Provider-agnostic summarization engine supporting multiple LLM backends.

**Key Features:**
- Multiple detail levels (brief, medium, detailed)
- Automatic text chunking for ultra-long transcripts (>100,000 characters)
- Multi-stage summarization (chunk → aggregate → final)
- Provider-specific optimization
- Template-based prompt engineering

### 4. QAHandler

Manages interactive and non-interactive question-answering sessions with conversation context.

**Key Features:**
- Conversation history management
- Context-aware responses
- Session persistence and restoration
- Multi-provider support
- Auto-save functionality

### 5. Format Summary Module

Enhanced visual presentation layer for terminal output.

**Key Features:**
- Color-coded sections (cyan for topics, yellow for key points, etc.)
- Icon-based visual cues (📌, 🔑, 💡, 🎯)
- Responsive panel layouts
- Markdown rendering support

---

## Technical Implementation

### Audio Processing Pipeline

1. **Download Phase**:
   - URL validation and video ID extraction
   - yt-dlp configuration with optimal settings
   - Format selection (bestaudio with MP3 conversion)
   - Metadata extraction during download

2. **Chunking Strategy**:
   - File size evaluation (24MB threshold)
   - Dynamic chunk duration calculation
   - PyDub-based audio splitting
   - Chunk isolation in subdirectories

3. **Transcription Workflow**:
   ```
   Audio File → Size Check → [Chunking if >24MB] → 
   Whisper API → [Merge if chunked] → Final Transcript
   ```

### Caching Architecture

**Unified Cache Structure**:
```json
{
  "video_info": {
    "title": "Video Title",
    "duration": 765,
    "uploader": "Channel Name",
    "view_count": 22985,
    "upload_date": "20250620"
  },
  "transcript": "Full transcript text...",
  "qa_history": [
    {
      "question": "User question",
      "answer": "AI response"
    }
  ],
  "summaries": {
    "openai": {
      "brief": "Brief summary...",
      "medium": "Medium summary...",
      "detailed": "Detailed summary..."
    },
    "anthropic": {
      "brief": "Brief summary..."
    }
  },
  "last_updated": "2025-09-06T16:44:03.424320"
}
```

**Cache File Naming**:
```
transcripts/{VIDEO_ID}_{SANITIZED_TITLE}.json
```

### File Organization

The tool creates and uses the following structure:

```
.transcriptor/
├── transcripts/           # Unified cache files
│   ├── {video_id}_{title}.json
│   └── ...
├── .env                   # API keys and configuration
├── transcriptor.py        # Main application
├── format_summary.py      # Display formatting module
└── pyproject.toml        # Project dependencies
```

---

## API Integration

### OpenAI Integration

**Services Used:**

1. **Whisper API** (Audio Transcription)
   - Model: `whisper-1`
   - Format: MP3 input, text output
   - Max file size: 25MB
   - Supported languages: 50+

2. **Chat Completions API** (Summarization & Q&A)
   - Default model: `gpt-4o-mini`
   - Temperature: 0.3 (configurable)
   - Max tokens: 3000 (configurable)

### Anthropic Integration

**Services Used:**

1. **Messages API** (Summarization & Q&A)
   - Default model: `claude-3-5-sonnet-20241022`
   - Temperature: 0.3
   - Max tokens: 3000

### YouTube Data Extraction (yt-dlp)

**Configuration:**
- Format: Best audio quality available
- Codec: MP3 conversion at 192kbps
- Metadata extraction: Title, duration, uploader, views, upload date

---

## Advanced Features

### Intelligent Caching System

**Multi-Level Cache Strategy:**
- **Transcript Cache**: Permanent storage of raw transcriptions
- **Summary Cache**: Provider and detail-level specific
- **Q&A History Cache**: Session-based with incremental updates

**Cache Lookup Algorithm:**
1. Check video_id + title combination
2. Fallback to video_id glob pattern
3. Create new cache if not found
4. Merge with existing data on write

### Automatic Audio Chunking

**Dynamic Chunk Calculation:**
```python
file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
num_chunks = int(file_size_mb / 24) + 1
chunk_duration_ms = audio_length_ms / num_chunks
```

**Benefits:**
- Handles videos of any length
- Maintains continuity across chunks
- Parallel processing capability
- Failure isolation per chunk

### Conversation Context Management

**Features:**
- Full conversation history retention
- Intelligent context pruning for long sessions
- System prompt injection for consistency
- Cross-session context preservation

### Enhanced Display Formats

- **Standard Format**: Basic panel display
- **Enhanced Format**: Structured with icons and sections
- **Cards Format**: Individual panels per section

---

## Configuration

### Environment Variables

**Required:**
```bash
OPENAI_API_KEY=your_openai_key        # Required for transcription
```

**Optional:**
```bash
ANTHROPIC_API_KEY=your_anthropic_key  # For Claude models
TRANSCRIPTOR_LLM_TEMPERATURE=0.3      # Response creativity (0-1)
TRANSCRIPTOR_LLM_MAX_TOKENS=3000      # Max response length
TRANSCRIPTOR_OPENAI_CHAT_MODEL=gpt-4o-mini
TRANSCRIPTOR_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Python Dependencies

Key dependencies managed via `pyproject.toml`:
- `yt-dlp>=2024.10.7` - YouTube video downloading
- `openai==1.54.0` - OpenAI API client
- `anthropic==0.39.0` - Anthropic API client
- `pydub>=0.25.1` - Audio processing
- `click==8.1.7` - Command line interface
- `rich==13.9.4` - Terminal UI
- `python-dotenv==1.0.1` - Environment management

---

## Performance & Optimization

### Processing Benchmarks

**Typical Performance Metrics:**
- **Audio Download**: 10-30 seconds (depends on video length)
- **Transcription**: 1-3 minutes per 10 minutes of audio
- **Summarization**: 5-15 seconds
- **Q&A Response**: 2-5 seconds

### Optimization Strategies

1. **Caching Efficiency**:
   - Zero-latency for cached content
   - Incremental cache updates
   - Provider-specific summary caching

2. **Chunking Optimization**:
   - Parallel chunk processing capability
   - Optimal chunk size calculation
   - Memory-efficient streaming

3. **API Call Minimization**:
   - Batch processing where possible
   - Cache-first retrieval strategy
   - Intelligent retry mechanisms

### Performance Tuning Tips

**For Large Videos (>1 hour):**
- Expect 5-10 minute processing time
- Consider using `--transcript-only` first
- Use `--keep-audio` for re-processing

**For Batch Processing:**
- Implement rate limiting (60 seconds between videos)
- Monitor API usage dashboard
- Use caching strategically

**For Interactive Sessions:**
- Pre-load transcripts with `--transcript-only`
- Use `--provider anthropic` for longer context
- Clear context periodically with `/clear`

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Error Message | Solution |
|-------|---------------|----------|
| API Key Missing | "API key not found" | Set `OPENAI_API_KEY` in .env file |
| Network Error | "Connection failed" | Check internet connection |
| Video Not Found | "Could not extract video ID" | Verify URL format is correct |
| Transcription Failed | "Transcription failed for chunk" | Check API limits and quota |
| Cache Error | "Could not load cached data" | Delete corrupt cache file in `transcripts/` |
| FFmpeg Missing | "FFmpeg not found" | Install FFmpeg and ensure it's in PATH |

### Debug Mode

For detailed error information:
```bash
# Show temp directory for debugging
uv run ./transcriptor.py --show-temp-dir "URL"

# Keep audio files for inspection
uv run ./transcriptor.py --keep-audio "URL"
```

### API Rate Limits

- **OpenAI Whisper**: 50 requests per minute (default tier)
- **OpenAI GPT**: Varies by model and tier
- **Anthropic Claude**: Varies by model and tier

---

## Development

### Project Structure

```
.transcriptor/
├── transcriptor.py           # Main application entry point
├── format_summary.py         # Display formatting utilities
├── pyproject.toml           # Project configuration and dependencies
├── .env.example             # Environment variable template
├── README.md                # This documentation
├── SYSTEM_SPECIFICATION.md  # Technical specification
└── transcripts/             # Cache directory (auto-created)
```

### Extension Points

The system is designed for extensibility:

**Plugin Architecture (Proposed):**
```python
class TranscriptorPlugin:
    def pre_transcription(self, audio_file: str) -> str
    def post_transcription(self, transcript: str) -> str
    def pre_summary(self, transcript: str) -> str
    def post_summary(self, summary: str) -> str
    def custom_command(self, *args, **kwargs) -> Any
```

**Provider Interface:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def summarize(self, text: str, options: dict) -> str
    
    @abstractmethod
    def answer_question(self, question: str, context: str) -> str
```

### Future Enhancements

**Planned Features:**
1. **Additional LLM Providers**: Google Gemini, Local LLMs (Ollama)
2. **Advanced Features**: Batch processing, Playlist support, Real-time transcription
3. **Export Capabilities**: PDF generation, Notion/Obsidian integration
4. **UI Enhancements**: Web interface, GUI application, Browser extension

**Integration Opportunities:**
- CI/CD pipelines for automated documentation
- Slack/Discord bot integration
- Content management system connectors

### Contributing

Contributions are welcome! Please ensure:
1. Code follows existing style patterns
2. Documentation is updated for new features
3. Error handling is comprehensive
4. Caching strategy is maintained

---

## Security Considerations

### API Key Management
- Environment variable isolation
- .env file gitignore protection
- No hardcoded credentials
- Runtime validation only

### Data Privacy
- Local cache storage only
- No telemetry or analytics
- User-controlled data retention
- Session isolation

### Input Validation
- URL format validation
- Video ID extraction verification
- File size limits enforcement
- API response validation

---

## License

This tool is for educational and personal use. Please respect YouTube's Terms of Service and content creators' rights when using this tool.

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review cached transcripts in `transcripts/` directory

---

*YouTube Transcriptor - Transform video content into actionable intelligence*