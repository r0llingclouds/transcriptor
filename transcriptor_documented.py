#!/usr/bin/env python3
"""
YouTube Transcriptor - Advanced Video Content Intelligence System

This module provides a comprehensive solution for downloading, transcribing, summarizing,
and interacting with YouTube video content through AI-powered analysis.

Main Features:
    - YouTube video audio extraction via yt-dlp
    - Automatic audio chunking for large files (>24MB)
    - High-accuracy transcription using OpenAI Whisper API
    - Multi-provider LLM support (OpenAI GPT, Anthropic Claude)
    - Interactive Q&A sessions with conversation context
    - Intelligent caching system for performance optimization
    - Rich terminal UI with progress tracking and formatting

Architecture:
    The system follows a modular design with clear separation of concerns:
    1. Audio Extraction Layer (YouTubeAudioExtractor)
    2. Transcription Engine (ChunkedWhisperTranscriber)
    3. Summarization Engine (Summarizer)
    4. Q&A Handler (QAHandler)
    5. Display Formatting (format_summary module)
    6. CLI Interface (Click framework)

Usage:
    Basic transcription and summary:
        $ ./transcriptor.py "https://youtube.com/watch?v=VIDEO_ID"
    
    Quick question mode:
        $ ./transcriptor.py "URL" "What is the main topic?"
    
    Interactive Q&A:
        $ ./transcriptor.py --qa "URL"

Author: YouTube Transcriptor Team
Version: 1.0.0
License: MIT
"""

import os
import re
import sys
import tempfile
from typing import Optional, List, Dict, Tuple, Any, Union
from pathlib import Path

# Third-party imports
import click
import yt_dlp
from openai import OpenAI
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from dotenv import load_dotenv

# Import enhanced formatting functions with graceful fallback
try:
    from format_summary import print_summary_enhanced, print_summary_cards
except ImportError:
    # Fallback to None if formatting module not available
    print_summary_enhanced = None
    print_summary_cards = None

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console for terminal output
console = Console()

# ================================================================================
# GLOBAL CONFIGURATION
# ================================================================================
# These values can be overridden via environment variables for flexibility

# LLM Generation Parameters
LLM_TEMPERATURE = float(os.getenv("TRANSCRIPTOR_LLM_TEMPERATURE", "0.3"))  # Controls randomness (0=deterministic, 1=creative)
LLM_MAX_TOKENS = int(os.getenv("TRANSCRIPTOR_LLM_MAX_TOKENS", "3000"))     # Maximum response length

# Model Selection
OPENAI_CHAT_MODEL = os.getenv("TRANSCRIPTOR_OPENAI_CHAT_MODEL", "gpt-4o-mini")  # OpenAI chat model for summaries/Q&A
OPENAI_WHISPER_MODEL = os.getenv("TRANSCRIPTOR_OPENAI_WHISPER_MODEL", "whisper-1")  # Whisper model for transcription
ANTHROPIC_MODEL = os.getenv("TRANSCRIPTOR_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")  # Claude model


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def sanitize_filename(title: str, max_length: int = 50) -> str:
    """
    Sanitize a video title for safe use as a filename.
    
    This function ensures that video titles can be safely used as filenames
    across different operating systems by removing special characters and
    limiting length.
    
    Args:
        title (str): The original video title to sanitize
        max_length (int): Maximum allowed length for the filename (default: 50)
    
    Returns:
        str: A sanitized filename safe for filesystem use
        
    Process:
        1. Replace spaces with underscores for readability
        2. Remove all special characters except alphanumeric, underscore, hyphen
        3. Collapse multiple consecutive underscores
        4. Truncate to maximum length
        5. Remove trailing underscores
        6. Return "untitled" if result is empty
    
    Examples:
        >>> sanitize_filename("Hello World! @#$")
        'Hello_World'
        >>> sanitize_filename("A" * 100)
        'A' * 50  # Truncated to max_length
    """
    if not title:
        return "untitled"
    
    # Replace spaces with underscores for better readability
    sanitized = title.replace(" ", "_")
    
    # Keep only alphanumeric characters, underscores, and hyphens
    # This ensures compatibility across all filesystems
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    
    # Remove multiple consecutive underscores for cleaner names
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Truncate to maximum length to avoid filesystem limitations
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove trailing underscores for aesthetic purposes
    sanitized = sanitized.rstrip('_')
    
    # If nothing remains after sanitization, use default
    if not sanitized:
        return "untitled"
    
    return sanitized


# ================================================================================
# YOUTUBE AUDIO EXTRACTOR CLASS
# ================================================================================

class YouTubeAudioExtractor:
    """
    Handles YouTube video audio extraction and metadata retrieval.
    
    This class manages the complete lifecycle of downloading audio from YouTube videos,
    including URL validation, audio extraction, format conversion, and metadata collection.
    It uses yt-dlp as the backend for robust video/audio downloading.
    
    Attributes:
        session_temp_dir (str): Path to the temporary directory for this session
        
    Key Features:
        - Multiple YouTube URL format support
        - Automatic MP3 conversion at 192kbps
        - Metadata extraction (title, duration, uploader, etc.)
        - Session-based temporary file management
        - Automatic cleanup on completion
    """
    
    def __init__(self):
        """
        Initialize the YouTube audio extractor with a session-specific temp directory.
        
        Creates an isolated temporary directory for this session to store
        downloaded audio files and intermediate processing artifacts.
        """
        # Create a unique temp directory for this session
        # Prefix helps identify directories in system temp folder
        self.session_temp_dir = tempfile.mkdtemp(prefix="yt_transcriber_")
        
    def cleanup(self):
        """
        Clean up all temporary files created during this session.
        
        Safely removes the entire session temporary directory and all its contents.
        Uses ignore_errors=True to handle cases where files might be locked or
        already deleted.
        """
        if hasattr(self, 'session_temp_dir') and os.path.exists(self.session_temp_dir):
            import shutil
            # Remove entire directory tree, ignoring any errors
            shutil.rmtree(self.session_temp_dir, ignore_errors=True)
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract the video ID from various YouTube URL formats.
        
        Supports multiple YouTube URL patterns including standard watch URLs,
        shortened youtu.be links, embedded URLs, and direct video IDs.
        
        Args:
            url (str): YouTube URL or video ID
            
        Returns:
            Optional[str]: 11-character video ID if found, None otherwise
            
        Supported Formats:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - https://www.youtube.com/embed/VIDEO_ID
            - https://www.youtube.com/v/VIDEO_ID
            - Direct 11-character video ID
            
        Examples:
            >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
            >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
            >>> extract_video_id("dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
        """
        # Define regex patterns for various YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',      # Standard and /v/ URLs
            r'(?:embed\/)([0-9A-Za-z_-]{11})',       # Embedded player URLs
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})',     # Watch URLs
            r'youtu\.be\/([0-9A-Za-z_-]{11})',       # Shortened URLs
        ]
        
        # Try each pattern to find a match
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Check if the input is already a valid video ID (11 characters)
        if re.match(r'^[0-9A-Za-z_-]{11}$', url):
            return url
        
        return None
    
    def download_audio(self, url: str, output_path: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Download audio from a YouTube video and extract metadata.
        
        Uses yt-dlp to download the best available audio quality and converts
        it to MP3 format at 192kbps for optimal balance between quality and file size.
        
        Args:
            url (str): YouTube video URL
            output_path (str, optional): Custom output path for the audio file.
                                        If None, uses session temp directory.
        
        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing:
                - str: Path to the downloaded MP3 file
                - Dict: Video metadata including:
                    - title: Video title
                    - duration: Duration in seconds
                    - uploader: Channel/uploader name
                    - view_count: Number of views
                    - upload_date: Upload date (YYYYMMDD format)
        
        Raises:
            yt_dlp.utils.DownloadError: If video cannot be downloaded
            
        Process:
            1. Configure yt-dlp options for audio extraction
            2. Download audio in best available quality
            3. Convert to MP3 using FFmpeg
            4. Extract and return metadata
        """
        if not output_path:
            # Use session temp directory with template for filename
            # %(title)s and %(ext)s are yt-dlp template variables
            output_path = os.path.join(self.session_temp_dir, '%(title)s.%(ext)s')
        
        # Configure yt-dlp options for optimal audio extraction
        ydl_opts = {
            'format': 'bestaudio/best',  # Get best audio quality available
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',  # Use FFmpeg for audio extraction
                'preferredcodec': 'mp3',      # Convert to MP3 format
                'preferredquality': '192',    # 192kbps quality (good balance)
            }],
            'outtmpl': output_path,      # Output file template
            'quiet': True,                # Suppress yt-dlp output
            'no_warnings': True,          # Don't show warnings
            'extract_flat': False,        # Actually download, don't just extract info
        }
        
        # Download the audio using yt-dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info and download in one operation
            info = ydl.extract_info(url, download=True)
            
            # Get the actual output filename (before MP3 conversion)
            filename = ydl.prepare_filename(info)
            # Replace extension with mp3 (post-processor changes it)
            audio_file = filename.rsplit('.', 1)[0] + '.mp3'
            
            # Extract and return relevant metadata
            return audio_file, {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),  # Duration in seconds
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', ''),  # YYYYMMDD format
            }


# ================================================================================
# CHUNKED WHISPER TRANSCRIBER CLASS
# ================================================================================

class ChunkedWhisperTranscriber:
    """
    Handles audio transcription using OpenAI's Whisper API with automatic chunking.
    
    This class manages the transcription of audio files using OpenAI's Whisper API,
    automatically splitting large files into chunks to stay within API limits (25MB).
    It provides progress tracking and maintains transcript continuity across chunks.
    
    Attributes:
        api_key (str): OpenAI API key for authentication
        client (OpenAI): OpenAI client instance
        max_size_mb (int): Maximum file size before chunking (24MB, under 25MB limit)
        session_temp_dir (str): Temporary directory for chunk storage
        
    Key Features:
        - Automatic file size detection and chunking
        - Progress tracking for long transcriptions
        - Language-specific transcription support
        - Chunk-level transcript preservation
        - Seamless chunk merging
    """
    
    def __init__(self, api_key: str = None, session_temp_dir: str = None):
        """
        Initialize the Whisper transcriber with API credentials.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, reads from environment.
            session_temp_dir (str, optional): Temp directory for chunks. Creates new if None.
            
        Raises:
            ValueError: If no API key is found in arguments or environment
        """
        # Get API key from argument or environment variable
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or use --api-key")
        
        # Initialize OpenAI client with API key
        self.client = OpenAI(api_key=self.api_key)
        
        # Set maximum file size to stay safely under Whisper's 25MB limit
        self.max_size_mb = 24  # Conservative limit to ensure success
        
        # Use provided temp directory or create a new one
        self.session_temp_dir = session_temp_dir or tempfile.mkdtemp(prefix="yt_transcriber_")
    
    def split_audio(self, audio_file: str, progress_task=None, progress=None) -> List[str]:
        """
        Split large audio files into chunks under 24MB for Whisper API compliance.
        
        Dynamically calculates the optimal number of chunks based on file size
        and splits the audio evenly to maintain context continuity.
        
        Args:
            audio_file (str): Path to the audio file to split
            progress_task: Rich progress task for updating status
            progress: Rich progress instance for UI updates
            
        Returns:
            List[str]: List of paths to audio chunk files
            
        Process:
            1. Check if file needs splitting (>24MB)
            2. Calculate optimal number of chunks
            3. Split audio into equal duration chunks
            4. Export each chunk as separate MP3 file
            5. Return list of chunk file paths
        """
        # Load the audio file using pydub
        audio = AudioSegment.from_mp3(audio_file)
        
        # Calculate file size in megabytes
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        
        # If file is small enough, no need to split
        if file_size_mb <= self.max_size_mb:
            return [audio_file]
        
        # Calculate how many chunks we need
        # Add 1 to ensure we're under the limit per chunk
        num_chunks = int(file_size_mb / self.max_size_mb) + 1
        
        # Calculate duration per chunk in milliseconds
        chunk_duration_ms = len(audio) // num_chunks
        
        chunks = []
        # Create a subdirectory for chunks with unique identifier
        # Uses process ID and object ID to ensure uniqueness
        temp_dir = os.path.join(self.session_temp_dir, f"chunks_{os.getpid()}_{id(self)}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Update progress if available
        if progress:
            progress.update(progress_task, description=f"Splitting audio into {num_chunks} chunks...")
        
        # Split the audio into chunks
        for i in range(num_chunks):
            # Calculate start and end time for this chunk
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, len(audio))
            
            # Extract the chunk from the audio
            chunk = audio[start_ms:end_ms]
            
            # Save chunk to a file
            chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunks.append(chunk_file)
            
            # Update progress with chunk details
            if progress:
                progress.update(progress_task, 
                              description=f"Created chunk {i+1}/{num_chunks} ({(end_ms-start_ms)/1000:.0f}s)")
        
        return chunks
    
    def transcribe_chunk(self, audio_file: str, language: Optional[str] = None) -> str:
        """
        Transcribe a single audio chunk using Whisper API.
        
        Args:
            audio_file (str): Path to the audio chunk file
            language (Optional[str]): ISO 639-1 language code (e.g., 'en', 'es', 'fr')
                                      If None, Whisper auto-detects the language
            
        Returns:
            str: Transcribed text from the audio chunk
            
        Raises:
            Exception: If transcription fails with API error details
            
        API Parameters:
            - model: Whisper model to use (whisper-1)
            - file: Audio file binary data
            - response_format: 'text' for plain text output
            - language: Optional language hint for better accuracy
        """
        try:
            # Open and read the audio file in binary mode
            with open(audio_file, 'rb') as audio:
                # Prepare API parameters
                params = {
                    'model': OPENAI_WHISPER_MODEL,  # Use configured Whisper model
                    'file': audio,                   # Audio file handle
                    'response_format': 'text',       # Get plain text response
                }
                
                # Add language parameter if specified
                if language:
                    params['language'] = language
                
                # Call Whisper API for transcription
                transcript = self.client.audio.transcriptions.create(**params)
                return transcript
                
        except Exception as e:
            # Re-raise with more context about the failure
            raise Exception(f"Transcription failed for chunk: {str(e)}")
    
    def transcribe(self, audio_file: str, language: Optional[str] = None, 
                  progress_task=None, progress=None) -> str:
        """
        Transcribe an audio file with automatic chunking for large files.
        
        This is the main entry point for transcription. It handles the complete
        transcription workflow including chunking, parallel processing, and
        result merging.
        
        Args:
            audio_file (str): Path to the audio file to transcribe
            language (Optional[str]): Language code for transcription
            progress_task: Rich progress task for status updates
            progress: Rich progress instance for UI
            
        Returns:
            str: Complete transcribed text
            
        Process:
            1. Split audio into chunks if necessary
            2. Transcribe each chunk sequentially
            3. Save individual chunk transcripts
            4. Merge all transcripts into final result
            5. Save merged transcript
        """
        # Split the audio file into chunks if needed
        chunks = self.split_audio(audio_file, progress_task, progress)
        
        # Handle single chunk (small file) case
        if len(chunks) == 1:
            if progress:
                progress.update(progress_task, description="Transcribing audio...")
            return self.transcribe_chunk(chunks[0], language)
        
        # Handle multiple chunks (large file) case
        transcripts = []
        for i, chunk in enumerate(chunks, 1):
            # Update progress for each chunk
            if progress:
                progress.update(progress_task, 
                              description=f"Transcribing chunk {i}/{len(chunks)}...")
            
            # Transcribe the chunk
            transcript = self.transcribe_chunk(chunk, language)
            transcripts.append(transcript)
            
            # Save individual chunk transcript for debugging/recovery
            if self.session_temp_dir and os.path.exists(self.session_temp_dir):
                chunk_transcript_file = os.path.join(
                    os.path.dirname(chunk),  # Same directory as audio chunks
                    f"chunk_{i}_transcript.txt"
                )
                try:
                    with open(chunk_transcript_file, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                except Exception:
                    pass  # Silent fail - not critical for operation
            
            # Note: Chunks are not deleted here - cleaned up with session dir
        
        # Merge all chunk transcripts with space separator
        merged_transcript = " ".join(transcripts)
        
        # Save the merged transcript for debugging/recovery
        if self.session_temp_dir and os.path.exists(self.session_temp_dir):
            merged_transcript_file = os.path.join(
                self.session_temp_dir,
                "merged_transcript.txt"
            )
            try:
                with open(merged_transcript_file, 'w', encoding='utf-8') as f:
                    f.write(merged_transcript)
            except Exception:
                pass  # Silent fail - not critical for operation
        
        return merged_transcript


# ================================================================================
# SUMMARIZER CLASS
# ================================================================================

class Summarizer:
    """
    Multi-provider LLM summarization engine for transcript analysis.
    
    Provides flexible summarization capabilities using either OpenAI GPT or
    Anthropic Claude models. Supports multiple detail levels and handles
    very long transcripts through intelligent chunking and aggregation.
    
    Attributes:
        provider (str): LLM provider ('openai' or 'anthropic')
        api_key (str): API key for the selected provider
        client: Provider-specific client instance
        
    Features:
        - Multiple detail levels (brief, medium, detailed)
        - Provider-agnostic interface
        - Automatic handling of long transcripts
        - Template-based prompt engineering
        - Chunk-and-aggregate strategy for very long content
    """
    
    def __init__(self, api_key: str = None, provider: str = 'openai'):
        """
        Initialize the summarizer with specified LLM provider.
        
        Args:
            api_key (str, optional): API key for the provider. Uses environment if None.
            provider (str): LLM provider - 'openai' or 'anthropic'
            
        Raises:
            ValueError: If API key is not found for the selected provider
        """
        self.provider = provider
        
        if provider == 'openai':
            # Initialize OpenAI client
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        else:
            # Initialize Anthropic client
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("Anthropic API key not found")
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
    
    def summarize(self, text: str, detail_level: str = "medium") -> str:
        """
        Generate a summary of the transcript at the specified detail level.
        
        Handles very long transcripts by splitting into chunks, summarizing each,
        then creating a final aggregated summary.
        
        Args:
            text (str): Transcript text to summarize
            detail_level (str): Level of detail - 'brief', 'medium', or 'detailed'
            
        Returns:
            str: Generated summary at the requested detail level
            
        Detail Levels:
            - brief: 2-3 sentence executive summary
            - medium: Structured summary with main points and conclusions
            - detailed: Comprehensive analysis with themes, evidence, and implications
        """
        # Define prompt templates for different detail levels
        prompts = {
            "brief": """Provide a very brief summary (2-3 sentences) of the following transcript. 
Focus only on the main topic and conclusion.""",
            
            "medium": """Provide a clear summary of the following transcript with:
1. Main topic (1-2 sentences)
2. Key points (3-5 bullet points)
3. Important details or examples mentioned
4. Conclusion or main takeaway""",
            
            "detailed": """Provide a comprehensive summary of the following transcript with:
1. Overview of the main topic
2. All key points discussed (organized by theme)
3. Important examples, data, or evidence presented
4. Notable quotes or insights
5. Conclusions and implications
6. Any action items or recommendations mentioned"""
        }
        
        # Get the appropriate prompt template
        prompt = prompts.get(detail_level, prompts["medium"])
        
        # Handle very long transcripts by chunking
        max_context = 100000  # Conservative token limit estimate
        
        if len(text) > max_context:
            # Split transcript into manageable chunks
            chunks = [text[i:i+max_context] for i in range(0, len(text), max_context)]
            summaries = []
            
            # Summarize each chunk
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"{prompt}\n\n[Part {i+1} of {len(chunks)}]\n\nTranscript:\n{chunk}"
                
                # Use provider-specific method
                if self.provider == 'openai':
                    summary = self._summarize_openai(chunk_prompt)
                else:
                    summary = self._summarize_anthropic(chunk_prompt)
                summaries.append(summary)
            
            # Create final aggregated summary from chunk summaries
            final_prompt = f"Combine these partial summaries into one coherent summary:\n\n" + "\n\n---\n\n".join(summaries)
            
            if self.provider == 'openai':
                return self._summarize_openai(final_prompt)
            else:
                return self._summarize_anthropic(final_prompt)
        else:
            # Process normally for shorter transcripts
            full_prompt = f"{prompt}\n\nTranscript:\n{text}"
            
            if self.provider == 'openai':
                return self._summarize_openai(full_prompt)
            else:
                return self._summarize_anthropic(full_prompt)
    
    def _summarize_openai(self, prompt: str) -> str:
        """
        Generate summary using OpenAI's GPT model.
        
        Args:
            prompt (str): Complete prompt including instructions and transcript
            
        Returns:
            str: Generated summary from GPT model
        """
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that creates clear, structured summaries of video transcripts."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,  # Control randomness
            max_tokens=LLM_MAX_TOKENS    # Limit response length
        )
        return response.choices[0].message.content
    
    def _summarize_anthropic(self, prompt: str) -> str:
        """
        Generate summary using Anthropic's Claude model.
        
        Args:
            prompt (str): Complete prompt including instructions and transcript
            
        Returns:
            str: Generated summary from Claude model
        """
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system="You are a helpful assistant that creates clear, structured summaries of video transcripts.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


# ================================================================================
# Q&A HANDLER CLASS
# ================================================================================

class QAHandler:
    """
    Manages interactive and non-interactive Q&A sessions with conversation context.
    
    Provides capabilities for both single questions and full interactive sessions
    with persistent conversation history. Supports multiple LLM providers and
    maintains context across questions for coherent conversations.
    
    Attributes:
        provider (str): LLM provider ('openai' or 'anthropic')
        qa_history (List[Dict]): Conversation history for context
        sessions_dir (str): Directory for storing session data
        api_key (str): API key for the provider
        client: Provider-specific client instance
        
    Features:
        - Interactive Q&A sessions with commands
        - Quick single-question mode
        - Conversation context management
        - Session persistence and restoration
        - Multi-provider support
        - Auto-save functionality
    """
    
    def __init__(self, api_key: str = None, provider: str = 'openai'):
        """
        Initialize the Q&A handler with specified provider.
        
        Args:
            api_key (str, optional): API key for the provider
            provider (str): LLM provider - 'openai' or 'anthropic'
            
        Raises:
            ValueError: If API key is not found
        """
        self.provider = provider
        self.qa_history = []  # Initialize empty conversation history
        self.sessions_dir = "transcripts"  # Directory for session storage
        
        # Initialize provider-specific client
        if provider == 'openai':
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("Anthropic API key not found")
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
    
    def answer_question(self, question: str, transcript: str, video_info: dict, 
                       conversation_history: list = None) -> str:
        """
        Answer a question based on the transcript with optional conversation context.
        
        Generates contextual answers using the full transcript and any previous
        Q&A exchanges in the conversation history.
        
        Args:
            question (str): The user's question
            transcript (str): Full video transcript for context
            video_info (dict): Video metadata (title, duration, uploader)
            conversation_history (list, optional): Previous Q&A exchanges
            
        Returns:
            str: AI-generated answer to the question
            
        System Prompt Features:
            - Video metadata context
            - Full transcript access
            - Conversation continuity
            - Citation encouragement
            - Clear admission when information not available
        """
        # Construct system prompt with video context
        system_prompt = f"""You are an AI assistant helping users understand video content. 
You have access to the full transcript of a video titled "{video_info.get('title', 'Unknown')}" 
by {video_info.get('uploader', 'Unknown')} (duration: {video_info.get('duration', 0)}s).

Answer questions based on the transcript content. Be specific and cite relevant parts when appropriate.
If the answer isn't in the transcript, say so clearly.
You can reference previous questions and answers in this conversation to provide contextual responses.

Transcript:
{transcript}"""
        
        # Route to provider-specific implementation
        if self.provider == 'openai':
            return self._answer_openai(system_prompt, question, conversation_history)
        else:
            return self._answer_anthropic(system_prompt, question, conversation_history)
    
    def _answer_openai(self, system_prompt: str, question: str, 
                      conversation_history: list = None) -> str:
        """
        Generate answer using OpenAI's GPT model with conversation context.
        
        Args:
            system_prompt (str): System message with transcript context
            question (str): Current question to answer
            conversation_history (list): Previous Q&A exchanges
            
        Returns:
            str: Generated answer from GPT model
        """
        # Build messages array starting with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for qa in conversation_history:
                messages.append({"role": "user", "content": qa["question"]})
                messages.append({"role": "assistant", "content": qa["answer"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Generate response with conversation context
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content
    
    def _answer_anthropic(self, system_prompt: str, question: str, 
                         conversation_history: list = None) -> str:
        """
        Generate answer using Anthropic's Claude model with conversation context.
        
        Args:
            system_prompt (str): System message with transcript context
            question (str): Current question to answer
            conversation_history (list): Previous Q&A exchanges
            
        Returns:
            str: Generated answer from Claude model
        """
        # Build messages array for Claude
        messages = []
        
        # Add conversation history if available
        if conversation_history:
            for qa in conversation_history:
                messages.append({"role": "user", "content": qa["question"]})
                messages.append({"role": "assistant", "content": qa["answer"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Generate response with conversation context
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,  # Claude uses separate system parameter
            messages=messages
        )
        return response.content[0].text
    
    def quick_answer(self, question: str, transcript: str, video_info: dict) -> str:
        """
        Get a quick answer without loading conversation history.
        
        Simplified interface for single-question scenarios where
        conversation context is not needed.
        
        Args:
            question (str): The question to answer
            transcript (str): Video transcript
            video_info (dict): Video metadata
            
        Returns:
            str: Generated answer
        """
        # Use answer_question with no history for a single response
        return self.answer_question(question, transcript, video_info, conversation_history=None)
    
    def get_session_filename(self, video_id: str, title: str = None) -> str:
        """
        Generate the unified cache filename for a video.
        
        Creates a consistent filename for storing session data that matches
        the transcript cache naming convention.
        
        Args:
            video_id (str): YouTube video ID
            title (str, optional): Video title for filename
            
        Returns:
            str: Full path to the session file
        """
        if title:
            sanitized_title = sanitize_filename(title)
            return os.path.join(self.sessions_dir, f"{video_id}_{sanitized_title}.json")
        else:
            # Fallback for videos without title
            return os.path.join(self.sessions_dir, f"{video_id}.json")
    
    def load_session(self, video_id: str, title: str = None) -> bool:
        """
        Load existing Q&A session from unified cache file.
        
        Attempts to restore previous conversation history from cached session data.
        Uses glob pattern matching to find cache files by video ID.
        
        Args:
            video_id (str): YouTube video ID
            title (str, optional): Video title for exact match
            
        Returns:
            bool: True if session with Q&A history was loaded, False otherwise
        """
        import json
        import glob
        
        # Try to find unified cache file by video ID
        session_file = None
        if title:
            # Try with title first for exact match
            session_file = self.get_session_filename(video_id, title)
        
        # If not found or no title, try glob search
        if not session_file or not os.path.exists(session_file):
            cache_files = glob.glob(f"transcripts/{video_id}_*.json")
            if cache_files:
                session_file = cache_files[0]  # Use first match
        
        # Try to load QA history from unified file
        if session_file and os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.qa_history = data.get('qa_history', [])
                    # Only return True if there's actual QA history
                    return len(self.qa_history) > 0
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load session: {e}[/yellow]")
        
        return False
    
    def append_qa_to_history(self, question: str, answer: str, video_id: str, 
                            video_info: dict) -> int:
        """
        Append a single Q&A exchange to history and save immediately.
        
        Used for quick questions to maintain history without full session.
        Merges with existing cache data to preserve all information.
        
        Args:
            question (str): The question asked
            answer (str): The generated answer
            video_id (str): Video ID for cache file
            video_info (dict): Video metadata
            
        Returns:
            int: Total number of Q&A exchanges in history
        """
        import json
        from datetime import datetime
        import glob
        
        # Ensure directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        title = video_info.get('title', '')
        
        # Find or create cache file
        session_file = None
        if title:
            session_file = self.get_session_filename(video_id, title)
        
        # If not found, try glob search
        if not session_file or not os.path.exists(session_file):
            cache_files = glob.glob(f"transcripts/{video_id}_*.json")
            if cache_files:
                session_file = cache_files[0]
            else:
                # Create new file if none exists
                session_file = self.get_session_filename(video_id, title)
        
        # Load existing data
        existing_data = {}
        existing_qa_history = []
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_qa_history = existing_data.get('qa_history', [])
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing cache: {e}[/yellow]")
        
        # Append new Q&A
        existing_qa_history.append({"question": question, "answer": answer})
        
        # Merge with existing data
        merged_data = existing_data.copy()
        merged_data.update({
            'video_info': existing_data.get('video_info', video_info),
            'transcript': existing_data.get('transcript', ''),
            'qa_history': existing_qa_history,
            'last_updated': datetime.now().isoformat()
        })
        
        # Save back to file
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            return len(existing_qa_history)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save Q&A: {e}[/yellow]")
            return 0
    
    def save_session_json(self, video_id: str, video_info: dict):
        """
        Save complete Q&A session to unified cache file.
        
        Merges current session history with existing cache data,
        preserving transcripts, summaries, and other metadata.
        
        Args:
            video_id (str): Video ID for cache file
            video_info (dict): Video metadata
        """
        import json
        from datetime import datetime
        
        # Skip if no history to save
        if not self.qa_history:
            return
        
        # Ensure directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        title = video_info.get('title', '')
        session_file = self.get_session_filename(video_id, title)
        
        # Try to load existing cache data first
        existing_data = {}
        existing_qa_history = []
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # Get existing QA history that might not be in our current session
                    existing_qa_history = existing_data.get('qa_history', [])
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing cache for QA merge: {e}[/yellow]")
        
        # Use current session history as complete history
        # (assumes it was loaded at start if resuming)
        merged_qa_history = self.qa_history
        
        # Merge QA data with existing cache, preserving other fields
        if not isinstance(existing_data, dict):
            existing_data = {}
        merged_data = existing_data.copy()
        merged_data.update({
            'video_info': existing_data.get('video_info', video_info),
            'transcript': existing_data.get('transcript', ''),
            'qa_history': merged_qa_history,
            'last_updated': datetime.now().isoformat()
        })
        
        # Save to file
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            console.print(f"[dim]✓ QA history saved ({len(merged_qa_history)} exchanges)[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save session: {e}[/yellow]")
    
    def interactive_session(self, transcript: str, video_info: dict, video_id: str):
        """
        Run a full interactive Q&A session with commands and context.
        
        Provides a rich interactive experience with session management,
        command support, and automatic saving. Supports resuming previous
        sessions and maintains conversation context throughout.
        
        Args:
            transcript (str): Full video transcript
            video_info (dict): Video metadata
            video_id (str): Video ID for session persistence
            
        Interactive Commands:
            - /quit, /exit, /q: End session and save
            - /save: Manually save current progress
            - /clear: Reset conversation context
            - /help: Display available commands
        """
        title = video_info.get('title', '')
        
        # Check for existing session and load automatically
        if self.load_session(video_id, title):
            console.print(f"[green]✓ Loaded existing Q&A session with {len(self.qa_history)} previous exchanges[/green]")
            
            # Show last Q&A to refresh context
            if self.qa_history:
                from rich.text import Text
                from rich.console import Group
                
                last_qa = self.qa_history[-1]
                # Truncate answer preview if too long
                answer_preview = last_qa['answer'][:400] + "..." if len(last_qa['answer']) > 400 else last_qa['answer']
                
                # Create formatted preview
                question_text = Text()
                question_text.append("Last Question: ", style="bold cyan")
                question_text.append(last_qa['question'])
                
                prev_content = Group(
                    question_text,
                    Text(),  # Empty line
                    Text("Last Answer: ", style="bold green"),
                    Markdown(answer_preview)
                )
                
                # Display previous context panel
                console.print(Panel(
                    prev_content,
                    title="[dim]Previous Context[/dim]",
                    border_style="dim"
                ))
        
        # Display session information and commands
        console.print(Panel.fit(
            f"[bold green]Interactive Q&A Session[/bold green]\n"
            f"Video: {video_info.get('title', 'Unknown')}\n"
            f"Duration: {video_info.get('duration', 0)}s | Uploader: {video_info.get('uploader', 'Unknown')}\n\n"
            f"[dim]Commands:[/dim]\n"
            f"  • Type your question and press Enter\n"
            f"  • [cyan]/quit[/cyan], [cyan]/exit[/cyan], or [cyan]/q[/cyan] - End session\n"
            f"  • [cyan]/save[/cyan] - Save Q&A history\n"
            f"  • [cyan]/clear[/cyan] - Clear conversation context\n"
            f"  • [cyan]/help[/cyan] - Show this help",
            border_style="green"
        ))
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")
                
                # Handle commands
                if question.lower() in ['/quit', '/exit', '/q']:
                    # Save and exit
                    self.save_session_json(video_id, video_info)
                    console.print("[yellow]Ending Q&A session...[/yellow]")
                    console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id, title)}[/green]")
                    break
                
                elif question.lower() == '/save':
                    # Manual save
                    self.save_session_json(video_id, video_info)
                    console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id, title)}[/green]")
                    continue
                
                elif question.lower() == '/clear':
                    # Clear conversation context
                    self.qa_history = []
                    console.print("[yellow]Conversation context cleared[/yellow]")
                    continue
                
                elif question.lower() == '/help':
                    # Show help
                    console.print(Panel.fit(
                        "[dim]Commands:[/dim]\n"
                        "  • Type your question and press Enter\n"
                        "  • [cyan]/quit[/cyan], [cyan]/exit[/cyan], or [cyan]/q[/cyan] - End session\n"
                        "  • [cyan]/save[/cyan] - Save Q&A history\n"
                        "  • [cyan]/clear[/cyan] - Clear conversation context\n"
                        "  • [cyan]/help[/cyan] - Show this help",
                        border_style="dim"
                    ))
                    continue
                
                # Process the question with conversation history
                with console.status("[bold green]Thinking..."):
                    answer = self.answer_question(question, transcript, video_info, self.qa_history)
                
                # Store in history
                self.qa_history.append({"question": question, "answer": answer})
                
                # Auto-save session after each Q&A
                self.save_session_json(video_id, video_info)
                
                # Display Q&A with proper markdown rendering
                from rich.text import Text
                from rich.console import Group
                
                # Create formatted Q&A display
                question_text = Text()
                question_text.append("Question: ", style="bold cyan")
                question_text.append(question)
                
                # Combine question and answer with markdown support
                qa_content = Group(
                    question_text,
                    Text(),  # Empty line
                    Text("Answer: ", style="bold green"),
                    Markdown(answer)
                )
                
                # Display in panel
                console.print(Panel(
                    qa_content,
                    title="[bold]Q&A[/bold]",
                    border_style="bright_cyan"
                ))
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                console.print("\n[yellow]Session interrupted. Saving and exiting...[/yellow]")
                self.save_session_json(video_id, video_info)
                console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id)}[/green]")
                break
                
            except Exception as e:
                # Handle other errors
                console.print(f"[red]Error: {str(e)}[/red]")


# ================================================================================
# MAIN CLI INTERFACE
# ================================================================================

@click.command()
@click.argument('video_url')
@click.argument('question', required=False)  # Optional for shorthand syntax
@click.option('--api-key', help='OpenAI API key for transcription and summarization')
@click.option('--language', help='Language code for transcription (e.g., en, es, fr)')
@click.option('--detail', type=click.Choice(['brief', 'medium', 'detailed']), 
              default='medium', help='Level of detail for summary')
@click.option('--output', '-o', help='Output file path for summary')
@click.option('--keep-audio', is_flag=True, help='Keep the downloaded audio file')
@click.option('--transcript-only', is_flag=True, help='Only transcribe without summarizing')
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), 
              default='openai', help='LLM provider for summarization')
@click.option('--show-temp-dir', is_flag=True, help='Show temp directory location and preserve it')
@click.option('--qa', is_flag=True, help='Interactive Q&A mode - ask questions about the video')
@click.option('--ask', help='Quick question mode - get a single answer without interaction')
@click.option('--display-format', type=click.Choice(['standard', 'enhanced', 'cards']), 
              default='enhanced', help='Display format for summaries')
def main(video_url: str, question: Optional[str], api_key: Optional[str], 
         language: Optional[str], detail: str, output: Optional[str], 
         keep_audio: bool, transcript_only: bool, provider: str, 
         show_temp_dir: bool, qa: bool, ask: Optional[str], display_format: str):
    """
    YouTube Transcriptor - Transform video content into actionable intelligence.
    
    This is the main entry point for the application, handling all command-line
    arguments and orchestrating the complete workflow from video download to
    final output.
    
    Args:
        video_url: YouTube video URL or video ID
        question: Optional question for shorthand syntax
        
    Workflow:
        1. Parse and validate command-line arguments
        2. Extract video ID from URL
        3. Check for cached transcript
        4. Download and transcribe if needed
        5. Execute requested operation (summary/Q&A/transcript)
        6. Save results and cleanup
    """
    # Detect shorthand syntax: "transcriptor URL QUESTION"
    if question and not ask and not qa and not transcript_only:
        ask = question  # Treat second argument as quick question
        question = None  # Clear to avoid confusion
    
    # Validate mutually exclusive options
    if qa and transcript_only:
        console.print("[red]Error: Cannot use --qa and --transcript-only together[/red]")
        sys.exit(1)
    
    if ask and (qa or transcript_only):
        console.print("[red]Error: Cannot use --ask with --qa or --transcript-only[/red]")
        sys.exit(1)
    
    # Initialize variables
    audio_file = None
    extractor = None
    
    try:
        # Create progress display for long operations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Step 1: Extract video ID from URL
            task = progress.add_task("Extracting video ID...", total=None)
            video_id = YouTubeAudioExtractor.extract_video_id(video_url)
            
            if not video_id:
                console.print("[red]Error: Could not extract video ID from URL[/red]")
                sys.exit(1)
            
            # Step 2: Check for cached transcript
            progress.update(task, description="Checking for cached transcript...")
            
            # Look for cached JSON file by video ID pattern
            import glob
            import json
            cached_files = glob.glob(f"transcripts/{video_id}_*.json")
            cache_file = cached_files[0] if cached_files else None
            
            # Initialize data variables
            video_info = None
            transcript = None
            summaries_cache = {}
            
            # Step 3: Load cached data if available
            if cache_file:
                progress.update(task, description="Loading cached data...")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # Extract cached components
                    transcript = cached_data.get('transcript', '')
                    video_info = cached_data.get('video_info', {})
                    summaries_cache = cached_data.get('summaries', {})
                    
                    console.print(f"[green]✓ Using cached transcript and video info from {cache_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load cached data: {e}[/yellow]")
                    cache_file = None
                
                audio_file = None  # No audio file when using cache
                
            # Step 4: Download and transcribe if no cache
            if not cache_file:
                # Create extractor only when needed
                extractor = YouTubeAudioExtractor()
                
                if show_temp_dir:
                    console.print(f"[cyan]Temp directory: {extractor.session_temp_dir}[/cyan]")
                
                # Download audio from YouTube
                progress.update(task, description="Downloading audio from YouTube...")
                audio_file, video_info = extractor.download_audio(video_url)
                
                # Display download information
                file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                console.print(f"[green]✓ Downloaded: {video_info['title']}[/green]")
                console.print(f"[dim]Duration: {video_info['duration']}s | Size: {file_size_mb:.1f}MB | Uploader: {video_info['uploader']}[/dim]")
                
                # Transcribe audio with chunking if needed
                progress.update(task, description="Initializing transcription...")
                
                # Pass session temp dir for chunk storage
                session_temp = extractor.session_temp_dir if extractor else None
                transcriber = ChunkedWhisperTranscriber(api_key=api_key, 
                                                       session_temp_dir=session_temp)
                transcript = transcriber.transcribe(audio_file, language=language, 
                                                   progress_task=task, progress=progress)
            
            console.print(f"[green]✓ Transcription complete ({len(transcript)} characters)[/green]")
            
            # Variable to track new summary generation
            new_summary_value = None
            
            # Step 5: Execute requested operation
            if ask:
                # Quick answer mode - single question with history saving
                progress.update(task, description="Getting answer...")
                qa_handler = QAHandler(api_key=api_key, provider=provider)
                
                # Load existing history for context
                qa_handler.load_session(video_id, video_info.get('title'))
                
                # Get the answer
                answer = qa_handler.quick_answer(ask, transcript, video_info)
                
                # Save Q&A to history
                total_qa_count = qa_handler.append_qa_to_history(ask, answer, video_id, video_info)
                
                # Display Q&A with rich formatting
                from rich.text import Text
                from rich.console import Group
                
                question_text = Text()
                question_text.append("Question: ", style="bold cyan")
                question_text.append(ask)
                
                qa_content = Group(
                    question_text,
                    Text(),  # Empty line
                    Text("Answer: ", style="bold green"),
                    Markdown(answer)
                )
                
                console.print(Panel(
                    qa_content,
                    title="[bold]Q&A[/bold]",
                    border_style="bright_cyan"
                ))
                
                # Show save confirmation
                console.print(f"[green]✓ Added to Q&A history ({total_qa_count} total exchanges)[/green]")
                
                # Optionally save to file
                if output:
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(f"# Quick Q&A: {video_info['title']}\n\n")
                        f.write(f"**Question:** {ask}\n\n")
                        f.write(f"**Answer:** {answer}\n")
                    console.print(f"[green]✓ Answer also saved to {output}[/green]")
                
                result = None  # No result for file saving
                
            elif qa:
                # Interactive Q&A mode
                progress.update(task, description="Starting Q&A session...", completed=True)
                progress.stop()  # Stop progress display for interactive mode
                
                qa_handler = QAHandler(api_key=api_key, provider=provider)
                # Start interactive session
                qa_handler.interactive_session(transcript, video_info, video_id)
                result = None  # No result for file saving
                
            elif transcript_only:
                # Transcript-only mode
                result = transcript
                if len(transcript) < 5000:  # Display if reasonably sized
                    console.print(Panel(transcript, title="Transcript", border_style="green"))
                else:
                    console.print("[yellow]Transcript is too long to display. Saving to file...[/yellow]")
                    
            else:
                # Summary mode (default)
                # Check for cached summary first
                cached_summary = None
                if isinstance(summaries_cache, dict):
                    provider_map = summaries_cache.get(provider, {})
                    if isinstance(provider_map, dict):
                        cached_summary = provider_map.get(detail)
                
                if cached_summary:
                    # Use cached summary
                    result = cached_summary
                    console.print(f"[green]✓ Using cached summary ({provider}:{detail})[/green]")
                    
                    # Display with selected format
                    if display_format == 'cards' and print_summary_cards:
                        print_summary_cards(cached_summary, video_info)
                    elif display_format == 'enhanced' and print_summary_enhanced:
                        print_summary_enhanced(cached_summary, "Summary", video_info)
                    else:
                        console.print(Panel(cached_summary, title="Summary", border_style="green"))
                else:
                    # Generate new summary
                    progress.update(task, description="Generating summary...")
                    summarizer = Summarizer(api_key=api_key, provider=provider)
                    summary = summarizer.summarize(transcript, detail_level=detail)
                    result = summary
                    new_summary_value = summary  # Track for caching
                    
                    # Display with selected format
                    if display_format == 'cards' and print_summary_cards:
                        print_summary_cards(summary, video_info)
                    elif display_format == 'enhanced' and print_summary_enhanced:
                        print_summary_enhanced(summary, "Summary", video_info)
                    else:
                        console.print(Panel(summary, title="Summary", border_style="green"))
            
            # Step 6: Save cache and results
            
            # Save new transcript to cache (if just generated)
            if not cache_file:
                os.makedirs("transcripts", exist_ok=True)
                
                # Generate cache filename
                sanitized_title = sanitize_filename(video_info['title'])
                cache_file = f"transcripts/{video_id}_{sanitized_title}.json"
                
                # Check for existing file and merge
                existing_data = {}
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        pass  # Overwrite if can't read
                
                # Prepare cache data with merging
                summaries_map = existing_data.get('summaries', {})
                if new_summary_value is not None:
                    if provider not in summaries_map or not isinstance(summaries_map.get(provider), dict):
                        summaries_map[provider] = {}
                    summaries_map[provider][detail] = new_summary_value
                
                cached_data = {
                    'video_info': video_info,
                    'transcript': transcript,
                    'qa_history': existing_data.get('qa_history', []),
                    'summaries': summaries_map,
                    'last_updated': existing_data.get('last_updated')
                }
                
                # Clean up None values
                cached_data = {k: v for k, v in cached_data.items() 
                             if v is not None and (k != 'qa_history' or v)}
                
                # Save cache file
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cached_data, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Transcript and video info cached in {cache_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not save cached data: {e}[/yellow]")

            # Save new summary to existing cache
            if cache_file and new_summary_value is not None:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = {}
                
                summaries = data.get('summaries', {})
                if provider not in summaries or not isinstance(summaries.get(provider), dict):
                    summaries[provider] = {}
                summaries[provider][detail] = new_summary_value
                data['summaries'] = summaries
                
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    console.print(f"[dim]✓ Summary cached ({provider}:{detail})[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not save summary cache: {e}[/yellow]")
            
            # Save output file if requested (skip for Q&A mode)
            if not qa and (output or (transcript_only and len(transcript) > 5000)):
                if output:
                    output_file = output
                else:
                    # Auto-generate filename
                    sanitized_title = sanitize_filename(video_info.get('title', 'untitled'))
                    file_type = 'transcript' if transcript_only else 'summary'
                    output_file = f"{video_id}_{sanitized_title}_{file_type}.md"
                
                progress.update(task, description="Saving to file...")
                
                # Write markdown file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {video_info['title']}\n\n")
                    f.write(f"**URL:** {video_url}\n")
                    f.write(f"**Duration:** {video_info['duration']}s\n")
                    f.write(f"**Uploader:** {video_info['uploader']}\n\n")
                    f.write("---\n\n")
                    if transcript_only:
                        f.write("## Transcript\n\n")
                    else:
                        f.write("## Summary\n\n")
                    f.write(result)
                    
                console.print(f"[green]✓ Saved to {output_file}[/green]")
            
            # Show completion for non-interactive modes
            if not qa:
                progress.update(task, description="Done!", completed=True)
    
    except ValueError as e:
        # Handle configuration errors
        console.print(f"[red]Configuration Error: {str(e)}[/red]")
        console.print("[yellow]Tip: Set OPENAI_API_KEY in your .env file or use --api-key option[/yellow]")
        sys.exit(1)
        
    except Exception as e:
        # Handle unexpected errors
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up temporary files
        if extractor:
            if show_temp_dir:
                console.print(f"[yellow]Temp directory preserved: {extractor.session_temp_dir}[/yellow]")
            elif keep_audio:
                console.print(f"[yellow]Audio file kept in: {extractor.session_temp_dir}[/yellow]")
            else:
                extractor.cleanup()
                console.print(f"[dim]Cleaned up all temporary files[/dim]")


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    """
    Script entry point - runs the main CLI function when executed directly.
    """
    main()