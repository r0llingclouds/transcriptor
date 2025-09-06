#!/usr/bin/env python3

import os
import re
import sys
import tempfile
from typing import Optional, List
from pathlib import Path

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

# Import enhanced formatting functions
try:
    from format_summary import print_summary_enhanced, print_summary_cards
except ImportError:
    print_summary_enhanced = None
    print_summary_cards = None

load_dotenv()

console = Console()

# Global configuration for model names and generation parameters
# Can be overridden via environment variables
LLM_TEMPERATURE = float(os.getenv("TRANSCRIPTOR_LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("TRANSCRIPTOR_LLM_MAX_TOKENS", "3000"))

OPENAI_CHAT_MODEL = os.getenv("TRANSCRIPTOR_OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_WHISPER_MODEL = os.getenv("TRANSCRIPTOR_OPENAI_WHISPER_MODEL", "whisper-1")
ANTHROPIC_MODEL = os.getenv("TRANSCRIPTOR_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize video title for use in filename"""
    if not title:
        return "untitled"
    
    # Replace spaces with underscores
    sanitized = title.replace(" ", "_")
    
    # Keep only alphanumeric, underscores, and hyphens
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove trailing underscores
    sanitized = sanitized.rstrip('_')
    
    # If nothing left, use default
    if not sanitized:
        return "untitled"
    
    return sanitized

class YouTubeAudioExtractor:
    def __init__(self):
        # Create a main temp directory for this session
        self.session_temp_dir = tempfile.mkdtemp(prefix="yt_transcriber_")
        
    def cleanup(self):
        """Clean up the entire session temp directory"""
        if hasattr(self, 'session_temp_dir') and os.path.exists(self.session_temp_dir):
            import shutil
            shutil.rmtree(self.session_temp_dir, ignore_errors=True)
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        if re.match(r'^[0-9A-Za-z_-]{11}$', url):
            return url
        
        return None
    
    def download_audio(self, url: str, output_path: str = None) -> tuple[str, dict]:
        if not output_path:
            # Use the session temp directory
            output_path = os.path.join(self.session_temp_dir, '%(title)s.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get the actual output filename
            filename = ydl.prepare_filename(info)
            # Replace extension with mp3
            audio_file = filename.rsplit('.', 1)[0] + '.mp3'
            
            return audio_file, {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', ''),
            }


class ChunkedWhisperTranscriber:
    def __init__(self, api_key: str = None, session_temp_dir: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or use --api-key")
        self.client = OpenAI(api_key=self.api_key)
        self.max_size_mb = 24  # Stay under 25MB limit
        self.session_temp_dir = session_temp_dir or tempfile.mkdtemp(prefix="yt_transcriber_")
    
    def split_audio(self, audio_file: str, progress_task=None, progress=None) -> List[str]:
        """Split audio file into chunks under 24MB"""
        audio = AudioSegment.from_mp3(audio_file)
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        
        if file_size_mb <= self.max_size_mb:
            return [audio_file]
        
        # Calculate how many chunks we need
        num_chunks = int(file_size_mb / self.max_size_mb) + 1
        chunk_duration_ms = len(audio) // num_chunks
        
        chunks = []
        # Create a subdirectory for chunks in the session temp dir
        temp_dir = os.path.join(self.session_temp_dir, f"chunks_{os.getpid()}_{id(self)}")
        os.makedirs(temp_dir, exist_ok=True)
        
        if progress:
            progress.update(progress_task, description=f"Splitting audio into {num_chunks} chunks...")
        
        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, len(audio))
            
            chunk = audio[start_ms:end_ms]
            chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunks.append(chunk_file)
            
            if progress:
                progress.update(progress_task, 
                              description=f"Created chunk {i+1}/{num_chunks} ({(end_ms-start_ms)/1000:.0f}s)")
        
        return chunks
    
    def transcribe_chunk(self, audio_file: str, language: Optional[str] = None) -> str:
        """Transcribe a single audio chunk"""
        try:
            with open(audio_file, 'rb') as audio:
                params = {
                    'model': OPENAI_WHISPER_MODEL,
                    'file': audio,
                    'response_format': 'text',
                }
                if language:
                    params['language'] = language
                
                transcript = self.client.audio.transcriptions.create(**params)
                return transcript
        except Exception as e:
            raise Exception(f"Transcription failed for chunk: {str(e)}")
    
    def transcribe(self, audio_file: str, language: Optional[str] = None, 
                  progress_task=None, progress=None) -> str:
        """Transcribe audio file, splitting if necessary"""
        chunks = self.split_audio(audio_file, progress_task, progress)
        
        if len(chunks) == 1:
            if progress:
                progress.update(progress_task, description="Transcribing audio...")
            return self.transcribe_chunk(chunks[0], language)
        
        # Transcribe multiple chunks
        transcripts = []
        for i, chunk in enumerate(chunks, 1):
            if progress:
                progress.update(progress_task, 
                              description=f"Transcribing chunk {i}/{len(chunks)}...")
            
            transcript = self.transcribe_chunk(chunk, language)
            transcripts.append(transcript)
            
            # Save chunk transcript to temp folder
            if self.session_temp_dir and os.path.exists(self.session_temp_dir):
                chunk_transcript_file = os.path.join(
                    os.path.dirname(chunk), 
                    f"chunk_{i}_transcript.txt"
                )
                try:
                    with open(chunk_transcript_file, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                except Exception:
                    pass  # Silent fail for transcript saving
            
            # No need to clean up individual chunks - they'll be cleaned with session dir
        
        # Save merged transcript to temp folder
        merged_transcript = " ".join(transcripts)
        if self.session_temp_dir and os.path.exists(self.session_temp_dir):
            merged_transcript_file = os.path.join(
                self.session_temp_dir,
                "merged_transcript.txt"
            )
            try:
                with open(merged_transcript_file, 'w', encoding='utf-8') as f:
                    f.write(merged_transcript)
            except Exception:
                pass  # Silent fail for transcript saving
        
        return merged_transcript


class Summarizer:
    def __init__(self, api_key: str = None, provider: str = 'openai'):
        self.provider = provider
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
    
    def summarize(self, text: str, detail_level: str = "medium") -> str:
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
        
        prompt = prompts.get(detail_level, prompts["medium"])
        
        # Handle very long transcripts by chunking if needed
        max_context = 100000  # Conservative limit for context
        if len(text) > max_context:
            # Split into chunks and summarize each
            chunks = [text[i:i+max_context] for i in range(0, len(text), max_context)]
            summaries = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"{prompt}\n\n[Part {i+1} of {len(chunks)}]\n\nTranscript:\n{chunk}"
                if self.provider == 'openai':
                    summary = self._summarize_openai(chunk_prompt)
                else:
                    summary = self._summarize_anthropic(chunk_prompt)
                summaries.append(summary)
            
            # Create final summary from chunk summaries
            final_prompt = f"Combine these partial summaries into one coherent summary:\n\n" + "\n\n---\n\n".join(summaries)
            if self.provider == 'openai':
                return self._summarize_openai(final_prompt)
            else:
                return self._summarize_anthropic(final_prompt)
        else:
            full_prompt = f"{prompt}\n\nTranscript:\n{text}"
            if self.provider == 'openai':
                return self._summarize_openai(full_prompt)
            else:
                return self._summarize_anthropic(full_prompt)
    
    def _summarize_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, structured summaries of video transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content
    
    def _summarize_anthropic(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system="You are a helpful assistant that creates clear, structured summaries of video transcripts.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class QAHandler:
    def __init__(self, api_key: str = None, provider: str = 'openai'):
        self.provider = provider
        self.qa_history = []
        self.sessions_dir = "transcripts"
        
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
    
    def answer_question(self, question: str, transcript: str, video_info: dict, conversation_history: list = None) -> str:
        """Answer a single question based on the transcript with conversation context"""
        system_prompt = f"""You are an AI assistant helping users understand video content. 
You have access to the full transcript of a video titled "{video_info.get('title', 'Unknown')}" 
by {video_info.get('uploader', 'Unknown')} (duration: {video_info.get('duration', 0)}s).

Answer questions based on the transcript content. Be specific and cite relevant parts when appropriate.
If the answer isn't in the transcript, say so clearly.
You can reference previous questions and answers in this conversation to provide contextual responses.

Transcript:
{transcript}"""
        
        if self.provider == 'openai':
            return self._answer_openai(system_prompt, question, conversation_history)
        else:
            return self._answer_anthropic(system_prompt, question, conversation_history)
    
    def _answer_openai(self, system_prompt: str, question: str, conversation_history: list = None) -> str:
        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for qa in conversation_history:
                messages.append({"role": "user", "content": qa["question"]})
                messages.append({"role": "assistant", "content": qa["answer"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content
    
    def _answer_anthropic(self, system_prompt: str, question: str, conversation_history: list = None) -> str:
        # Build messages array with conversation history
        messages = []
        
        # Add conversation history if available
        if conversation_history:
            for qa in conversation_history:
                messages.append({"role": "user", "content": qa["question"]})
                messages.append({"role": "assistant", "content": qa["answer"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    
    def quick_answer(self, question: str, transcript: str, video_info: dict) -> str:
        """Get a quick answer without loading conversation history or starting interactive session"""
        # Use answer_question with no history for a single response
        return self.answer_question(question, transcript, video_info, conversation_history=None)
    
    def get_session_filename(self, video_id: str, title: str = None) -> str:
        """Get the unified cache filename for a video ID (same as transcript cache)"""
        if title:
            sanitized_title = sanitize_filename(title)
            return os.path.join(self.sessions_dir, f"{video_id}_{sanitized_title}.json")
        else:
            # Fallback for videos without title
            return os.path.join(self.sessions_dir, f"{video_id}.json")
    
    def load_session(self, video_id: str, title: str = None) -> bool:
        """Load existing Q&A session from unified cache file"""
        import json
        import glob
        
        # Try to find unified cache file by video ID
        session_file = None
        if title:
            # Try with title first
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
                    return len(self.qa_history) > 0  # Only return True if there's actual QA history
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load session: {e}[/yellow]")
        
        return False
    
    def append_qa_to_history(self, question: str, answer: str, video_id: str, video_info: dict) -> int:
        """Append a single Q&A to the history and save. Returns total QA count."""
        import json
        from datetime import datetime
        import glob
        
        os.makedirs(self.sessions_dir, exist_ok=True)
        title = video_info.get('title', '')
        
        # Find the cache file
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
        
        # Save back
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            return len(existing_qa_history)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save Q&A: {e}[/yellow]")
            return 0
    
    def save_session_json(self, video_id: str, video_info: dict):
        """Save Q&A session by merging with existing unified cache file"""
        import json
        from datetime import datetime
        
        if not self.qa_history:
            return
        
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
        
        # Merge histories: keep all existing QAs that aren't in our current session
        # This handles the case where quick questions were added outside this session
        # We assume self.qa_history contains ALL history if loaded at start
        # So we just use self.qa_history as the full history
        merged_qa_history = self.qa_history
        
        # Merge QA data with existing cache, preserving any existing fields like summaries
        if not isinstance(existing_data, dict):
            existing_data = {}
        merged_data = existing_data.copy()
        merged_data.update({
            'video_info': existing_data.get('video_info', video_info),
            'transcript': existing_data.get('transcript', ''),
            'qa_history': merged_qa_history,
            'last_updated': datetime.now().isoformat()
        })
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            console.print(f"[dim]✓ QA history saved ({len(merged_qa_history)} exchanges)[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save session: {e}[/yellow]")
    
    def interactive_session(self, transcript: str, video_info: dict, video_id: str):
        """Run an interactive Q&A session"""
        
        title = video_info.get('title', '')
        
        # Check for existing session and load automatically
        if self.load_session(video_id, title):
            console.print(f"[green]✓ Loaded existing Q&A session with {len(self.qa_history)} previous exchanges[/green]")
            # Show last Q&A to refresh memory
            if self.qa_history:
                from rich.text import Text
                from rich.console import Group
                
                last_qa = self.qa_history[-1]
                # Truncate answer if too long
                answer_preview = last_qa['answer'][:400] + "..." if len(last_qa['answer']) > 400 else last_qa['answer']
                
                question_text = Text()
                question_text.append("Last Question: ", style="bold cyan")
                question_text.append(last_qa['question'])
                
                prev_content = Group(
                    question_text,
                    Text(),
                    Text("Last Answer: ", style="bold green"),
                    Markdown(answer_preview)
                )
                
                console.print(Panel(
                    prev_content,
                    title="[dim]Previous Context[/dim]",
                    border_style="dim"
                ))
        
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
        
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")
                
                if question.lower() in ['/quit', '/exit', '/q']:
                    # Always save JSON session
                    self.save_session_json(video_id, video_info)
                    console.print("[yellow]Ending Q&A session...[/yellow]")
                    console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id, title)}[/green]")
                    break
                
                elif question.lower() == '/save':
                    self.save_session_json(video_id, video_info)
                    console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id, title)}[/green]")
                    continue
                
                elif question.lower() == '/clear':
                    self.qa_history = []
                    console.print("[yellow]Conversation context cleared[/yellow]")
                    continue
                
                elif question.lower() == '/help':
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
                
                # Display Q&A with proper markdown rendering for the answer
                from rich.text import Text
                from rich.console import Group
                
                # Create question part
                question_text = Text()
                question_text.append("Question: ", style="bold cyan")
                question_text.append(question)
                
                # Create answer part with markdown
                answer_text = Text()
                answer_text.append("Answer: ", style="bold green")
                
                # Combine question and answer with markdown support
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
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Saving and exiting...[/yellow]")
                self.save_session_json(video_id, video_info)
                console.print(f"[green]✓ Session saved to {self.get_session_filename(video_id)}[/green]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")


def browse_cached_videos(api_key: str = None, provider: str = 'openai'):
    """Browse cached videos and select one for Q&A mode"""
    import glob
    import json
    from datetime import datetime
    from rich.table import Table
    from rich.prompt import IntPrompt
    
    # Find all cached files
    cached_files = glob.glob("transcripts/*.json")
    if not cached_files:
        console.print("[yellow]No cached videos found. Process a video first with transcriptor.[/yellow]")
        return
    
    # Load metadata from each file
    videos = []
    for cache_file in cached_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_info = data.get('video_info', {})
            qa_history = data.get('qa_history', [])
            summaries = data.get('summaries', {})
            last_updated = data.get('last_updated', '')
            
            # Count available summaries
            summary_count = 0
            for provider_summaries in summaries.values():
                if isinstance(provider_summaries, dict):
                    summary_count += len(provider_summaries)
            
            # Format duration
            duration_seconds = video_info.get('duration', 0)
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            seconds = duration_seconds % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"
            
            # Parse last updated date
            if last_updated:
                try:
                    updated_dt = datetime.fromisoformat(last_updated)
                    days_ago = (datetime.now() - updated_dt).days
                    if days_ago == 0:
                        updated_str = "Today"
                    elif days_ago == 1:
                        updated_str = "Yesterday"
                    elif days_ago < 7:
                        updated_str = f"{days_ago} days ago"
                    else:
                        updated_str = updated_dt.strftime("%b %d")
                except:
                    updated_str = "Unknown"
            else:
                updated_str = "Unknown"
            
            videos.append({
                'file': cache_file,
                'video_id': cache_file.split('/')[-1].split('_')[0],
                'title': video_info.get('title', 'Unknown'),
                'duration': duration_seconds,
                'duration_str': duration_str,
                'qa_count': len(qa_history),
                'summary_count': summary_count,
                'updated': updated_str,
                'transcript': data.get('transcript', ''),
                'video_info': video_info
            })
        except Exception as e:
            console.print(f"[dim]Warning: Could not load {cache_file}: {e}[/dim]")
            continue
    
    if not videos:
        console.print("[yellow]No valid cached videos found.[/yellow]")
        return
    
    # Sort by last updated (most recent first)
    videos.sort(key=lambda x: x['updated'], reverse=False)
    
    # Create and display table
    table = Table(title="[bold green]Cached Videos[/bold green]", show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Title", style="white", max_width=40)
    table.add_column("Duration", style="dim", width=10, justify="right")
    table.add_column("Q&A", style="green", width=6, justify="right")
    table.add_column("Summaries", style="yellow", width=10, justify="right")
    table.add_column("Updated", style="dim", width=12)
    
    for i, video in enumerate(videos, 1):
        # Truncate title if too long
        title = video['title']
        if len(title) > 40:
            title = title[:37] + "..."
        
        # Color code Q&A count
        qa_style = "green" if video['qa_count'] > 0 else "dim"
        qa_text = str(video['qa_count']) if video['qa_count'] > 0 else "0"
        
        # Format summary count
        summary_text = str(video['summary_count']) if video['summary_count'] > 0 else "-"
        
        table.add_row(
            str(i),
            title,
            video['duration_str'],
            f"[{qa_style}]{qa_text}[/{qa_style}]",
            summary_text,
            video['updated']
        )
    
    console.print(table)
    
    # Show statistics
    total_qa = sum(v['qa_count'] for v in videos)
    total_duration = sum(v['duration'] for v in videos)
    total_hours = total_duration // 3600
    total_minutes = (total_duration % 3600) // 60
    
    console.print(f"\n[dim]Total: {len(videos)} video(s) | {total_qa} Q&A exchanges | {total_hours}h {total_minutes}m total duration[/dim]")
    
    # Interactive selection
    console.print("\n[bold cyan]Select a video for Q&A mode[/bold cyan]")
    try:
        choice = IntPrompt.ask(
            "Enter video number (0 to exit)",
            choices=[str(i) for i in range(len(videos) + 1)],
            default=0,
            show_choices=False
        )
        
        if choice == 0:
            console.print("[yellow]Exiting...[/yellow]")
            return
        
        # Get selected video
        selected = videos[choice - 1]
        console.print(f"\n[green]✓ Loading \"{selected['title']}\"...[/green]")
        
        # Start Q&A session with cached data
        qa_handler = QAHandler(api_key=api_key, provider=provider)
        qa_handler.interactive_session(
            transcript=selected['transcript'],
            video_info=selected['video_info'],
            video_id=selected['video_id']
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Selection cancelled.[/yellow]")
        return
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return


@click.command()
@click.argument('video_url', required=False)  # Make video_url optional for --show-history
@click.argument('question', required=False)  # Optional second argument for shorthand
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
@click.option('--show-history', is_flag=True, help='Show all Q&A pairs from history')
@click.option('--browse', is_flag=True, help='Browse cached videos and select one for Q&A')
def main(video_url, question, api_key, language, detail, output, keep_audio, transcript_only, provider, show_temp_dir, qa, ask, display_format, show_history, browse):
    # Handle --show-history without video URL (list all available histories)
    if show_history and not video_url:
        import glob
        import json
        
        console.print(Panel.fit(
            "[bold green]Available Q&A Histories[/bold green]",
            border_style="green"
        ))
        
        cached_files = glob.glob("transcripts/*.json")
        if not cached_files:
            console.print("[yellow]No Q&A histories found[/yellow]")
            sys.exit(0)
        
        videos_with_qa = []
        for cache_file in cached_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                qa_history = data.get('qa_history', [])
                if qa_history:
                    video_info = data.get('video_info', {})
                    videos_with_qa.append({
                        'title': video_info.get('title', 'Unknown'),
                        'url': f"https://www.youtube.com/watch?v={cache_file.split('/')[-1].split('_')[0]}",
                        'qa_count': len(qa_history),
                        'file': cache_file
                    })
            except Exception:
                continue
        
        if not videos_with_qa:
            console.print("[yellow]No videos with Q&A history found[/yellow]")
            sys.exit(0)
        
        for video in videos_with_qa:
            console.print(f"\n[bold cyan]{video['title']}[/bold cyan]")
            console.print(f"  URL: [dim]{video['url']}[/dim]")
            console.print(f"  Q&A exchanges: [green]{video['qa_count']}[/green]")
        
        console.print(f"\n[dim]Total: {len(videos_with_qa)} video(s) with Q&A history[/dim]")
        console.print("\n[yellow]Tip: Use the URL with --show-history to view specific Q&A history[/yellow]")
        sys.exit(0)
    
    # Handle --browse mode
    if browse:
        browse_cached_videos(api_key, provider)
        sys.exit(0)
    
    # Require video_url for all other operations
    if not video_url:
        console.print("[red]Error: VIDEO_URL is required (except when using --show-history or --browse alone)[/red]")
        sys.exit(1)
    
    # Detect shorthand syntax: if question is provided without --ask flag, treat it as --ask
    if question and not ask and not qa and not transcript_only:
        ask = question
        question = None  # Clear to avoid confusion
    
    # Validate options
    if qa and transcript_only:
        console.print("[red]Error: Cannot use --qa and --transcript-only together[/red]")
        sys.exit(1)
    
    if ask and (qa or transcript_only):
        console.print("[red]Error: Cannot use --ask with --qa or --transcript-only[/red]")
        sys.exit(1)
    
    if show_history and (qa or transcript_only or ask):
        console.print("[red]Error: Cannot use --show-history with other modes[/red]")
        sys.exit(1)
    
    if browse and (qa or transcript_only or ask or show_history):
        console.print("[red]Error: Cannot use --browse with other modes[/red]")
        sys.exit(1)
    
    audio_file = None
    extractor = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Extract video ID
            task = progress.add_task("Extracting video ID...", total=None)
            video_id = YouTubeAudioExtractor.extract_video_id(video_url)
            
            if not video_id:
                console.print("[red]Error: Could not extract video ID from URL[/red]")
                sys.exit(1)
            
            # Handle --show-history mode early
            if show_history:
                progress.update(task, description="Loading Q&A history...")
                import glob
                import json
                from rich.text import Text
                from rich.console import Group
                
                # Find cached file for this video
                cached_files = glob.glob(f"transcripts/{video_id}_*.json")
                if not cached_files:
                    console.print(f"[yellow]No cached data found for video ID: {video_id}[/yellow]")
                    sys.exit(0)
                
                cache_file = cached_files[0]
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    qa_history = data.get('qa_history', [])
                    video_info = data.get('video_info', {})
                    
                    if not qa_history:
                        console.print(f"[yellow]No Q&A history found for: {video_info.get('title', 'Unknown')}[/yellow]")
                        sys.exit(0)
                    
                    # Display all Q&A pairs
                    console.print(Panel.fit(
                        f"[bold green]Q&A History[/bold green]\n"
                        f"Video: {video_info.get('title', 'Unknown')}\n"
                        f"Total exchanges: {len(qa_history)}",
                        border_style="green"
                    ))
                    
                    for i, qa in enumerate(qa_history, 1):
                        question_text = Text()
                        question_text.append(f"Q{i}: ", style="bold cyan")
                        question_text.append(qa['question'])
                        
                        qa_content = Group(
                            question_text,
                            Text(),  # Empty line
                            Text(f"A{i}: ", style="bold green"),
                            Markdown(qa['answer'])
                        )
                        
                        console.print(Panel(
                            qa_content,
                            title=f"[dim]Exchange {i}/{len(qa_history)}[/dim]",
                            border_style="dim"
                        ))
                        
                        if i < len(qa_history):
                            console.print()  # Add spacing between Q&A pairs
                    
                    progress.stop()
                    sys.exit(0)
                    
                except Exception as e:
                    console.print(f"[red]Error loading Q&A history: {e}[/red]")
                    sys.exit(1)
            
            # Check for cached transcript using video ID only (no title extraction needed)
            progress.update(task, description="Checking for cached transcript...")
            
            # Look for cached JSON file starting with the video ID
            import glob
            import json
            cached_files = glob.glob(f"transcripts/{video_id}_*.json")
            cache_file = cached_files[0] if cached_files else None
            
            video_info = None
            transcript = None
            summaries_cache = {}
            if cache_file:
                progress.update(task, description="Loading cached data...")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    transcript = cached_data.get('transcript', '')
                    video_info = cached_data.get('video_info', {})
                    summaries_cache = cached_data.get('summaries', {})
                    
                    console.print(f"[green]✓ Using cached transcript and video info from {cache_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load cached data: {e}[/yellow]")
                    cache_file = None
                
                audio_file = None
            if not cache_file:
                # Create extractor only when we need to download
                extractor = YouTubeAudioExtractor()
                
                if show_temp_dir:
                    console.print(f"[cyan]Temp directory: {extractor.session_temp_dir}[/cyan]")
                
                # Download audio
                progress.update(task, description="Downloading audio from YouTube...")
                audio_file, video_info = extractor.download_audio(video_url)
                
                file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                console.print(f"[green]✓ Downloaded: {video_info['title']}[/green]")
                console.print(f"[dim]Duration: {video_info['duration']}s | Size: {file_size_mb:.1f}MB | Uploader: {video_info['uploader']}[/dim]")
                
                # Transcribe audio (with chunking if needed)
                progress.update(task, description="Initializing transcription...")
                # Pass the session temp dir if we have an extractor
                session_temp = extractor.session_temp_dir if extractor else None
                transcriber = ChunkedWhisperTranscriber(api_key=api_key, 
                                                       session_temp_dir=session_temp)
                transcript = transcriber.transcribe(audio_file, language=language, 
                                                   progress_task=task, progress=progress)
            
            console.print(f"[green]✓ Transcription complete ({len(transcript)} characters)[/green]")
            new_summary_value = None
            
            if ask:
                # Quick answer mode - saves to history like interactive sessions
                progress.update(task, description="Getting answer...")
                qa_handler = QAHandler(api_key=api_key, provider=provider)
                
                # Load existing history to show context
                qa_handler.load_session(video_id, video_info.get('title'))
                
                # Get the answer
                answer = qa_handler.quick_answer(ask, transcript, video_info)
                
                # Save Q&A to history
                total_qa_count = qa_handler.append_qa_to_history(ask, answer, video_id, video_info)
                
                # Display Q&A with proper markdown rendering
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
                
                # Show that it was saved to history
                console.print(f"[green]✓ Added to Q&A history ({total_qa_count} total exchanges)[/green]")
                
                # Optionally save to file if output specified
                if output:
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(f"# Quick Q&A: {video_info['title']}\n\n")
                        f.write(f"**Question:** {ask}\n\n")
                        f.write(f"**Answer:** {answer}\n")
                    console.print(f"[green]✓ Answer also saved to {output}[/green]")
                
                result = None
            elif qa:
                # Interactive Q&A mode - close progress first to avoid display conflicts
                progress.update(task, description="Starting Q&A session...", completed=True)
                progress.stop()  # Stop the progress display
                
                qa_handler = QAHandler(api_key=api_key, provider=provider)
                # Pass video_id for session persistence
                qa_handler.interactive_session(transcript, video_info, video_id)
                result = None  # No result to save in regular output flow
            elif transcript_only:
                result = transcript
                if len(transcript) < 5000:  # Only show in panel if not too long
                    console.print(Panel(transcript, title="Transcript", border_style="green"))
                else:
                    console.print("[yellow]Transcript is too long to display. Saving to file...[/yellow]")
            else:
                # Summarize transcript (use cache if available)
                cached_summary = None
                if isinstance(summaries_cache, dict):
                    provider_map = summaries_cache.get(provider, {})
                    if isinstance(provider_map, dict):
                        cached_summary = provider_map.get(detail)
                if cached_summary:
                    result = cached_summary
                    console.print(f"[green]✓ Using cached summary ({provider}:{detail})[/green]")
                    # Use selected display format
                    if display_format == 'cards' and print_summary_cards:
                        print_summary_cards(cached_summary, video_info)
                    elif display_format == 'enhanced' and print_summary_enhanced:
                        print_summary_enhanced(cached_summary, "Summary", video_info)
                    else:
                        console.print(Panel(cached_summary, title="Summary", border_style="green"))
                else:
                    progress.update(task, description="Generating summary...")
                    summarizer = Summarizer(api_key=api_key, provider=provider)
                    summary = summarizer.summarize(transcript, detail_level=detail)
                    result = summary
                    new_summary_value = summary
                    # Use selected display format
                    if display_format == 'cards' and print_summary_cards:
                        print_summary_cards(summary, video_info)
                    elif display_format == 'enhanced' and print_summary_enhanced:
                        print_summary_enhanced(summary, "Summary", video_info)
                    else:
                        console.print(Panel(summary, title="Summary", border_style="green"))
            
            # Always save the transcript and video info for reuse (if we just transcribed it)
            if not cache_file:
                os.makedirs("transcripts", exist_ok=True)
                # Generate filename with title (new format only)
                sanitized_title = sanitize_filename(video_info['title'])
                cache_file = f"transcripts/{video_id}_{sanitized_title}.json"
                
                # Check if file already exists (e.g., from QA session) and merge
                existing_data = {}
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        pass  # If we can't read it, we'll overwrite
                
                # Merge new transcript data with existing QA data
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
                cached_data = {k: v for k, v in cached_data.items() if v is not None and (k != 'qa_history' or v)}
                
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cached_data, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Transcript and video info cached in {cache_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not save cached data: {e}[/yellow]")

            # If we used an existing cache file and generated a new summary, persist it
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
            
            # Save output if requested (skip for Q&A mode as it handles its own saving)
            if not qa and (output or (transcript_only and len(transcript) > 5000)):
                if output:
                    output_file = output
                else:
                    sanitized_title = sanitize_filename(video_info.get('title', 'untitled'))
                    file_type = 'transcript' if transcript_only else 'summary'
                    output_file = f"{video_id}_{sanitized_title}_{file_type}.md"
                progress.update(task, description="Saving to file...")
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
            
            if not qa:  # Only show "Done!" for non-interactive modes
                progress.update(task, description="Done!", completed=True)
    
    except ValueError as e:
        console.print(f"[red]Configuration Error: {str(e)}[/red]")
        console.print("[yellow]Tip: Set OPENAI_API_KEY in your .env file or use --api-key option[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up the entire session temp directory
        if extractor:
            if show_temp_dir:
                console.print(f"[yellow]Temp directory preserved: {extractor.session_temp_dir}[/yellow]")
            elif keep_audio:
                console.print(f"[yellow]Audio file kept in: {extractor.session_temp_dir}[/yellow]")
            else:
                extractor.cleanup()
                console.print(f"[dim]Cleaned up all temporary files[/dim]")


if __name__ == '__main__':
    main()