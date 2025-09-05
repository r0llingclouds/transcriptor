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
from dotenv import load_dotenv

load_dotenv()

console = Console()

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
                    'model': 'whisper-1',
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
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, structured summaries of video transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _summarize_anthropic(self, prompt: str) -> str:
        response = self.client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=2000,
            temperature=0.3,
            system="You are a helpful assistant that creates clear, structured summaries of video transcripts.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


@click.command()
@click.argument('video_url')
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
def main(video_url, api_key, language, detail, output, keep_audio, transcript_only, provider, show_temp_dir):
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
            
            # Check for cached transcript
            transcript_file = f"transcripts/{video_id}_transcript.txt"
            if os.path.exists(transcript_file):
                progress.update(task, description="Loading cached transcript...")
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = f.read()
                console.print(f"[green]✓ Using cached transcript from {transcript_file}[/green]")
                
                # Get video info without downloading
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    video_info = ydl.extract_info(video_url, download=False)
                    video_info = {
                        'title': video_info.get('title', 'Unknown'),
                        'duration': video_info.get('duration', 0),
                        'uploader': video_info.get('uploader', 'Unknown'),
                    }
                audio_file = None
            else:
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
            
            if transcript_only:
                result = transcript
                if len(transcript) < 5000:  # Only show in panel if not too long
                    console.print(Panel(transcript, title="Transcript", border_style="green"))
                else:
                    console.print("[yellow]Transcript is too long to display. Saving to file...[/yellow]")
            else:
                # Summarize transcript
                progress.update(task, description="Generating summary...")
                summarizer = Summarizer(api_key=api_key, provider=provider)
                summary = summarizer.summarize(transcript, detail_level=detail)
                result = summary
                console.print(Panel(summary, title="Summary", border_style="green"))
            
            # Always save the transcript for reuse (if we just transcribed it)
            if not os.path.exists(transcript_file):
                os.makedirs("transcripts", exist_ok=True)
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                console.print(f"[green]✓ Transcript cached in {transcript_file}[/green]")
            
            # Save output if requested
            if output or (transcript_only and len(transcript) > 5000):
                output_file = output or f"{video_info['title'][:50]}_{'transcript' if transcript_only else 'summary'}.md"
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