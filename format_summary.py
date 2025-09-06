#!/usr/bin/env python3
"""Enhanced summary formatting for transcriptor"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich import box

def format_summary_enhanced(summary: str) -> str:
    """Format summary text with improved markdown structure and visual hierarchy."""
    lines = summary.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append('')
                in_list = False
            continue
        
        # Main Topic section - left-aligned with markdown headers
        if '**Main Topic:**' in line or 'Main topic' in line.lower():
            content = line.replace('**Main Topic:**', '').replace('**', '').strip()
            formatted_lines.append("### 📌 Main Topic\n")
            formatted_lines.append(f"{content}\n")
        
        # Key Points section - left-aligned with markdown headers
        elif '**Key Points:**' in line or 'Key points' in line.lower():
            formatted_lines.append("\n### 🔑 Key Points\n")
            in_list = True
        
        # Important Details section - left-aligned with markdown headers
        elif '**Important Details' in line or '**Examples' in line or 'Important details' in line.lower():
            formatted_lines.append("\n### 💡 Important Details & Examples\n")
            in_list = True
        
        # Conclusion section - left-aligned with markdown headers
        elif '**Conclusion' in line or '**Main Takeaway' in line or 'Conclusion' in line.lower():
            formatted_lines.append("\n### 🎯 Conclusion\n")
            in_list = False
        
        # Handle bullet points
        elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
            content = line.lstrip('-•* ').strip()
            # Use better bullet formatting
            formatted_lines.append(f"• **{content.split('.')[0]}**")
            if '.' in content:
                rest = '.'.join(content.split('.')[1:]).strip()
                if rest:
                    formatted_lines.append(f"  {rest}")
            formatted_lines.append('')
            in_list = True
        
        # Handle numbered items
        elif line[0].isdigit() and (line[1] == '.' or (len(line) > 1 and line[1].isdigit() and line[2] == '.')):
            formatted_lines.append(line)
            formatted_lines.append('')
            in_list = True
        
        # Regular paragraphs
        else:
            if line.startswith('**') and line.endswith('**'):
                # Section headers
                content = line.replace('**', '').strip()
                formatted_lines.append(f"### {content}\n")
            else:
                formatted_lines.append(line)
                if not in_list:
                    formatted_lines.append('')
    
    # Clean up formatting
    result = '\n'.join(formatted_lines)
    # Remove excessive blank lines
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    return result.strip()


def print_summary_enhanced(summary: str, title: str = "Summary", video_info: dict = None):
    """Print a beautifully formatted summary with enhanced visual display."""
    console = Console()
    
    # Parse the summary into sections
    sections = create_summary_sections(summary)
    
    # Create header with video info if available
    if video_info:
        console.print()
        console.print(f"[bold bright_cyan]📹 {video_info.get('title', 'Unknown')}[/bold bright_cyan]")
        console.print(f"[dim]👤 {video_info.get('uploader', 'Unknown')} | ⏱️ {video_info.get('duration', 0)}s[/dim]")
        console.print()
    
    # Create content with TRUE left alignment - no markdown, just formatted text
    content_lines = []
    
    # Main Topic
    if sections['main_topic']:
        content_lines.append("[bold bright_cyan]📌 Main Topic[/bold bright_cyan]")
        content_lines.append("")
        content_lines.append(sections['main_topic'])
        content_lines.append("")
    
    # Key Points
    if sections['key_points']:
        content_lines.append("[bold bright_yellow]🔑 Key Points[/bold bright_yellow]")
        content_lines.append("")
        for point in sections['key_points']:
            content_lines.append(f"• {point}")
            content_lines.append("")
    
    # Important Details
    if sections['details']:
        content_lines.append("[bold bright_magenta]💡 Important Details & Examples[/bold bright_magenta]")
        content_lines.append("")
        for detail in sections['details']:
            content_lines.append(f"• {detail}")
            content_lines.append("")
    
    # Conclusion
    if sections['conclusion']:
        content_lines.append("[bold bright_green]🎯 Conclusion[/bold bright_green]")
        content_lines.append("")
        content_lines.append(sections['conclusion'])
    
    # Join all content
    formatted_content = "\n".join(content_lines)
    
    # Create main panel with plain text (not markdown) for true left alignment
    panel = Panel(
        formatted_content,
        title=f"[bold bright_cyan]{title}[/bold bright_cyan]",
        border_style="bright_cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True
    )
    
    console.print(panel)
    console.print()


def create_summary_sections(summary: str) -> dict:
    """Parse summary into structured sections for advanced display."""
    sections = {
        'main_topic': '',
        'key_points': [],
        'details': [],
        'conclusion': ''
    }
    
    lines = summary.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if 'Main Topic' in line or 'main topic' in line.lower():
            current_section = 'main_topic'
        elif 'Key Points' in line or 'key points' in line.lower():
            current_section = 'key_points'
        elif 'Important Details' in line or 'Examples' in line:
            current_section = 'details'
        elif 'Conclusion' in line or 'Main Takeaway' in line:
            current_section = 'conclusion'
        elif line.startswith('-') or line.startswith('•'):
            content = line.lstrip('-• ').strip()
            if current_section == 'key_points':
                sections['key_points'].append(content)
            elif current_section == 'details':
                sections['details'].append(content)
        elif current_section == 'main_topic' and not line.startswith('**'):
            sections['main_topic'] += line + ' '
        elif current_section == 'conclusion' and not line.startswith('**'):
            sections['conclusion'] += line + ' '
    
    # Clean up
    sections['main_topic'] = sections['main_topic'].strip()
    sections['conclusion'] = sections['conclusion'].strip()
    
    return sections


def print_summary_cards(summary: str, video_info: dict = None):
    """Display summary as interactive cards/sections."""
    console = Console()
    sections = create_summary_sections(summary)
    
    # Header
    if video_info:
        console.print()
        console.print(f"[bold bright_cyan]📹 {video_info.get('title', 'Unknown')}[/bold bright_cyan]")
        console.print(f"[dim]👤 {video_info.get('uploader')} | ⏱️ {video_info.get('duration')}s[/dim]")
        console.print()
    
    # Main Topic Card - left-aligned
    if sections['main_topic']:
        topic_panel = Panel(
            sections['main_topic'],
            title="[bold]📌 Main Topic[/bold]",
            title_align="left",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        console.print(topic_panel)
    
    # Key Points Card - left-aligned
    if sections['key_points']:
        points_text = '\n'.join([f"• {point}" for point in sections['key_points']])
        points_panel = Panel(
            points_text,
            title="[bold]🔑 Key Points[/bold]",
            title_align="left",
            border_style="bright_green",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        console.print(points_panel)
    
    # Details Card - left-aligned
    if sections['details']:
        details_text = '\n'.join([f"• {detail}" for detail in sections['details']])
        details_panel = Panel(
            details_text,
            title="[bold]💡 Important Details & Examples[/bold]",
            title_align="left",
            border_style="bright_yellow",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        console.print(details_panel)
    
    # Conclusion Card - left-aligned
    if sections['conclusion']:
        conclusion_panel = Panel(
            sections['conclusion'],
            title="[bold]🎯 Conclusion[/bold]",
            title_align="left",
            border_style="bright_magenta",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        console.print(conclusion_panel)
    
    console.print()