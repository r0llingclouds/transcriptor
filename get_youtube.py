import datetime
import os
import re
import argparse
from pytube import YouTube
from pydub import AudioSegment
from openai import OpenAI
from pydub.utils import make_chunks
import tiktoken

def download_youtube_video(url, output_path="/Users/tirsolopezausens/Downloads"):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()

    video_name = f"""{stream.title}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.mp4"""

    _ = stream.download(output_path=output_path, filename=video_name)
    
    return video_name

def parse_time(time_str):
    """
    Parses a time string into milliseconds.

    :param time_str: Time string (e.g., "2h53m34s", "1h34m", "23m66s", "34s").
    :return: Time in milliseconds.
    """
    time_pattern = re.compile(r'(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?')
    match = time_pattern.match(time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    time_parts = match.groupdict()
    hours = int(time_parts['hours']) if time_parts['hours'] else 0
    minutes = int(time_parts['minutes']) if time_parts['minutes'] else 0
    seconds = int(time_parts['seconds']) if time_parts['seconds'] else 0

    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
    return total_ms

def extract_audio_segment(filename, path, start_time_str, end_time_str):
    """
    Extracts a segment from an MP3 file.

    :param mp3_file_path: Path to the input MP3 file.
    :param start_time_str: Start time of the segment (e.g., "2h53m34s", "1h34m", "23m66s", "34s").
    :param end_time_str: End time of the segment (e.g., "2h53m34s", "1h34m", "23m66s", "34s").
    :param output_file_path: Path to save the extracted MP3 segment.
    """
    # Convert start and end times from string to milliseconds
    start_time_ms = parse_time(start_time_str)
    end_time_ms = parse_time(end_time_str)

    # Load the audio file
    mp3_file_path = os.path.join(path, filename)
    audio = AudioSegment.from_file(mp3_file_path)

    # Extract the segment
    extracted_segment = audio[start_time_ms:end_time_ms]

    # Export the extracted segment
    output_filename = f"{filename[:-4]}_seg_{start_time_str}_{end_time_str}.mp3"
    output_full_path = os.path.join(path, output_filename)
    extracted_segment.export(output_full_path, format="mp3")

    return output_filename

def split_audio(file, path):
    myaudio = AudioSegment.from_file(os.path.join(path, file)) 
    chunk_length_ms = 20 * 60 * 1000 # pydub calculates in millisec, 15 min
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks
    chunk_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{file[:-4]}_chunk{i}.mp3" 
        chunk_names.append(chunk_name)
        chunk_path = os.path.join(path, chunk_name)
        chunk.export(chunk_path, format="mp3")
    return chunk_names

def get_chunk_transcription(file_path):
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcript.text

def get_transcription(file, path, max_filesize_mb=24):
    filesize_mb = os.path.getsize(os.path.join(path, file)) / (1024 * 1024)
    if filesize_mb < max_filesize_mb:
        transcript = get_chunk_transcription(os.path.join(path, file))
    else:
        print("* splitting audio")
        chunk_names = split_audio(file, path)
        transcripts = []
        for i, chunk in enumerate(chunk_names):
            print(f"* transcribing chunk {i}")
            transcripts.append(get_chunk_transcription(os.path.join(path, chunk)))
        transcript = " ".join(transcripts)
    return transcript

def load_and_trim_text(text, max_tokens = 27000):
    
    # Initialize the tokenizer
    encoder = tiktoken.encoding_for_model("gpt-4o")
    
    # Tokenize the text
    tokens = encoder.encode(text)
    length = len(tokens)
    
    # Check if the number of tokens exceeds the max limit
    if length > max_tokens:
        # Trim the tokens to the max limit
        tokens = tokens[:max_tokens]
        # Decode the tokens back to text
        trimmed_text = encoder.decode(tokens)
        print(f"* tokens {length} trimmed to {max_tokens}")
    else:
        print(f"* tokens {length}")
        trimmed_text = text
    
    return trimmed_text

def summarize(transcription):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I'm going to give you a text. Summarize it, extracting the most important information. Aim to retain the most important points. Return a summary of the text."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def summarize_keypoints(transcription):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. Read the following text and extract the key points of it. Aim to retain the most important points and return these key points in a list. You can give a brief abstract before the list."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def qna(context, question):

    # Define the messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the following question based on the given context."},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {question}"}
    ]

    # Call the OpenAI API with the GPT-4 model
    response = client.chat.completions.create(
        model="gpt-4o",  # Specify the GPT-4 model
        messages=messages,
        max_tokens=300,  # Adjust as needed for the expected length of the answer
        temperature=0.3,  # Adjust the creativity level of the model
        n=1  # Number of responses to generate
    )

    # Extract the answer from the API response
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos or segments")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start_time", help="start time (e.g., 2h34m45s)")
    parser.add_argument("end_time", help="end time (e.g., 2h34m45s)")
    parser.add_argument("question", nargs='?', help="question to answer")

    args = parser.parse_args()

    path = "/Users/tirsolopezausens/Downloads"
    print(f"* downloading")
    url = args.url.split("&")[0]
    video_name = download_youtube_video(url, path)

    print(f"* extracting segment")
    segment = extract_audio_segment(video_name, path, args.start_time, args.end_time)

    print(f"* transcribing")
    transcript = get_transcription(segment, path)
    with open(os.path.join(path, f"{segment[:-4]}.txt"), "w") as f:
        f.write(transcript)

    trimmed_transcript = load_and_trim_text(transcript, 26000)

    # operations
    if args.question:
        print("* question answering")
        response = qna(trimmed_transcript, args.question)
        print(args.question)
        print()
        print(response)
        with open(f"{segment[:-4]}_qna_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt", "w") as f:
            f.write(args.question + "\n\n")
            f.write(response)
    else:
        print("* summarizing")
        response = summarize(trimmed_transcript)
        with open(f"{segment[:-4]}_summary.txt", "w") as f:
            f.write(response)
        print("* key points")
        response = summarize_keypoints(trimmed_transcript)
        print(response)
        with open(f"{segment[:-4]}_keypoints.txt", "w") as f:
            f.write(response)

    # clean up
    os.remove(os.path.join(path, video_name))
    os.remove(os.path.join(path, segment))

if __name__ == "__main__":
    client = OpenAI()    
    main()