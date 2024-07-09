import os
import re
import datetime
import argparse
from pytube import YouTube
from pydub import AudioSegment
from openai import OpenAI
from pydub.utils import make_chunks
import tiktoken
from dotenv import load_dotenv

# Initialize OpenAI client
client = OpenAI()

# Function to download YouTube video audio
def download_youtube_video(url, output_path=os.getenv("OUTPUT_PATH", "~/Downloads")):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    video_name = f"{stream.title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    stream.download(output_path=output_path, filename=video_name)
    return video_name

# Function to parse time strings into milliseconds
def parse_time(time_str):
    time_pattern = re.compile(r'(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?')
    match = time_pattern.match(time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    time_parts = match.groupdict()
    hours = int(time_parts['hours']) if time_parts['hours'] else 0
    minutes = int(time_parts['minutes']) if time_parts['minutes'] else 0
    seconds = int(time_parts['seconds']) if time_parts['seconds'] else 0
    return (hours * 3600 + minutes * 60 + seconds) * 1000

# Function to extract an audio segment
def extract_audio_segment(filename, path, start_time_str, end_time_str):
    start_time_ms = parse_time(start_time_str)
    end_time_ms = parse_time(end_time_str)
    audio = AudioSegment.from_file(os.path.join(path, filename))
    extracted_segment = audio[start_time_ms:end_time_ms]
    output_filename = f"{filename[:-4]}_seg_{start_time_str}_{end_time_str}.mp3"
    output_full_path = os.path.join(path, output_filename)
    extracted_segment.export(output_full_path, format="mp3")
    return output_filename

# Function to split audio into chunks
def split_audio(file, path):
    myaudio = AudioSegment.from_file(os.path.join(path, file)) 
    chunk_length_ms = 20 * 60 * 1000  # 20 minutes in milliseconds
    chunks = make_chunks(myaudio, chunk_length_ms)
    chunk_names = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{file[:-4]}_chunk{i}.mp3" 
        chunk_names.append(chunk_name)
        chunk.export(os.path.join(path, chunk_name), format="mp3")
    return chunk_names

# Function to transcribe audio chunk
def get_chunk_transcription(file_path):
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcript.text

# Function to transcribe audio with optional splitting
def get_transcription(file, path, max_filesize_mb=24):
    filesize_mb = os.path.getsize(os.path.join(path, file)) / (1024 * 1024)
    if filesize_mb < max_filesize_mb:
        return get_chunk_transcription(os.path.join(path, file))
    else:
        chunk_names = split_audio(file, path)
        transcripts = [get_chunk_transcription(os.path.join(path, chunk)) for chunk in chunk_names]
        # remove chunk files
        for chunk in chunk_names:
            os.remove(os.path.join(path, chunk))
        return " ".join(transcripts)

# Function to load and trim text based on token limit
def load_and_trim_text(text, max_tokens=27000):
    encoder = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoder.decode(tokens)
    return text

# Function to summarize transcription
def summarize(transcription, language="en"):
    system_content = {
        "en": "You are a highly skilled AI trained in language comprehension and summarization. I'm going to give you a text. Summarize it, extracting the most important information. Aim to retain the most important points. Return a summary of the text.",
        "es": "Eres una IA altamente capacitada en comprensión y resumen de textos. Voy a darte un texto. Resúmelo, extrayendo la información más importante. Intenta retener los puntos más importantes. Devuelve un resumen del texto."
    }
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_content[language]},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message.content

# Function to summarize key points
def summarize_keypoints(transcription, language="en"):
    system_content = {
        "en": "You are a highly skilled AI trained in language comprehension and summarization. Read the following text and extract the key points of it. Aim to retain the most important points and return these key points in a list. You can give a brief abstract before the list.",
        "es": "Eres una IA altamente capacitada en comprensión y resumen de textos. Lee el siguiente texto y extrae los puntos clave del mismo. Intenta retener los puntos más importantes y devuelve estos puntos clave en una lista. Puedes dar un breve resumen antes de la lista."
    }
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_content[language]},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message.content

# Function for Q&A
def qna(context, question, language="en"):
    system_content = {
        "en": "You are a helpful assistant. Answer the following question based on the given context.",
        "es": "Eres un asistente útil. Responde la siguiente pregunta basada en el contexto dado."
    }
    messages = [
        {"role": "system", "content": system_content[language]},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {question}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        n=1
    )
    return response.choices[0].message.content

# Function to read text aloud
def read_aloud(filename, language="en"):
    with open(filename, "r") as f:
        text = f.read()
    output_file = filename[:-4] + "_speech.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="nova",
        input=text,
    ) as response:
        response.stream_to_file(output_file)
    os.system(f'ffplay -nodisp -autoexit "{output_file}"')

# Function to extract arguments from input text
def extract_arguments(text):
    pattern = r'(https://www\.youtube\.com/watch\?v=[^&\s]+)(&[^&\s]*)*'
    clean_text = re.sub(pattern, r'\1', text)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant. Extract the mode, YouTube link, start time, end time, and question from the following text: \"{clean_text}\".\n\nFormat the output as a JSON object with the keys 'mode', 'youtube_link', 'start_time', 'end_time', and 'question'. Mode will always come first as one ore more characters, in case it is present. If a key is not present, set its value to None. For the questions, if present, return them as a list of strings. If there are no questions, return an empty list."
            }
        ],
        response_format={ "type": "json_object" }
    )
    response_text = response.choices[0].message.content.strip().replace('null', 'None')
    return eval(response_text)

# Main function
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos or segments")
    parser.add_argument("data", help="mode link start end questions")
    args = parser.parse_args()
    args_json = extract_arguments(args.data)
    path = os.getenv("OUTPUT_PATH", "~/Downloads")

    args_json["mode"] = 'kp' if args_json["mode"] is None else args_json["mode"] # default mode is key points
    args_json["mode"] = 'kpa' if args_json["mode"] == 'a' else args_json["mode"] # shortcut for key points with audio
    args_json["mode"] = 'kpe' if args_json["mode"] == 'e' else args_json["mode"] # shortcut for key points in Spanish
    args_json["mode"] = 'kpea' if args_json["mode"] == 'ea' else args_json["mode"] # shortcut for key points in Spanish with audio
    audio = 'a' in args_json["mode"]
    args_json["mode"] = args_json["mode"].replace('a', '')

    print(f"* downloading")
    url = args_json["youtube_link"]
    video_name = download_youtube_video(url, path)

    print(f"* extracting segment")
    args_json["start_time"] = '0s' if args_json["start_time"] is None else args_json["start_time"]
    if args_json["end_time"] is None:
        audio_segment = AudioSegment.from_file(os.path.join(path, video_name))
        video_length = len(audio_segment) / 1000
        args_json["end_time"] = f"{int(video_length // 3600)}h{int((video_length % 3600) // 60)}m{int(video_length % 60)}s"

    print(f"* data {args_json}")
    print(f"* audio {audio}")
    segment = extract_audio_segment(video_name, path, args_json["start_time"], args_json["end_time"])

    print(f"* transcribing")
    transcript = get_transcription(segment, path)
    with open(os.path.join(path, f"{segment[:-4]}.txt"), "w") as f:
        f.write(transcript)

    trimmed_transcript = load_and_trim_text(transcript, 26000)

    # operations based on mode
    qna_dict = {}
    if args_json["question"]:
        for question in args_json["question"]:
            print('--------------------------------------')
            print("* question answering")
            response = qna(trimmed_transcript, question, language="es" if 'e' in args_json["mode"] else "en")
            qna_dict[question] = response
            print(question, response, sep="\n\n")
        filename_out = os.path.join(path, f"{segment[:-4]}_qna_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
        with open(filename_out, "w") as f:
            for question, response in qna_dict.items():
                f.write(f"{'Pregunta' if 'e' in args_json['mode'] else 'Question'}: {question}\n{response}\n\n")
        if audio:
            print()
            read_aloud(filename_out)
    else:
        if args_json["mode"] in ['kp', 'kpe']:
            print('--------------------------------------')
            print("* key points")
            response = summarize_keypoints(trimmed_transcript, language="es" if args_json["mode"] == 'kpe' else "en")
            print(response)
            output_filename = f"{segment[:-4]}_keypoints.txt"
        elif args_json["mode"] in ['s', 'se']:
            print('--------------------------------------')
            print("* summarizing")
            response = summarize(trimmed_transcript, language="es" if args_json["mode"] == 'se' else "en")
            print(response)
            output_filename = f"{segment[:-4]}_summary.txt"
        elif args_json["mode"] == 'tr':
            return
        else:
            raise ValueError(f"Invalid mode: {args_json['mode']}")
        with open(os.path.join(path, output_filename), "w") as f:
            f.write(response)
        if audio:
            read_aloud(os.path.join(path, output_filename))

    # clean up
    os.remove(os.path.join(path, video_name))
    os.remove(os.path.join(path, segment))

if __name__ == "__main__":
    main()