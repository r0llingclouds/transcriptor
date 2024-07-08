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

def summarize(transcription, language="en"):
    if language.lower() == "es":
        system_content = "Eres una IA altamente capacitada en comprensión y resumen de textos. Voy a darte un texto. Resúmelo, extrayendo la información más importante. Intenta retener los puntos más importantes. Devuelve un resumen del texto."
    else:  # default to English
        system_content = "You are a highly skilled AI trained in language comprehension and summarization. I'm going to give you a text. Summarize it, extracting the most important information. Aim to retain the most important points. Return a summary of the text."

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def summarize_keypoints(transcription, language="en"):
    if language.lower() == "es":
        system_content = "Eres una IA altamente capacitada en comprensión y resumen de textos. Lee el siguiente texto y extrae los puntos clave del mismo. Intenta retener los puntos más importantes y devuelve estos puntos clave en una lista. Puedes dar un breve resumen antes de la lista."
    else:  # default to English
        system_content = "You are a highly skilled AI trained in language comprehension and summarization. Read the following text and extract the key points of it. Aim to retain the most important points and return these key points in a list. You can give a brief abstract before the list."

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def qna(context, question, language="en"):
    if language.lower() == "es":
        system_content = "Eres un asistente útil. Responde la siguiente pregunta basada en el contexto dado."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Contexto: {context}"},
            {"role": "user", "content": f"Pregunta: {question}"}
        ]
    else:  # default to English
        system_content = "You are a helpful assistant. Answer the following question based on the given context."
        messages = [
            {"role": "system", "content": system_content},
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

    response_text = response.choices[0].message.content.strip()
    response_text = response_text.replace('null', 'None')

    response_text = eval(response_text)
    return response_text

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos or segments")
    parser.add_argument("data", help="mode link start end questions")
    args = parser.parse_args()
    args_json = extract_arguments(args.data)

    path = "/Users/tirsolopezausens/Downloads"

    if args_json["mode"] is None:
        args_json["mode"] = 'kp' # default get keypoints

    print(f"* downloading")
    url = args_json["youtube_link"]
    video_name = download_youtube_video(url, path)

    print(f"* extracting segment")
    if args_json["start_time"] is None:
        args_json["start_time"] = "0s"
    if args_json["end_time"] is None:
        audio = AudioSegment.from_file(os.path.join(path, video_name))
        video_length = len(audio) / 1000
        args_json["end_time"] = f"{int(video_length // 3600)}h{int((video_length % 3600) // 60)}m{int(video_length % 60)}s"

    print(f"* data {args_json}")

    segment = extract_audio_segment(video_name, path, args_json["start_time"], args_json["end_time"])

    print(f"* transcribing")
    transcript = get_transcription(segment, path)
    with open(os.path.join(path, f"{segment[:-4]}.txt"), "w") as f:
        f.write(transcript)

    trimmed_transcript = load_and_trim_text(transcript, 26000)

    # operations
    qna_dict = dict()
    if len(args_json["question"]) > 0:
        for question in args_json["question"]:
            print('--------------------------------------')
            print("* question answering")
            if args_json["mode"] == 'qe':
                response = qna(trimmed_transcript, question, language="es")
            else:
                response = qna(trimmed_transcript, question)
            qna_dict[question] = response
            print(question)
            print(response)
            print()
        with open(os.path.join(path, f"{segment[:-4]}_qna_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt"), "w") as f:
            for question, response in qna_dict.items():
                f.write(f"Question: {question}\nAnswer: {response}\n\n\n")
    else:
        if args_json["mode"] == 'kp':
            print('--------------------------------------')
            print("* key points")
            response = summarize_keypoints(trimmed_transcript)
            print(response)
            with open(os.path.join(path, f"{segment[:-4]}_keypoints.txt"), "w") as f:
                f.write(response)
        elif args_json["mode"] == 's':
            print('--------------------------------------')
            print("* summarizing")
            response = summarize(trimmed_transcript)
            print(response)
            with open(os.path.join(path, f"{segment[:-4]}_summary.txt"), "w") as f:
                f.write(response)
        elif args_json["mode"] == 'kpe':
            print('--------------------------------------')
            print("* key points")
            response = summarize_keypoints(trimmed_transcript, language="es")
            print(response)
            with open(os.path.join(path, f"{segment[:-4]}_keypoints.txt"), "w") as f:
                f.write(response)
        elif args_json["mode"] == 'se':
            print('--------------------------------------')
            print("* summarizing")
            response = summarize(trimmed_transcript, language="es")
            print(response)
            with open(os.path.join(path, f"{segment[:-4]}_summary.txt"), "w") as f:
                f.write(response)
        elif args_json["mode"] == 'tr': # just get the transcript
            pass
        else:
            raise ValueError(f"Invalid mode: {args_json['mode']}")

    # clean up
    os.remove(os.path.join(path, video_name))
    os.remove(os.path.join(path, segment))

if __name__ == "__main__":
    client = OpenAI()    
    main()