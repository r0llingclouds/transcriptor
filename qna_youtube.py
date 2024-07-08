from utils import extract_audio_segment, download_youtube_video, get_transcription, load_and_trim_text, qna
import os
import argparse
import datetime

def main():
    parser = argparse.ArgumentParser(description="QNA on youtube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start_time", help="start time (e.g., 2h34m45s)")
    parser.add_argument("end_time", help="end time (e.g., 2h34m45s)")
    parser.add_argument("question", help="question about the video")

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

    # summarize the transcript
    trimmed_transcript = load_and_trim_text(transcript, 26000)

    print("* answering")
    response = qna(trimmed_transcript, args.question)
    print(args.question)
    print()
    print(response)
    with open(f"{args.file_path[:-4]}_qna_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt", "w") as f:
        f.write(args.question + "\n\n")
        f.write(response)

    # clean up
    os.remove(os.path.join(path, video_name))
    os.remove(os.path.join(path, segment))

if __name__ == "__main__":
    main()