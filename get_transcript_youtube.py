from utils import extract_audio_segment, download_youtube_video, get_transcription
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos or segments")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start_time", help="start time (e.g., 2h34m45s)")
    parser.add_argument("end_time", help="end time (e.g., 2h34m45s)")

    args = parser.parse_args()

    path = "/Users/tirsolopezausens/Downloads"
    print(f"* downloading")
    url = args.url.split("&")[0]
    video_name = download_youtube_video(url, path)

    print(f"* extracting segment")
    segment = extract_audio_segment(video_name, path, args.start_time, args.end_time)
    transcript = get_transcription(segment, path)
    with open(os.path.join(path, f"{segment[:-4]}.txt"), "w") as f:
        f.write(transcript)

    # clean up
    os.remove(os.path.join(path, video_name))
    os.remove(os.path.join(path, segment))

if __name__ == "__main__":
    main()