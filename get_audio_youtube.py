from utils import download_youtube_video, extract_audio_segment
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

if __name__ == "__main__":
    main()