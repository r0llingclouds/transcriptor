from utils import *
import argparse
import datetime

def load_transcript(file_path):
    with open(file_path, 'r') as file:
        transcript = file.read()
    return transcript

def main():
    parser = argparse.ArgumentParser(description="Process a transcript text file")
    parser.add_argument("mode", help="summarize (s) or key points (kp) or question answering (q)")
    parser.add_argument("file_path", help="Path to the transcript text file")
    parser.add_argument("question", nargs='?', help="question to answer")

    args = parser.parse_args()

    transcript = load_transcript(args.file_path)
    trimmed_transcript = load_and_trim_text(transcript, 26000)

    if args.question:
        print("* question answering")
        response = qna(trimmed_transcript, args.question)
        print(args.question)
        print()
        print(response)
        with open(f"{args.file_path[:-4]}_qna_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt", "w") as f:
            f.write(args.question + "\n\n")
            f.write(response)
    else:
        print("* summarizing")
        response = summarize(trimmed_transcript)
        print(response)
        with open(f"{args.file_path[:-4]}_summary.txt", "w") as f:
            f.write(response)
        print()
        print("* key points")
        response = summarize_keypoints(trimmed_transcript)
        print(response)
        with open(f"{args.file_path[:-4]}_keypoints.txt", "w") as f:
            f.write(response)

if __name__ == "__main__":
    main()
