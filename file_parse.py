import os
import subprocess
from pathlib import Path


def extract_audio_from_video(video_path, output_dir):
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path (str): Path to the video file
        output_dir (Path or str): Directory to save the extracted audio

    Returns:
        str: Path to the extracted audio file

    Raises:
        FileNotFoundError: If ffmpeg is not installed
        subprocess.CalledProcessError: If audio extraction fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_audio = output_dir / "extracted_audio.wav"

    # Use ffmpeg to extract audio
    subprocess.run([
        'ffmpeg', '-y',  # Overwrite output file if exists
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM audio codec
        '-ar', '44100',  # Sample rate
        '-ac', '2',  # Stereo
        str(extracted_audio)
    ], check=True, capture_output=True)

    return str(extracted_audio)


def is_video_file(file_path):
    """
    Check if a file is a video file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        bool: True if the file is a video, False otherwise
    """
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    file_extension = Path(file_path).suffix.lower()
    return file_extension in video_formats


def prepare_audio_file(file_path, output_dir=None):
    """
    Prepare an audio file for processing. If it's a video, extract the audio.

    Args:
        file_path (str): Path to the audio or video file
        output_dir (Path or str, optional): Directory to save extracted audio.
                                           If None, uses "output/project_{filename}"

    Returns:
        tuple: (audio_file_path, output_directory)

    Raises:
        FileNotFoundError: If the input file doesn't exist or ffmpeg is not installed
        subprocess.CalledProcessError: If audio extraction fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found!")

    # Create output directory if not specified
    if output_dir is None:
        base_filename = Path(file_path).stem
        output_dir = Path("output") / f"project_{base_filename}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if it's a video file
    if is_video_file(file_path):
        print(f"Detected video file. Extracting audio...")
        audio_file = extract_audio_from_video(file_path, output_dir)
        print(f"Audio extracted successfully!")
    else:
        audio_file = file_path

    return audio_file, str(output_dir)
