import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from chord_detect import get_chord_sequence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_chord_detect.py <audio_path> [output_dir] [output_format]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'json'

    print(f"Starting chord detection for: {audio_path}")
    chords = get_chord_sequence(audio_path, output_dir=output_dir, output_format=output_format)
    print(f"Detected {len(chords)} chord segments")
