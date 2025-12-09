import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from file_parse import prepare_audio_file
from separator import AudioSeparator
import subprocess
def main():
    sep=AudioSeparator()
    sep.load_model()
    file_path=input("Enter file path to the audio/video:")
    audio_file, output_dir=prepare_audio_file(file_path)
    waveform, sample_rate=sep.load_audio(audio_file)
    waveform, sample_rate=sep.resample_if_needed(waveform,sample_rate)
    separated_sources=sep.separate(waveform)
    saved_files=sep.save_sources(separated_sources, output_dir, sample_rate)
    print("\nStarting chord detection (in separate process)...")
    chord_script = os.path.join(os.path.dirname(__file__), "run_chord_detect.sh")
    result = subprocess.run([
        chord_script,
        saved_files["other"],
        output_dir,
        "json"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        print("Chord detection complete!")
    else:
        print("Chord detection failed:")
        print(result.stderr)

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 
    main()