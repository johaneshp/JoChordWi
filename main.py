from separator import AudioSeparator
from file_parse import prepare_audio_file
def main():
    sep=AudioSeparator()
    sep.load_model()
    file_path=input("Enter file path to the audio/video:")
    audio_file, output_dir=prepare_audio_file(file_path)
    waveform, sample_rate=sep.load_audio(audio_file)
    waveform, sample_rate=sep.resample_if_needed(waveform,sample_rate)
    separated_sources=sep.separate(waveform)
    sep.save_sources(separated_sources, output_dir, sample_rate)

if __name__ == "__main__":
    main()