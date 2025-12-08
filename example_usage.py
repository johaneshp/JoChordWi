"""
Example usage of the AudioSeparator class.
Shows different ways to use the audio separation functionality.
"""

from separator import AudioSeparator


def example_1_simple_usage():
    """Example 1: Simple one-line usage"""
    print("=" * 60)
    print("Example 1: Simple Usage")
    print("=" * 60)

    separator = AudioSeparator()
    result = separator.separate_audio_file("path/to/your/audio.mp3")

    print(f"\nResults saved to: {result['output_dir']}")
    print(f"Vocal track: {result['files']['vocals']}")
    print(f"Instrumental tracks: {result['files']['drums']}, {result['files']['bass']}, {result['files']['other']}")


def example_2_custom_output():
    """Example 2: Specify custom output directory"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Output Directory")
    print("=" * 60)

    separator = AudioSeparator()
    result = separator.separate_audio_file(
        "path/to/your/song.mp4",
        output_dir="my_custom_output/my_project"
    )

    print(f"\nResults: {result}")


def example_3_batch_processing():
    """Example 3: Process multiple files with the same model instance"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing (Reuses Model)")
    print("=" * 60)

    files = [
        "song1.mp3",
        "song2.wav",
        "video1.mp4"
    ]

    separator = AudioSeparator()
    separator.load_model()  # Load model once

    for file in files:
        try:
            result = separator.separate_audio_file(file)
            print(f"\n✓ Processed {file}")
            print(f"  Output: {result['output_dir']}")
        except Exception as e:
            print(f"\n✗ Failed to process {file}: {e}")


def example_4_advanced_manual_control():
    """Example 4: Advanced usage with manual control of each step"""
    print("\n" + "=" * 60)
    print("Example 4: Advanced Manual Control")
    print("=" * 60)

    from file_parse import prepare_audio_file

    separator = AudioSeparator()
    separator.load_model()

    # Step 1: Prepare audio file
    audio_file, output_dir = prepare_audio_file("video.mp4")

    # Step 2: Load audio
    waveform, sample_rate = separator.load_audio(audio_file)

    # Step 3: Resample if needed
    waveform, sample_rate = separator.resample_if_needed(waveform, sample_rate)

    # Step 4: Separate
    separated_sources = separator.separate(waveform)

    # Step 5: Save (or process further)
    saved_files = separator.save_sources(separated_sources, output_dir, sample_rate)

    print(f"\nManual processing complete!")
    print(f"Separated files: {saved_files}")


def example_5_get_vocals_only():
    """Example 5: Extract only vocals without saving other tracks"""
    print("\n" + "=" * 60)
    print("Example 5: Extract Vocals Only")
    print("=" * 60)

    from file_parse import prepare_audio_file
    import soundfile as sf

    separator = AudioSeparator()
    separator.load_model()

    audio_file, output_dir = prepare_audio_file("song.mp3")
    waveform, sample_rate = separator.load_audio(audio_file)
    waveform, sample_rate = separator.resample_if_needed(waveform, sample_rate)

    separated_sources = separator.separate(waveform)

    # Get only vocals (index 3)
    vocals = separated_sources[3]
    vocals_path = f"{output_dir}/vocals_only.wav"
    sf.write(vocals_path, vocals.numpy().T, sample_rate)

    print(f"\nVocals saved to: {vocals_path}")


if __name__ == "__main__":
    print("AudioSeparator Usage Examples")
    print("=" * 60)
    print("\nNote: Update file paths before running these examples!\n")

    # Uncomment the example you want to run:

    # example_1_simple_usage()
    # example_2_custom_output()
    # example_3_batch_processing()
    # example_4_advanced_manual_control()
    # example_5_get_vocals_only()

    print("\n" + "=" * 60)
    print("To use in your code, simply import and call:")
    print("=" * 60)
    print("""
from separator import AudioSeparator

separator = AudioSeparator()
result = separator.separate_audio_file("your_file.mp4")
print(result['files']['vocals'])  # Path to vocals file
    """)
