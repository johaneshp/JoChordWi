import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from pathlib import Path
import soundfile as sf
from file_parse import prepare_audio_file


class AudioSeparator:
    """
    Audio source separation using Hybrid Demucs model.
    Separates audio into drums, bass, other instruments, and vocals.
    """

    def __init__(self):
        """Initialize the Demucs model."""
        self.bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = None
        self.source_names = ["drums", "bass", "other", "vocals"]

    def load_model(self):
        """Load the Demucs model. Call this before separation."""
        if self.model is None:
            print(f"Loading model...")
            self.model = self.bundle.get_model()
        return self

    def load_audio(self, audio_file):
        """
        Load audio file and return waveform tensor.

        Args:
            audio_file (str): Path to audio file

        Returns:
            tuple: (waveform tensor, sample_rate)
        """
        print(f"Loading audio file: {audio_file}")
        audio_data, sample_rate = sf.read(audio_file, dtype='float32')
        waveform = torch.from_numpy(audio_data.T)
        return waveform, sample_rate

    def resample_if_needed(self, waveform, sample_rate):
        """
        Resample audio to model's expected sample rate if needed.

        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Current sample rate

        Returns:
            tuple: (resampled waveform, target sample_rate)
        """
        if sample_rate != self.bundle.sample_rate:
            print(f"Resampling from {sample_rate}Hz to {self.bundle.sample_rate}Hz...")
            waveform = torchaudio.transforms.Resample(sample_rate, self.bundle.sample_rate)(waveform)
            sample_rate = self.bundle.sample_rate
        return waveform, sample_rate

    def separate(self, waveform):
        """
        Separate audio into different sources.

        Args:
            waveform (torch.Tensor): Audio waveform

        Returns:
            torch.Tensor: Separated sources [drums, bass, other, vocals]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("Separating audio sources...")
        with torch.no_grad():
            separated_sources = self.model(waveform.unsqueeze(0))[0]
        return separated_sources

    def save_sources(self, separated_sources, output_dir, sample_rate):
        """
        Save separated audio sources to files.

        Args:
            separated_sources (torch.Tensor): Separated audio sources
            output_dir (str or Path): Directory to save files
            sample_rate (int): Sample rate for output files

        Returns:
            dict: Dictionary mapping source names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        print(f"Saving separated tracks to {output_dir}/")

        for i, name in enumerate(self.source_names):
            output_path = output_dir / f"{name}.wav"
            audio_np = separated_sources[i].numpy().T
            sf.write(str(output_path), audio_np, sample_rate)
            saved_files[name] = str(output_path)
            print(f"  ✓ Saved {name}.wav")

        return saved_files

    def separate_audio_file(self, file_path, output_dir=None):
        """
        Complete pipeline: separate an audio or video file into stems.

        Args:
            file_path (str): Path to audio or video file
            output_dir (str or Path, optional): Output directory. If None, auto-generates.

        Returns:
            dict: Dictionary with separation results
                {
                    'output_dir': str,
                    'files': {source_name: file_path, ...}
                }
        """
        # Prepare audio file (extract from video if needed)
        audio_file, auto_output_dir = prepare_audio_file(file_path, output_dir)

        if output_dir is None:
            output_dir = auto_output_dir

        # Load model if not already loaded
        self.load_model()

        # Load and process audio
        waveform, sample_rate = self.load_audio(audio_file)
        waveform, sample_rate = self.resample_if_needed(waveform, sample_rate)

        # Separate sources
        separated_sources = self.separate(waveform)

        # Save results
        saved_files = self.save_sources(separated_sources, output_dir, sample_rate)

        print(f"\nSeparation complete! Files saved in: {output_dir}")

        return {
            'output_dir': str(output_dir),
            'files': saved_files
        }
