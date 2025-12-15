import librosa
import numpy as np
import json
import csv
from pathlib import Path
from hmm_chord_trainer import ChordHMM



def get_chord_sequence(audio_path, chunk_dur=60.0, overlap=2.0, output_format='json', output_dir=None, use_hmm=True, hmm_model_path='models/chord_hmm.pkl'):
    import crema
    model = crema.models.chord.ChordModel()

    y, sr=librosa.load(audio_path)
    dur=librosa.get_duration(y=y, sr=sr)
    if dur<=60.0:
        chord_data = model.predict(audio_path)
        chords=_extract_chords(chord_data, time_offset=0.0)
    else:
        chords=_process_chunks(y, sr, chunk_dur, overlap, model)

    # Apply HMM to fill uncertain chords
    if use_hmm:
        chords = _apply_hmm_correction(chords, hmm_model_path)

    output_path=_save_chords(chords, audio_path, output_format, output_dir)
    print(f"Chords saved in : {output_path}")
    return chords

def _process_chunks(y, sr, chunk_duration, overlap, model):
    """Process audio in overlapping chunks."""
    chords = []
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    num_chunks = int(np.ceil((len(y) - overlap_samples) / step_samples))
    for i in range(num_chunks):
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, len(y))
        time_offset = start_sample / sr
        chunk_y = y[start_sample:end_sample]
        print(f"  Chunk {i+1}/{num_chunks}: {time_offset:.2f}s - {end_sample/sr:.2f}s")
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, chunk_y, sr)

            # Predict chords for this chunk
            chord_data = model.predict(tmp_file.name)
            chunk_chords = _extract_chords(chord_data, time_offset)

            # Remove chords in overlap region (except for last chunk)
            if i < num_chunks - 1:
                chunk_end = time_offset + chunk_duration
                chunk_chords = [c for c in chunk_chords
                               if c['start'] < chunk_end - overlap/2]

            chords.extend(chunk_chords)

            # Clean up temp file
            Path(tmp_file.name).unlink()

    # Merge consecutive identical chords
    chords = _merge_consecutive_chords(chords)

    return chords

def _extract_chords(chord_labels, time_offset=0.0):
    """Extract chord information from CREMA predictions."""
    chords = []
    for obs in chord_labels:
        chords.append({
            'start': obs.time + time_offset,
            'end': obs.time + obs.duration + time_offset,
            'chord': obs.value,
            'confidence': obs.confidence
        })
    return chords


def _merge_consecutive_chords(chords):
    """Merge consecutive chords with the same label."""
    if not chords:
        return chords

    merged = [chords[0].copy()]

    for chord in chords[1:]:
        last = merged[-1]

        # If same chord and close in time (within 0.5s), merge
        if (chord['chord'] == last['chord'] and
            abs(chord['start'] - last['end']) < 0.5):
            last['end'] = chord['end']
            last['confidence'] = max(last['confidence'], chord['confidence'])
        else:
            merged.append(chord.copy())

    return merged


def _apply_hmm_correction(chords, hmm_model_path):
    hmm_model_path = Path(hmm_model_path)

    if not hmm_model_path.exists():
        print(f"Warning: HMM model not found at {hmm_model_path}. Skipping HMM correction.")
        print("Train a model using: python train_hmm_from_dataset.py")
        return chords

    try:
        # Load HMM model
        hmm = ChordHMM()
        hmm.load(hmm_model_path)

        # Extract chord sequence and confidences
        chord_labels = [c['chord'] for c in chords]
        confidences = [c['confidence'] for c in chords]

        # Count uncertain chords before
        uncertain_before = sum(1 for c in chord_labels if c in ('X', 'N'))

        # Fill uncertain chords
        filled_labels = hmm.fill_uncertain_chords(chord_labels, confidences)

        # Update chord data
        for i, new_chord in enumerate(filled_labels):
            if chords[i]['chord'] != new_chord:
                chords[i]['chord'] = new_chord
                chords[i]['hmm_corrected'] = True

        # Count uncertain chords after
        uncertain_after = sum(1 for c in filled_labels if c in ('X', 'N'))
        print(f"HMM correction: {uncertain_before} uncertain chords -> {uncertain_after} uncertain chords")
        print(f"Filled {uncertain_before - uncertain_after} uncertain chords")

    except Exception as e:
        print(f"Error applying HMM correction: {e}")

    return chords


def _save_chords(chords, audio_path, output_format='json', output_dir=None):
    """Save chord sequence to file."""
    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if output_format == 'json':
        output_path = output_dir / 'chords.json'
        with open(output_path, 'w') as f:
            json.dump(chords, f, indent=2)
    else:  # csv
        output_path = output_dir / 'chords.csv'
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['start', 'end', 'chord', 'confidence'])
            writer.writeheader()
            writer.writerows(chords)

    return output_path
 