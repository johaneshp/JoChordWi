import numpy as np
import json
from pathlib import Path
import pickle


class ChordHMM:
    """Hidden Markov Model for chord sequences."""

    def __init__(self, smoothing=0.01):
        self.smoothing = smoothing
        self.chord_to_idx = {}
        self.idx_to_chord = {}
        self.transition_matrix = None
        self.start_probs = None
        self.trained = False

    def train(self, chord_sequences):
        # Build vocabulary
        all_chords = set()
        for seq in chord_sequences:
            all_chords.update(seq)

        # Remove 'X' and 'N' from training (uncertain/no chord)
        all_chords.discard('X')
        all_chords.discard('N')

        # Create mappings
        sorted_chords = sorted(all_chords)
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(sorted_chords)}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}

        n_states = len(self.chord_to_idx)

        # Initialize transition and start counts
        transition_counts = np.zeros((n_states, n_states))
        start_counts = np.zeros(n_states)

        # Count transitions
        for seq in chord_sequences:
            # Filter out X and N
            filtered_seq = [c for c in seq if c in self.chord_to_idx]

            if not filtered_seq:
                continue

            # Count start chord
            start_idx = self.chord_to_idx[filtered_seq[0]]
            start_counts[start_idx] += 1

            # Count transitions
            for i in range(len(filtered_seq) - 1):
                curr_idx = self.chord_to_idx[filtered_seq[i]]
                next_idx = self.chord_to_idx[filtered_seq[i + 1]]
                transition_counts[curr_idx, next_idx] += 1

        # Apply smoothing and normalize to get probabilities
        self.transition_matrix = (transition_counts + self.smoothing) / \
                                (transition_counts.sum(axis=1, keepdims=True) + self.smoothing * n_states)

        self.start_probs = (start_counts + self.smoothing) / \
                          (start_counts.sum() + self.smoothing * n_states)

        self.trained = True
        print(f"Trained HMM with {n_states} chord states")

    def fill_uncertain_chords(self, chord_sequence, confidences=None):
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")

        # If no uncertain chords, return as-is
        if 'X' not in chord_sequence and 'N' not in chord_sequence:
            return chord_sequence

        result = chord_sequence.copy()
        n = len(result)

        # Use dynamic programming to find best replacements
        # For simplicity, we'll use a forward-backward approach

        for i, chord in enumerate(result):
            if chord in ('X', 'N'):
                # Look at context to determine best chord
                result[i] = self._predict_chord_from_context(result, i, confidences)

        return result

    def _predict_chord_from_context(self, sequence, pos, confidences):
        """Predict best chord for position using context."""
        scores = np.zeros(len(self.chord_to_idx))

        # Look at previous chord
        if pos > 0 and sequence[pos - 1] in self.chord_to_idx:
            prev_idx = self.chord_to_idx[sequence[pos - 1]]
            scores += self.transition_matrix[prev_idx, :]

        # Look at next chord (backward transition)
        if pos < len(sequence) - 1 and sequence[pos + 1] in self.chord_to_idx:
            next_idx = self.chord_to_idx[sequence[pos + 1]]
            # Weight by reverse transition probability
            scores += self.transition_matrix[:, next_idx]

        # If no context, use start probabilities
        if scores.sum() == 0:
            scores = self.start_probs

        # Return most likely chord
        best_idx = np.argmax(scores)
        return self.idx_to_chord[best_idx]

    def get_most_common_transitions(self, chord, top_k=5):
        """Get the most likely next chords after a given chord."""
        if chord not in self.chord_to_idx:
            return []

        idx = self.chord_to_idx[chord]
        probs = self.transition_matrix[idx, :]
        top_indices = np.argsort(probs)[-top_k:][::-1]

        return [(self.idx_to_chord[i], probs[i]) for i in top_indices]

    def save(self, filepath):
        data = {
            'chord_to_idx': self.chord_to_idx,
            'idx_to_chord': self.idx_to_chord,
            'transition_matrix': self.transition_matrix,
            'start_probs': self.start_probs,
            'smoothing': self.smoothing,
            'trained': self.trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.chord_to_idx = data['chord_to_idx']
        self.idx_to_chord = data['idx_to_chord']
        self.transition_matrix = data['transition_matrix']
        self.start_probs = data['start_probs']
        self.smoothing = data['smoothing']
        self.trained = data['trained']
        print(f"Model loaded from {filepath}")


def parse_json_chord_sequences(json_files):
    sequences = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract just the chord labels
            chords = [item['chord'] for item in data]
            if chords:
                sequences.append(chords)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    return sequences


def parse_chordonomicon_dataset(max_songs=None, genre_filter=None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return []

    print("Loading Chordonomicon dataset from Hugging Face...")
    print("(This may take a few minutes on first download)")

    try:
        # Load dataset
        dataset = load_dataset("ailsntua/Chordonomicon", split="train")

        # Apply genre filter if specified
        if genre_filter:
            dataset = dataset.filter(lambda x: genre_filter.lower() in str(x['main_genre']).lower())
            print(f"Filtered to {len(dataset)} songs with genre '{genre_filter}'")

        # Limit number of songs if specified
        if max_songs and max_songs < len(dataset):
            dataset = dataset.select(range(max_songs))
            print(f"Using {max_songs} songs")
        else:
            print(f"Using all {len(dataset)} songs")

        sequences = []
        skipped = 0

        for i, item in enumerate(dataset):
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} songs...")

            chord_text = item.get('chords', '')
            if not chord_text:
                skipped += 1
                continue
            
            chords = []
            for token in chord_text.split():
                # Skip section markers (e.g., <intro_1>, <verse_1>)
                if token.startswith('<') and token.endswith('>'):
                    continue

                # Add chord
                chords.append(token)

            if chords:
                sequences.append(chords)
            else:
                skipped += 1

        print(f"\nLoaded {len(sequences)} chord sequences")
        if skipped > 0:
            print(f"Skipped {skipped} songs with no valid chords")

        return sequences

    except Exception as e:
        print(f"Error loading Chordonomicon dataset: {e}")
        return []