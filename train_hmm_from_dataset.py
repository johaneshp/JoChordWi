

import argparse
from pathlib import Path
import json
from hmm_chord_trainer import (
    ChordHMM,
    parse_json_chord_sequences,
    parse_chordonomicon_dataset
)


def train_from_json_files(json_dir, output_model='models/chord_hmm.pkl'):
    """Train HMM from your own JSON chord files."""
    json_dir = Path(json_dir)

    # Find all JSON files
    json_files = list(json_dir.rglob('*.json'))

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return None

    print(f"Loading chord sequences from {len(json_files)} JSON files...")
    sequences = parse_json_chord_sequences(json_files)

    if not sequences:
        print("No chord sequences found! Check JSON format.")
        return None

    print(f"Loaded {len(sequences)} chord sequences")

    # Train HMM
    hmm = ChordHMM(smoothing=0.01)
    hmm.train(sequences)

    # Save model
    output_path = Path(output_model)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    hmm.save(output_path)

    return hmm


def train_from_chordonomicon(output_model='models/chord_hmm.pkl', max_songs=None, genre=None):
    """Train HMM from Chordonomicon dataset (666k+ songs from Hugging Face)."""
    print("=" * 70)
    print("Training from Chordonomicon Dataset")
    print("=" * 70)
    print(f"Dataset: ailsntua/Chordonomicon (666,000+ songs)")
    if genre:
        print(f"Genre filter: {genre}")
    if max_songs:
        print(f"Max songs: {max_songs}")
    print()

    # Load sequences
    sequences = parse_chordonomicon_dataset(max_songs=max_songs, genre_filter=genre)

    if not sequences:
        print("Failed to load dataset!")
        print("\nMake sure you have installed the datasets library:")
        print("  pip install datasets")
        return None

    print(f"\nTotal chord sequences loaded: {len(sequences)}")

    # Train HMM
    print("\nTraining HMM model...")
    hmm = ChordHMM(smoothing=0.01)
    hmm.train(sequences)

    # Save model
    output_path = Path(output_model)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    hmm.save(output_path)

    return hmm


def test_model(model_path):
    """Test the trained model."""
    hmm = ChordHMM()
    hmm.load(model_path)

    # Test sequences
    test_sequences = [
        ['C', 'X', 'Am', 'X', 'C', 'X'],
        ['G', 'X', 'Em', 'X', 'D'],
        ['X', 'F', 'G', 'X', 'Am'],
    ]

    print("\n" + "=" * 60)
    print("Testing HMM Chord Completion")
    print("=" * 60)

    for seq in test_sequences:
        filled = hmm.fill_uncertain_chords(seq)
        print(f"\nOriginal: {' -> '.join(seq)}")
        print(f"Filled:   {' -> '.join(filled)}")

    # Show common transitions
    common_chords = ['C', 'G', 'Am', 'F', 'D', 'Em']
    print("\n" + "=" * 60)
    print("Common Chord Transitions")
    print("=" * 60)

    for chord in common_chords:
        transitions = hmm.get_most_common_transitions(chord, top_k=3)
        if transitions:
            print(f"\n{chord} most commonly goes to:")
            for next_chord, prob in transitions:
                print(f"  -> {next_chord}: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train HMM for chord completion from datasets'
    )
    parser.add_argument(
        '--source',
        choices=[ 'json', 'chordonomicon', 'download-info'],
        default='chordonomicon',
        help='Dataset source to use (default: chordonomicon)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/billboard',
        help='Path to dataset directory (for billboard/json sources)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/chord_hmm.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--max-songs',
        type=int,
        default=None,
        help='Maximum number of songs to use (for chordonomicon)'
    )
    parser.add_argument(
        '--genre',
        type=str,
        default=None,
        help='Filter by genre (for chordonomicon): rock, pop, jazz, etc.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the trained model'
    )

    args = parser.parse_args()

    if args.source == 'chordonomicon':
        hmm = train_from_chordonomicon(args.output, args.max_songs, args.genre)
        if hmm and args.test:
            test_model(args.output)

    elif args.source == 'json':
        hmm = train_from_json_files(args.dataset_dir, args.output)
        if hmm and args.test:
            test_model(args.output)


if __name__ == "__main__":
    main()
