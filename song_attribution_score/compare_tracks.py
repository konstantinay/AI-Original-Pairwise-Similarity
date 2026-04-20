import numpy as np
import argparse
from pathlib import Path
import json
import random
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from attribution_score import AttributionDetector

# Validate using MIPPIA dataset
def validate_on_mippia(detector, dataset_dir='../smp_dataset/final_dataset_clean', max_pairs=10, plot_roc=True):
    """
    Test detector on MIPPIA similar pairs (ground truth) with negative sampling
    
    Automatically detects if detector uses extended features (HNR, phase continuity)
    and includes them in the similarity computation.
    """
    # Fix random seed
    random.seed(42)
    
    dataset_path = Path(dataset_dir)
    
    # Collect all valid pair folders
    all_pair_folders = []
    for pair_folder in sorted(dataset_path.iterdir()):
        if pair_folder.is_dir():
            audio_files = list(pair_folder.glob('*.wav'))
            if len(audio_files) == 2:
                all_pair_folders.append((pair_folder, audio_files))
    
    all_pair_folders = all_pair_folders[:max_pairs]
    
    print(f"Validating on MIPPIA pairs")
    
    # POSITIVE PAIRS (similar songs)
    print("Processing positive pairs (similar songs)")
    positive_results = []
    
    for pair_folder, audio_files in all_pair_folders:
        result = detector.compare_tracks(
            audio_files[0], 
            audio_files[1]
        )
        result['label'] = 1  # Positive pair
        positive_results.append(result)
        print(f"Pair {pair_folder.name}: {result['overall_similarity']:.3f}")
    
    # NEGATIVE PAIRS (unrelated songs)
    print("\nGenerating negative pairs (unrelated songs)")
    negative_results = []
    
    # Create negative pairs by randomly pairing songs from different folders
    num_negatives = len(positive_results)  # Same number as positives
    
    for i in range(num_negatives):
        # Pick two different pair folders randomly
        if len(all_pair_folders) < 2:
            break
            
        folder1, folder2 = random.sample(all_pair_folders, 2)
        
        # Pick one song from each folder
        song1 = random.choice(folder1[1])
        song2 = random.choice(folder2[1])
        
        result = detector.compare_tracks(
            song1,
            song2
        )
        result['label'] = 0  # Negative pair
        negative_results.append(result)
        print(f"  Negative {i+1}: {result['overall_similarity']:.3f}")
    
    all_results = positive_results + negative_results
    
    # Extract scores and labels
    y_true = np.array([r['label'] for r in all_results])
    y_scores = np.array([r['overall_similarity'] for r in all_results])
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    pos_avg = np.mean([r['overall_similarity'] for r in positive_results])
    neg_avg = np.mean([r['overall_similarity'] for r in negative_results])
    
    matches = sum(1 for r in positive_results if r['is_attributed'])
    false_positives = sum(1 for r in negative_results if r['is_attributed'])
    
    # PRINT SUMMARY
    print(f"Validation Summary:")
    print(f"  Extended Features: {'ENABLED' if detector.use_extended else 'DISABLED'}")
    print(f"  Positive pairs tested: {len(positive_results)}")
    print(f"  Negative pairs tested: {len(negative_results)}")
    print(f"  Average similarity (positive): {pos_avg:.3f}")
    print(f"  Average similarity (negative): {neg_avg:.3f}")
    print(f"  Detected as similar (TP): {matches}/{len(positive_results)}")
    print(f"  False positives (FP): {false_positives}/{len(negative_results)}")
    print(f"  ROC AUC Score: {roc_auc:.3f}")
    
    # PLOT ROC CURVE
    if plot_roc:
        feature_label = "Extended" if detector.use_extended else "Standard"
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Audio Attribution ({feature_label} Features)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f'roc_curve_approach1_{"extended" if detector.use_extended else "standard"}.png'
        plt.savefig(filename, dpi=150)
        print(f"ROC curve saved to: {filename}")
        plt.show()
    
    return {
        'all_results': all_results,
        'positive_results': positive_results,
        'negative_results': negative_results,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

#Validate using SONICS dataset (real vs AI-generated)
def validate_on_sonics(detector, 
                       real_dir='../sonics_dataset/real_songs',
                       fake_dir='../sonics_dataset/fake_songs/fake_songs',
                       max_pairs=20):
    """Test detector on SONICS real vs AI-generated songs"""
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)
    
    # Collect audio files (SONICS is in mp3 format)
    real_files = list(real_path.glob('*.mp3'))[:max_pairs] if real_path.exists() else []
    fake_files = list(fake_path.glob('*.mp3'))[:max_pairs] if fake_path.exists() else []
    
    if not real_files or not fake_files:
        print(f"Error: SONICS files not found in {real_dir} or {fake_dir}")
        return None
    
    print(f"Validating on SONICS (Real vs AI-generated)")
    print(f"Real songs found: {len(real_files)}")
    print(f"Fake songs found: {len(fake_files)}")
    
    # create real vs fake pairs
    print("\nProcessing real vs fake pairs")
    results = []
    
    num_pairs = min(len(real_files), len(fake_files), max_pairs)
    
    for i in range(num_pairs):
        # Randomly pair real and fake songs
        real_file = random.choice(real_files)
        fake_file = random.choice(fake_files)
        
        result = detector.compare_tracks(
            real_file,
            fake_file
        )
        result['label'] = 0  # Negative pair (unrelated)
        results.append(result)
        print(f"  Pair {i+1}: {result['overall_similarity']:.3f}")
    
    avg_similarity = np.mean([r['overall_similarity'] for r in results])
    false_positives = sum(1 for r in results if r['is_attributed'])
    
    print(f"SONICS Validation Summary:")
    print(f"  Extended Features: {'ENABLED' if detector.use_extended else 'DISABLED'}")
    print(f"  Pairs tested (real vs AI): {len(results)}")
    print(f"  Average similarity: {avg_similarity:.3f}")
    print(f"  False positives (incorrectly matched): {false_positives}/{len(results)}")
    
    return results

# CLI INTERFACE
def main():
    parser = argparse.ArgumentParser(description='Audio Attribution Detection - Approach 1')
    parser.add_argument('--track_a', type=str, help='Path to first audio track')
    parser.add_argument('--track_b', type=str, help='Path to second audio track')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Similarity threshold for attribution (0-1)')
    parser.add_argument('--extended-features', action='store_true',
                       help='Use extended features (HNR, phase continuity, spectral flux)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on MIPPIA dataset')
    parser.add_argument('--validate_sonics', action='store_true',
                       help='Run validation on SONICS dataset (real vs AI)')
    parser.add_argument('--validate_all', action='store_true',
                       help='Run validation on both MIPPIA and SONICS')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AttributionDetector(
        similarity_threshold=args.threshold,
        use_extended=args.extended_features
    )
    
    if args.extended_features:
        print("Using EXTENDED features (HNR, phase continuity, spectral flux)\n")
    else:
        print("Using STANDARD features only\n")
    
    if args.validate or args.validate_all:
        # MIPPIA validation mode
        results = validate_on_mippia(detector)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"MIPPIA results saved to {args.output}")
    
    if args.validate_sonics or args.validate_all:
        # SONICS validation mode
        sonics_results = validate_on_sonics(detector)
        
        if args.output and sonics_results:
            output_path = args.output.replace('.json', '_sonics.json') if args.output else 'sonics_results.json'
            with open(output_path, 'w') as f:
                json.dump(sonics_results, f, indent=2)
            print(f"SONICS results saved to {output_path}")
    
    elif args.track_a and args.track_b:
        # Compare two tracks
        result = detector.compare_tracks(args.track_a, args.track_b)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

# USAGE EXAMPLES
"""
Validation on MIPPIA (standard features):
python compare_tracks.py --validate

Validation with extended features:
python compare_tracks.py --validate --extended-features

SONICS validation:
python compare_tracks.py --validate_sonics

Single pair comparison:
python compare_tracks.py --track_a file1.wav --track_b file2.wav
"""