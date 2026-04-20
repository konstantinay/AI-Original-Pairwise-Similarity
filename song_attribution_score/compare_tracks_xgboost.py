import numpy as np
import argparse
from pathlib import Path
import random
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
from attribution_score_xgboost import AttributionDetector

# TRAINING
def train_model(dataset_dir='../smp_dataset/final_dataset_clean', 
                max_pairs=50, use_extended=False, output_file='model.pkl', train_offset=0):
    """Train XGBoost model on MIPPIA dataset"""
    
    # Fix random seed
    random.seed(42)
    np.random.seed(42)
    
    dataset_path = Path(dataset_dir)
    
    # Collect pairs
    all_pair_folders = []
    for pair_folder in sorted(dataset_path.iterdir()):
        if pair_folder.is_dir():
            audio_files = list(pair_folder.glob('*.wav'))
            if len(audio_files) == 2:
                all_pair_folders.append((pair_folder, audio_files))
    
    # Use offset to avoid overlap with validation
    all_pair_folders = all_pair_folders[train_offset:train_offset + max_pairs]
    
    print(f"Training XGBoost model on MIPPIA")
    print(f"Extended features: {'ENABLED' if use_extended else 'DISABLED'}")
    print(f"Pairs available: {len(all_pair_folders)}\n")
    
    detector = AttributionDetector(use_extended=use_extended)
    
    X = []
    y = []
    feature_names = None
    
    # Positive pairs
    print("Processing positive pairs")
    for i, (pair_folder, audio_files) in enumerate(all_pair_folders):
        result = detector.compare_tracks(audio_files[0], audio_files[1])
        metrics = result['detailed_scores']
        
        if feature_names is None:
            feature_names = list(metrics.keys())
        
        X.append([metrics[k] for k in feature_names])
        y.append(1)
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(all_pair_folders)}")
    
    # Negative pairs
    print("Generating negative pairs")
    num_negatives = len(all_pair_folders)
    for i in range(num_negatives):
        if len(all_pair_folders) < 2:
            break
        
        folder1, folder2 = random.sample(all_pair_folders, 2)
        song1 = random.choice(folder1[1])
        song2 = random.choice(folder2[1])
        
        result = detector.compare_tracks(song1, song2)
        metrics = result['detailed_scores']
        
        X.append([metrics[k] for k in feature_names])
        y.append(0)
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_negatives}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTraining XGBoost")
    print(f"  Samples: {len(y)} (positive={sum(y)}, negative={len(y)-sum(y)})")
    print(f"  Features: {feature_names}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Get predictions for ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  ROC AUC: {roc_auc:.3f}")
    
    # Feature importance
    print(f"Feature Importances:")
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"  {name:20s}: {imp:.3f}")
    
    # Save model
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'roc_auc': float(roc_auc),
        'use_extended': use_extended
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_file}\n")
    
    return model

# VALIDATION
def validate_on_mippia(detector, dataset_dir='../smp_dataset/final_dataset_clean', 
                       max_pairs=10, plot_roc=True, val_offset=50):
    """Validate on MIPPIA with ROC curve"""
    
    # Fix random seed
    random.seed(42)
    
    dataset_path = Path(dataset_dir)
    
    all_pair_folders = []
    for pair_folder in sorted(dataset_path.iterdir()):
        if pair_folder.is_dir():
            audio_files = list(pair_folder.glob('*.wav'))
            if len(audio_files) == 2:
                all_pair_folders.append((pair_folder, audio_files))
    
    all_pair_folders = all_pair_folders[val_offset:val_offset + max_pairs]
    
    print(f"Validating on MIPPIA")
    
    # Positive pairs
    print("Processing positive pairs")
    positive_results = []
    for pair_folder, audio_files in all_pair_folders:
        result = detector.compare_tracks(audio_files[0], audio_files[1])
        result['label'] = 1
        positive_results.append(result)
        print(f"  {pair_folder.name}: {result['overall_similarity']:.3f}")
    
    # Negative pairs
    print("\nGenerating negative pairs")
    negative_results = []
    num_negatives = len(positive_results)
    
    for i in range(num_negatives):
        if len(all_pair_folders) < 2:
            break
        
        folder1, folder2 = random.sample(all_pair_folders, 2)
        song1 = random.choice(folder1[1])
        song2 = random.choice(folder2[1])
        
        result = detector.compare_tracks(song1, song2)
        result['label'] = 0
        negative_results.append(result)
        print(f"  Negative {i+1}: {result['overall_similarity']:.3f}")
    
    # Compute metrics
    all_results = positive_results + negative_results
    y_true = np.array([r['label'] for r in all_results])
    y_scores = np.array([r['overall_similarity'] for r in all_results])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    pos_avg = np.mean([r['overall_similarity'] for r in positive_results])
    neg_avg = np.mean([r['overall_similarity'] for r in negative_results])
    
    matches = sum(1 for r in positive_results if r['is_attributed'])
    false_positives = sum(1 for r in negative_results if r['is_attributed'])
    
    print(f"\n{'='*60}")
    print(f"Validation Summary:")
    print(f"  Positive pairs avg: {pos_avg:.3f}")
    print(f"  Negative pairs avg: {neg_avg:.3f}")
    print(f"  True positives: {matches}/{len(positive_results)}")
    print(f"  False positives: {false_positives}/{len(negative_results)}")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"{'='*60}\n")
    
    if plot_roc:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - XGBoost')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve_xgboost.png', dpi=150)
        print("ROC curve saved to: roc_curve_xgboost.png")
        plt.show()
    
    return all_results

# CLI
def main():
    parser = argparse.ArgumentParser(description='XGBoost Attribution Detection')
    parser.add_argument('--track_a', type=str)
    parser.add_argument('--track_b', type=str)
    parser.add_argument('--train', action='store_true', help='Train XGBoost model')
    parser.add_argument('--validate', action='store_true', help='Validate on MIPPIA')
    parser.add_argument('--model', type=str, default='model.pkl', help='Model file path')
    parser.add_argument('--extended', action='store_true', help='Use extended features')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.train:
        train_model(use_extended=args.extended, output_file=args.model)
        return
    
    detector = AttributionDetector(
        model_path=args.model if Path(args.model).exists() else None,
        use_extended=args.extended,
        threshold=args.threshold
    )
    
    if args.validate:
        validate_on_mippia(detector)
    elif args.track_a and args.track_b:
        detector.compare_tracks(args.track_a, args.track_b)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

# EXAMPLES
"""
# Train model
python compare_tracks_xgboost.py --train --extended

# Validate with trained model
python compare_tracks_xgboost.py --validate --model model.pkl

# Compare two tracks
python compare_tracks_xgboost.py --track_a file1.wav --track_b file2.wav --model model.pkl
"""
