import librosa as lb
import numpy as np
from feature_extraction import FeatureExtractor
from compute_similarities import SimilarityMetrics


class AttributionDetector:
    """Feature comparison Attribution Score computation"""
    
    def __init__(self, similarity_threshold=0.7, use_extended=False):
        self.feature_extractor = FeatureExtractor(use_extended=use_extended)
        self.similarity_metrics = SimilarityMetrics()
        self.threshold = similarity_threshold
        self.use_extended = use_extended
    
    def compare_tracks(self, track_a_path, track_b_path):
        """Compare two audio tracks and determine attribution"""
        # Extract features
        print(f"Processing track A: {track_a_path}")
        features_a = self.feature_extractor.extract_from_file(track_a_path)
        
        print(f"Processing track B: {track_b_path}")
        features_b = self.feature_extractor.extract_from_file(track_b_path)
        
        # Compute similarity
        overall_score, detailed_scores = self.similarity_metrics.compute_overall_similarity(
            features_a, features_b, use_extended=self.use_extended
        )
        
        # Determine attribution
        is_attributed = overall_score >= self.threshold
        
        result = {
            'track_a': str(track_a_path),
            'track_b': str(track_b_path),
            'overall_similarity': float(overall_score),
            'detailed_scores': {k: float(v) for k, v in detailed_scores.items()},
            'is_attributed': is_attributed,
            'threshold': self.threshold
        }
        
        print(f"Overall Similarity: {overall_score:.3f}")
        print(f"Attribution Detected: {'YES' if is_attributed else 'NO'}")
        print(f"Extended Features: {'ENABLED' if self.use_extended else 'DISABLED'}")
        print(f"\nDetailed Scores:")
        for metric, score in detailed_scores.items():
            print(f"  {metric:20s}: {score:.3f}")
        
        return result