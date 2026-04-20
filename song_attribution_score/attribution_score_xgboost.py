import librosa as lb
import numpy as np
from feature_extraction import FeatureExtractor
from compute_similarities import SimilarityMetrics
import pickle

class AttributionDetector:
    """Attribution detection using trained XGBoost model"""
    
    def __init__(self, model_path=None, use_extended=False, threshold=0.5):
        self.feature_extractor = FeatureExtractor(use_extended=use_extended)
        self.similarity_metrics = SimilarityMetrics()
        self.use_extended = use_extended
        self.threshold = threshold
        self.model = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained XGBoost model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            
            # Auto-detect if model was trained with extended features
            if 'use_extended' in data:
                self.use_extended = data['use_extended']
                self.feature_extractor = FeatureExtractor(use_extended=self.use_extended)
            
        print(f"Loaded model from: {model_path}")
    
    def compare_tracks(self, track_a_path, track_b_path):
        """Compare two tracks using trained model"""
        
        print(f"Processing track A: {track_a_path}")
        features_a = self.feature_extractor.extract_from_file(track_a_path)
        print(f"Processing track B: {track_b_path}")
        features_b = self.feature_extractor.extract_from_file(track_b_path)
        
        # Compute similarity metrics
        similarities = self.similarity_metrics.compute_all_similarities(
            features_a, features_b, use_extended=self.use_extended
        )
        
        # Prepare feature vector for model
        if self.model is None:
            # Fallback: calculate average if no model loaded
            overall_score = np.mean(list(similarities.values()))
        else:
            feature_vector = np.array([[similarities[k] for k in self.feature_names]])
            overall_score = self.model.predict_proba(feature_vector)[0, 1]
        
        is_attributed = overall_score >= self.threshold
        
        result = {
            'track_a': str(track_a_path),
            'track_b': str(track_b_path),
            'overall_similarity': float(overall_score),
            'detailed_scores': {k: float(v) for k, v in similarities.items()},
            'is_attributed': is_attributed,
            'threshold': self.threshold
        }
        
        print(f"Similarity Score (XGBoost): {overall_score:.3f}")
        print(f"Attribution: {'YES' if is_attributed else 'NO'}")
        
        return result