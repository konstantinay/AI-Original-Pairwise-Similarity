import librosa as lb
import numpy as np
from scipy.spatial.distance import cosine

class SimilarityMetrics:
    
    @staticmethod
    def mfcc_similarity(features_a, features_b):
        mfcc_a = features_a['mfcc_global_mean']
        mfcc_b = features_b['mfcc_global_mean']
        similarity = 1 - cosine(mfcc_a, mfcc_b)
        mfcc_sim = max(0, similarity)
        return mfcc_sim
    
    @staticmethod
    def chroma_similarity(features_a, features_b):
        chroma_a = features_a['chroma_global_mean']
        chroma_b = features_b['chroma_global_mean']
        
        max_sim = 0
        for shift in range(12):
            chroma_b_shifted = np.roll(chroma_b, shift)
            sim = 1 - cosine(chroma_a, chroma_b_shifted)
            max_sim = max(max_sim, sim)
            
        chroma_sim = max(0, max_sim)            
        return chroma_sim
    
    @staticmethod
    def spectral_similarity(features_a, features_b):
        spectral_features = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']
        
        similarities = []
        for feat in spectral_features:
            val_a = features_a.get(feat, 0)
            val_b = features_b.get(feat, 0)
            
            if val_a == 0 or val_b == 0:
                continue
            
            diff = abs(val_a - val_b) / max(val_a, val_b)
            sim = 1 - min(diff, 1.0)
            similarities.append(sim)

        spectral_sim = np.mean(similarities) if similarities else 0.5
        return spectral_sim
    
    @staticmethod
    def tempo_similarity(features_a, features_b):
        tempo_a = features_a.get('tempo', 0)
        tempo_b = features_b.get('tempo', 0)
        
        if tempo_a == 0 or tempo_b == 0:
            return 0.5
        
        ratios = [
            abs(tempo_a - tempo_b) / max(tempo_a, tempo_b),
            abs(tempo_a - 2 * tempo_b) / max(tempo_a, 2 * tempo_b),
            abs(2 * tempo_a - tempo_b) / max(2 * tempo_a, tempo_b),
        ]
        
        best_match = 1 - min(ratios)
        tempo_sim = max(0, best_match)
        return tempo_sim
    
    @staticmethod
    def dtw_similarity(features_a, features_b, scale=5.0, subseq=False):
        if 'segment_features' not in features_a or 'segment_features' not in features_b:
            return 0.5

        segments_a = features_a['segment_features']
        segments_b = features_b['segment_features']

        if not segments_a or not segments_b:
            return 0.5

        # Build feature sequences: shape -> (n_mfcc, n_segments)
        seq_a = np.array([seg['mfcc_mean'] for seg in segments_a], dtype=np.float32).T
        seq_b = np.array([seg['mfcc_mean'] for seg in segments_b], dtype=np.float32).T

        if seq_a.ndim != 2 or seq_b.ndim != 2 or seq_a.shape[1] == 0 or seq_b.shape[1] == 0:
            return 0.5

        try:
            # D accumulated cost matrix, wp warping path
            D, wp = lb.sequence.dtw(X=seq_a, Y=seq_b, metric='euclidean', subseq=subseq)

            # Final DTW distance
            if subseq:
                dtw_distance = np.min(D[-1, :])
            else:
                dtw_distance = D[-1, -1]

            # Normalize by warping path length
            path_len = max(1, len(wp))
            normalized_distance = dtw_distance / path_len

            # Convert distance to similarity
            similarity = np.exp(-normalized_distance / scale)
            dtw_sim = float(np.clip(similarity, 0.0, 1.0))
            return dtw_sim
        except Exception:
            return 0.5
    
    @staticmethod
    def phase_continuity_similarity(features_a, features_b):
        phase_a = features_a.get('phase_continuity', 0.5)
        phase_b = features_b.get('phase_continuity', 0.5)
        
        if phase_a == 0 or phase_b == 0:
            return 0.5
        
        diff = abs(phase_a - phase_b) / max(phase_a, phase_b)
        phase_cont_sim = max(0, 1 - min(diff, 1.0))
        return phase_cont_sim
    
    @staticmethod
    def hnr_similarity(features_a, features_b):
        hnr_a = features_a.get('hnr', 0)
        hnr_b = features_b.get('hnr', 0)
        
        diff = abs(hnr_a - hnr_b)
        sim = 1 - min(diff / 40.0, 1.0)
        hnr_sim = max(0, sim)
        return hnr_sim
    
    @staticmethod
    def spectral_flux_similarity(features_a, features_b):
        flux_a = features_a.get('spectral_flux_mean', 0)
        flux_b = features_b.get('spectral_flux_mean', 0)
        
        if flux_a == 0 or flux_b == 0:
            return 0.5
        
        diff = abs(flux_a - flux_b) / max(flux_a, flux_b)
        spectral_flux_sim = max(0, 1 - min(diff, 1.0))
        return spectral_flux_sim
    
    def compute_all_similarities(self, features_a, features_b, use_extended=False):
        """Compute all similarity metrics as a feature vector"""
        similarities = {
            'mfcc': self.mfcc_similarity(features_a, features_b),
            'chroma': self.chroma_similarity(features_a, features_b),
            'spectral': self.spectral_similarity(features_a, features_b),
            'tempo': self.tempo_similarity(features_a, features_b),
            'dtw': self.dtw_similarity(features_a, features_b),
        }
        
        if use_extended:
            similarities['phase_continuity'] = self.phase_continuity_similarity(features_a, features_b)
            similarities['hnr'] = self.hnr_similarity(features_a, features_b)
            similarities['spectral_flux'] = self.spectral_flux_similarity(features_a, features_b)
        
        return similarities
    
    def compute_overall_similarity(self, features_a, features_b, weights=None, use_extended=False):
        """Compute weighted similarity score"""
        
        # Check if extended features are available
        has_extended = 'phase_continuity' in features_a and 'phase_continuity' in features_b
        use_extended = use_extended and has_extended
        
        if weights is None:
            if use_extended:
                # Weights with extended features
                weights = {
                    'mfcc': 0.20,
                    'chroma': 0.20,
                    'spectral': 0.10,
                    'tempo': 0.08,
                    'dtw': 0.20,
                    'phase_continuity': 0.10,
                    'hnr': 0.07,
                    'spectral_flux': 0.05
                }
            else:
                # Standard weights
                weights = {
                    'mfcc': 0.25,
                    'chroma': 0.25,
                    'spectral': 0.15,
                    'tempo': 0.10,
                    'dtw': 0.25
                }
        
        # Compute base similarities
        similarities = {
            'mfcc': self.mfcc_similarity(features_a, features_b),
            'chroma': self.chroma_similarity(features_a, features_b),
            'spectral': self.spectral_similarity(features_a, features_b),
            'tempo': self.tempo_similarity(features_a, features_b),
            'dtw': self.dtw_similarity(features_a, features_b),
        }
        
        # Add extended similarities if enabled
        if use_extended:
            similarities['phase_continuity'] = self.phase_continuity_similarity(features_a, features_b)
            similarities['hnr'] = self.hnr_similarity(features_a, features_b)
            similarities['spectral_flux'] = self.spectral_flux_similarity(features_a, features_b)
        
        # Weighted average (only use metrics that are in weights)
        overall = sum(similarities[k] * weights[k] for k in weights.keys() if k in similarities)
        
        return overall, similarities