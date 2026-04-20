from audio_utils import preprocess_audio, segment_audio
import librosa as lb
import numpy as np

class FeatureExtractor:
    """Extract audio features"""    
    def __init__(self, sr=22050, n_mfcc=13, use_extended=False):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.use_extended = use_extended
    
    def compute_phase_continuity(self, y):
        D = lb.stft(y)
        phase = np.angle(D)
        phase_diff = np.diff(np.unwrap(phase, axis=1), axis=1)
        continuity = 1.0 / (1.0 + np.std(phase_diff))
        return continuity
    
    def compute_hnr(self, y):
        y_harmonic, y_percussive = lb.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        
        if percussive_energy < 1e-10:
            return 100.0
        
        hnr = 10 * np.log10(harmonic_energy / percussive_energy)
        return hnr
    
    def extract_segment_features(self, y):
        """Extract features from a segment"""
        features = {}
        
        # MFCCs (timbre)
        mfccs = lb.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        features['mfcc'] = mfccs
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        
        # Chroma (harmony)
        chroma = lb.feature.chroma_stft(y=y, sr=self.sr)
        features['chroma'] = chroma
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        # Spectral features
        features['spectral_centroid_mean'] = np.mean(lb.feature.spectral_centroid(y=y, sr=self.sr))
        features['spectral_rolloff_mean'] = np.mean(lb.feature.spectral_rolloff(y=y, sr=self.sr))
        features['spectral_bandwidth_mean'] = np.mean(lb.feature.spectral_bandwidth(y=y, sr=self.sr))
        # features['zcr_mean'] = np.mean(lb.feature.zero_crossing_rate(y))
        
        # Tempo
        try:
            tempo, _ = lb.beat.beat_track(y=y, sr=self.sr)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0
        
        # Extended features
        if self.use_extended:
            try:
                features['phase_continuity'] = self.compute_phase_continuity(y)
                features['hnr'] = self.compute_hnr(y)
                onset_env = lb.onset.onset_strength(y=y, sr=self.sr)
                features['spectral_flux_mean'] = np.mean(onset_env)
            except:
                features['phase_continuity'] = 0.5
                features['hnr'] = 0.0
                features['spectral_flux_mean'] = 0.0
        
        return features
    
    def aggregate_segment_features(self, segment_features_list):
        """Aggregate features from multiple segments"""
        aggregated = {}
        
        mfcc_all = np.concatenate([f['mfcc'] for f in segment_features_list], axis=1)
        aggregated['mfcc_global_mean'] = np.mean(mfcc_all, axis=1)
        
        chroma_all = np.concatenate([f['chroma'] for f in segment_features_list], axis=1)
        aggregated['chroma_global_mean'] = np.mean(chroma_all, axis=1)
        
        aggregated['spectral_centroid'] = np.mean([f['spectral_centroid_mean'] for f in segment_features_list])
        aggregated['spectral_rolloff'] = np.mean([f['spectral_rolloff_mean'] for f in segment_features_list])
        aggregated['spectral_bandwidth'] = np.mean([f['spectral_bandwidth_mean'] for f in segment_features_list])
        # aggregated['zcr'] = np.mean([f['zcr_mean'] for f in segment_features_list])
        aggregated['tempo'] = np.mean([f['tempo'] for f in segment_features_list])
        
        if 'phase_continuity' in segment_features_list[0]:
            aggregated['phase_continuity'] = np.mean([f['phase_continuity'] for f in segment_features_list])
            aggregated['hnr'] = np.mean([f['hnr'] for f in segment_features_list])
            aggregated['spectral_flux_mean'] = np.mean([f['spectral_flux_mean'] for f in segment_features_list])
        
        aggregated['segment_features'] = segment_features_list
        
        return aggregated
    
    def extract_from_file(self, file_path):
        """Extract features from audio file"""
        y, sr = preprocess_audio(file_path, sr=self.sr)
        segments = segment_audio(y, sr)
        segment_features = [self.extract_segment_features(seg) for seg in segments]
        return self.aggregate_segment_features(segment_features)
