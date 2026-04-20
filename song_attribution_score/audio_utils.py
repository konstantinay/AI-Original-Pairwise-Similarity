import librosa

def preprocess_audio(file_path, sr=22050, trim_silence=True):
    """Load and preprocess audio file"""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
    
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
    
    return y, sr


def segment_audio(y, sr, segment_duration=15, overlap=0.5):
    """Split audio into overlapping segments"""
    segment_samples = int(segment_duration * sr)
    hop_samples = max(1, int(segment_samples * (1 - overlap)))
    
    segments = []    
    for start in range(0, len(y), hop_samples):

        end = start + segment_samples
        segment = y[start:end]
        
        segments.append(segment)
        
        # stop if we reached the end
        if end >= len(y):
            break
    
    return segments