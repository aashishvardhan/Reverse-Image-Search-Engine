from PIL import Image
from features import Features
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    
    feat = Features()

    for img_path in sorted(Path("data/images").glob("*.jpg")):
        feature = feat.extract_features(img=Image.open(img_path))
        feature_path = Path("data/features") / (img_path.stem + ".npy") 
        np.save(feature_path, feature)