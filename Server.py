import numpy as np
from PIL import Image
from features import Features
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
feat = Features()
features = []
img_paths = []
for feature_path in Path("data/features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("data/images") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        query = Image.open(file.stream)  # PIL image
        uploaded_img_path = "data/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = feat.extract_features(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:10]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('web folder/index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('web folder/index.html')


if __name__=="__main__":
    app.run("0.0.0.0")