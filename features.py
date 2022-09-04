from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class Features:

  def __init__(self):
        vgg_model = VGG16(weights='imagenet')
        self.model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc1').output)

  def extract_features(image):
    image = image.resize((224,224)) 
    image = image.convert("RGB")
    img = Image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    feature = model.predict(img)[0]
    return feature / np.linalg.norm(feature)

