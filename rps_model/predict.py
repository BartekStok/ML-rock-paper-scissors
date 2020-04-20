import joblib
from PIL import Image
import os
import numpy as np

# Load model from file
model = joblib.load('model.joblib')

# Set path and image name
path = '../../../Obrazy/zdj_reka/'
image_name = '20.jpg'

# Open and preprocess image
im = Image.open(os.path.join(path, image_name))
im_np = np.asarray(im)
im_np = im_np.reshape(1, -1)

# Show image and print its prediction
im.show()
print(model.predict(im_np))

