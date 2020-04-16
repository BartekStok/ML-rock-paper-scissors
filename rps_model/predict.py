import joblib
from PIL import Image
import os
import numpy as np


model = joblib.load('model.joblib')

path = '../../../Obrazy/zdj_reka/'
image_name = '20.jpg'
im = Image.open(os.path.join(path, image_name))
im_np = np.asarray(im)
print(im_np.reshape(1, -1))
im_np = im_np.reshape(1, -1)
im.show()
print(model.predict(im_np))
