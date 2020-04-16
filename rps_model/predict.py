# TODO model evaluating, prediction all here
from rps_model.utils import ImageUtils
from PIL import Image
import os
from rps_model.constants import IMG_SIZE

path = '../../../Obrazy/zdj_reka/'
image_name = 'img-20200414-wa0000.jpg'
# ImageUtils.single_image_resize(path_to_image=path, image_name=image_name)

im = Image.open(os.path.join(path, image_name))
im.resize(IMG_SIZE)
im.show()
