from rps_model.settings import PATHS
from rps_model.utils import ImageUtils

for p in PATHS:
    ImageUtils.multi_image_resize(p)
    ImageUtils.images_to_pkl(p)
