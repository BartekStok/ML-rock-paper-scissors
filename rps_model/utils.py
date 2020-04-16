import joblib
import os
from PIL import Image
import re
from rps_model.constants import PATHS, LABELS, IMG_SIZE

class ImageUtils:

    @staticmethod
    def single_image_resize(path_to_image, image_name):
        """
        Function to resize single Image

        @param path_to_image: Path to an Image
        @param image_name: Image name
        @return: Information if image was found, or has a good size, or was
                successfully resized
        """
        try:
            img = Image.open(os.path.join(path_to_image, image_name))
        except FileNotFoundError:
            return print("There is no such file! Please check path or image name.")
        if img.size == IMG_SIZE:
            return print(f"Image is already in good size {IMG_SIZE}")
        else:
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(path_to_image, image_name), 'JPEG')
            return print("Image is resized to 336x336")


def image_convert(image_names, path_to_images):
    """
    Converts images to a dict with appropriate label and size to np array
    Then saves it to pickle file
    """
    image_dict = {'label': [], 'data': [], 'size': []}
    label = re.findall(r'(rock|paper|scissors)', path_to_images)
    for image in image_names:
        f = Image.open(os.path.join(path_to_images, image))
        image_dict['label'].append(label[0])
        image_dict['data'].append(f)
        image_dict['size'].append(f.size)
    joblib.dump(image_dict, f'./data/{label[0]}.pkl')

# TODO function for image resizing
# TODO function for image to right format converting
# TODO class or functions?
# TODO maybe one class, two functions (converting, saving)


url_rock = '../../../Obrazy/rock/'
url_paper = '../../../Obrazy/paper/'
url_scissors = '../../../Obrazy/scissors/'
rock_image_list = os.listdir(url_rock)
paper_image_list = os.listdir(url_paper)
scissors_image_list = os.listdir(url_scissors)
image_convert(rock_image_list, url_rock)
image_convert(paper_image_list, url_paper)
image_convert(scissors_image_list, url_scissors)
