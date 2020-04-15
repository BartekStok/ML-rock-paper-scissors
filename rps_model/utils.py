import joblib
import os
from PIL import Image
import re


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


url_rock = '../../../Obrazy/rock/'
url_paper = '../../../Obrazy/paper/'
url_scissors = '../../../Obrazy/scissors/'
rock_image_list = os.listdir(url_rock)
paper_image_list = os.listdir(url_paper)
scissors_image_list = os.listdir(url_scissors)
image_convert(rock_image_list, url_rock)
image_convert(paper_image_list, url_paper)
image_convert(scissors_image_list, url_scissors)
