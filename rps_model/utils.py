import joblib
import os
import re
from PIL import Image
from rps_model.constants import IMG_SIZE


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
        img = img.resize(IMG_SIZE)
        img.save(os.path.join(path_to_image, image_name), 'JPEG')
        return print(f"Image is resized to {IMG_SIZE}")

    @staticmethod
    def multi_image_resize(path_to_folder):
        """
        Function to resize multiple images
        @param path_to_folder: Path to folder
        @return: Returns information if image was resized, is in good size, or was not found
        """
        try:
            image_list = os.listdir(path_to_folder)
        except FileNotFoundError:
            return print("There is no such directory! Please check path to folder.")
        if not image_list:
            return print("Folder is empty!")
        for img in image_list:
            edited_image = Image.open(os.path.join(path_to_folder, img))
            if edited_image.size == IMG_SIZE:
                print(f"Image {edited_image} is in good size {IMG_SIZE}")
            else:
                edited_image = edited_image.resize(IMG_SIZE)
                edited_image.save(os.path.join(path_to_folder, img))
                print(f"Image {edited_image} was resized to {IMG_SIZE}")
        return print("Resizing done!")


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
