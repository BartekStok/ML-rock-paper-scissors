import joblib
import os
import re
from PIL import Image
from rps_model.constants import IMG_SIZE, LABELS


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
                edited_image.save(os.path.join(path_to_folder, img), 'JPEG')
                print(f"Image {edited_image} was resized to {IMG_SIZE}")
        return print("Resizing done!")

    @staticmethod
    def images_to_pkl(path_to_folder):
        try:
            image_names = os.listdir(path_to_folder)
        except FileNotFoundError:
            return print("There is no such directory! Please check path to folder.")
        if not image_names:
            return print("Folder is empty!")
        image_dict = {'label': [], 'data': []}
        labels = re.compile("".join([i + "|" for i in LABELS]).rstrip("|"))
        label = re.findall(labels, path_to_folder)[0]
        for image in image_names:
            f = Image.open(os.path.join(path_to_folder, image))
            image_dict['label'].append(label)
            image_dict['data'].append(f)
        joblib.dump(image_dict, f'./data/{label[0]}.pkl')
