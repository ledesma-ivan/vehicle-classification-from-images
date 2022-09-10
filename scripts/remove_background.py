"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os
import cv2
from utils import utils
from utils import detection



def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    #   2. Load the image
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.
    subsets = ["train", "test"]

    for subset in subsets:

        folder_input = os.path.join(data_folder, subset)
        folder_output = os.path.join(output_data_folder, subset)

        for img_ph, img_name in utils.walkdir(folder_input):
            load_img = cv2.imread(os.path.join(img_ph,img_name))
            vh_cord = detection.get_vehicle_coordinates(load_img)
            x1, y1, x2, y2 = vh_cord
            crop_img = load_img[int(y1):int(y2), int(x1):int(x2)]

            root, class_folder = os.path.split(root)

            folder_img_crop = os.path.join(folder_output, class_folder)
            if not os.path.exists(folder_img_crop):
                os.makedirs(folder_img_crop)
            
            path_img_crop = os.path.join(folder_img_crop, img_name)
            cv2.imwrite(path_img_crop, crop_img)




if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
