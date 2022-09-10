import pandas as pd
import os
"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """
    # For this function, you must:
    #   1. Load labels CSV file
    #   2. Iterate over each row in the CSV, create the corresponding
    #      train/test and class folders
    #   3. Copy the image to the new folder structure. We recommend you to
    #      use `os.link()` to avoid wasting disk space with duplicated files
    # TODO
     #   1. Load labels CSV file and used pandas for read csv
    labels_data = pd.read_csv(labels)
    # https://www.geeksforgeeks.org/create-a-directory-in-python/?ref=lbp
    if os.path.exists(output_data_folder):
        pass
    else:
        os.mkdir(output_data_folder)
    file_train = os.path.join(output_data_folder, 'train')
    if os.path.exists(file_train):
        pass
    else:    
        os.mkdir(file_train)

    file_test = os.path.join(output_data_folder, 'test')
    if os.path.exists(file_test):
        pass
    else:
        os.mkdir(file_test)

    # https://remarkablemark.org/blog/2020/08/26/python-iterate-csv-rows/
    for index, row in labels_data.iterrows():
        if row['subset'] == 'train':
            if os.path.exists(os.path.join(file_train, row['class'])):
                pass
            else:
                os.mkdir(os.path.join(file_train, row['class']))    
            os.link(os.path.join(data_folder, row['img_name']), 
            os.path.join(file_train, row['class'], row['img_name']))
        
        else:
            if os.path.exists(os.path.join(file_test, row['class'])):
                pass
            else:
                os.mkdir(os.path.join(file_test, row['class']))
            os.link(os.path.join(data_folder, row['img_name']),  
            os.path.join(os.path.join(file_test, row['class']), row['img_name']))
            
if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
