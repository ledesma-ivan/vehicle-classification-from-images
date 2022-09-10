import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL


# Config the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
    """
    # Load the image
    outputs = DET_MODEL(img)
    instances = outputs["instances"]
    class_indexes = instances.pred_classes
    pred_boxes = instances.pred_boxes
    height = instances.image_size[0]
    width = instances.image_size[1]
    cars_trucks = (class_indexes == 2) | (class_indexes == 7)

    if cars_trucks.any():
      box = pred_boxes[cars_trucks]
      max_area = int(box.area().argmax())

      coords = box[max_area].tensor.tolist()[0]
      x1 = int(coords[0])
      y1 = int(coords[1])
      x2 = int(coords[2])
      y2 = int(coords[3])
   
      box_coordinates = [x1, y1, x2, y2]
    else:
      box_coordinates = [0, 0, width, height]
    return box_coordinates
