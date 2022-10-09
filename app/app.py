from pinferencia import Server, task
from PIL import Image
from io import BytesIO
import base64
import json
from detect import parse_layout_api
from typing import List

import layoutparser as lp
import cv2
from detect import detect_text_list, detect_text, parse_layout


model = lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x",
                           label_map = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion",
                                        4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"},
                           extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85,
                                           "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.75])


label_mapping = {'TextRegion': 'text', 'ImageRegion': 'image', 'TableRegion': 'table',
                 'MathRegion': 'equation', 'SeparatorRegion': 'separator', 'OtherRegion': 'other'}


# base64 -> Image
# In test file, turn image to base64


def transform_and_detect(image_base64_str):
    '''

    :param image_base64_str: string of image in base64 format
    :return: parsed_layout: layout parsed into list
    '''
    input_img = Image.open(BytesIO(base64.b64decode(image_base64_str)))
    layout = model.detect(input_img)

    parsed_layout = parse_layout_api(layout, label_mapping)
    return parsed_layout


def predict(image_base64_str: str)-> str:
    '''

    :param image_base64_str: string of image in base64 format
    :return:  json_layout: layout response
    '''
    parsed_layout = transform_and_detect(image_base64_str)
    json_layout = json.dumps(parsed_layout, indent=2)

    return json_layout


service = Server()
service.register(
    model_name="layout",
    model=predict,
    metadata={"task": task.IMAGE_CLASSIFICATION,
              "display_name": 'Layout Detection'})
