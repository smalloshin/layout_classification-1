import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import layoutparser as lp
import cv2
from pdf2image import convert_from_path, convert_from_bytes


primaLayout = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion",
               4:"MathsRegion",5:"SeparatorRegion", 6:"OtherRegion"}
models = {'magazine': lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x")}


def detect_text(ocr_agent, text_blocks, image):
    for block in text_blocks:
        segment_image = (block
                         .pad(left=5, right=5, top=5, bottom=5)
                         .crop_image(image))
        # add padding in each image segment can help
        # improve robustness

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)

    text_list = [txt for txt in text_blocks.get_texts()]

    return text_list
