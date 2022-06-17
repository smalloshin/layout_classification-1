import streamlit as st
from glob import glob
from pathlib import Path
import requests
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

@st.cache
def load_model(model_path=None):
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    return model

with st.sidebar.expander("Input Data", expanded=True):
    types = ('samples/*.png', 'samples/*.jpeg')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))

    file_grabbbed = st.selectbox('Select Input Data: ', files_grabbed)
    st.write(f'File Selected:')
    st.write(file_grabbbed)

## Title.
st.title('Layout Detection')

if file_grabbbed is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(file_grabbbed)

# Display image.
st.image(image, use_column_width=True)

model = load_model()
layout = model.detect(image)

image2 = lp.draw_box(image, layout, box_width=3)
st.image(image2, caption='Sunrise by the mountains')

