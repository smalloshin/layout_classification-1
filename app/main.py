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
from pdf2image import convert_from_path, convert_from_bytes

primaLayout = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion",
               5:"SeparatorRegion", 6:"OtherRegion"}
models = {'magazine':lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x")}

st.set_page_config(layout="wide")

@st.experimental_memo
def load_model(model_path=None):
    #model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    #                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    #                                 label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    model = models['magazine']
    return model


with st.sidebar:
    uploaded_file = st.file_uploader("Select the PDF", type="pdf")
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())

        for ids, page in enumerate(pages):
            page.save(f'pdf_extract/{ids}-out.png', 'PNG')


with st.sidebar.expander("Input Data", expanded=True):
    types = ('samples/*.png', 'samples/*.jpeg', 'pdf_extract/*.png')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))

    file_grabbbed = st.selectbox('Select Input Data: ', files_grabbed)
    st.write(f'File Selected:')
    st.write(file_grabbbed)

with st.sidebar.expander('Select Model', expanded=True):
    st.selectbox('Models: ', ['Magazine', 'Newspaper', 'Academic Papers'])




## Title.
st.title('Layout Detection')

if file_grabbbed is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(file_grabbbed)
    image = cv2.imread(file_grabbbed)
    image = image[..., ::-1]

# Display image.

original, detected = st.columns(2)
with original:

    st.image(image, caption='Original Image', use_column_width=True)

model = load_model()

layout = model.detect(image)

text_blocks = lp.Layout([b for b in layout if b.type==1])
image_blocks = lp.Layout([b for b in layout if b.type==2])
table_blocks = lp.Layout([b for b in layout if b.type==3])
text_blocks = lp.Layout([b for b in text_blocks \
                         if not any(b.is_in(b_fig) for b_fig in image_blocks)])
ocr_agent = lp.TesseractAgent(languages='eng')


with detected:
    image2 = lp.draw_box(image, layout, box_width=5)
    st.image(image2, caption='All Detected Regions', use_column_width=True)


st.markdown('### Detected Texts and Figure Regions')

text_col, image_col = st.columns(2)

with text_col:
    text_image = lp.draw_box(image, text_blocks, box_width=3, show_element_id=True)
    st.image(text_image, caption='Detected Text Blocks', use_column_width=True)

with image_col:
    image2 = lp.draw_box(image, image_blocks, box_width=5)
    st.image(image2, caption='Detected Image Blocks', use_column_width=True)

text_col1, image_col1 = st.columns(2)

for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
        # add padding in each image segment can help
        # improve robustness

    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)

text_list = [txt for txt in text_blocks.get_texts()]


with text_col1:
    st.table(text_list)
with image_col1:
    st.image(text_image, caption='Text Captured Image', use_column_width=True)


table_image = lp.draw_box(image, table_blocks, box_width=5, show_element_id=True)
st.image(table_image, caption='Table Captured Image', use_column_width=True)
