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
import pandas as pd
import matplotlib.pyplot as plt
import layoutparser as lp
import cv2
from pdf2image import convert_from_path, convert_from_bytes
from detect import detect_text_list, detect_text

primaLayout = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion",
               5:"SeparatorRegion", 6:"OtherRegion"}

color_map = {
    'TextRegion':   'red',
    'ImageRegion':  'blue',
    'TableRegion':   'green',
    'MathsRegion':  'purple',
    'SeparatorRegion': 'pink',
    'OtherRegion': 'pink',
}

st.set_page_config(layout="wide")




with st.sidebar.expander("Upload PDF and detect the layout", expanded=False):
    uploaded_file = st.file_uploader("Select the PDF", type="pdf")
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())

        for ids, page in enumerate(pages):
            page.save(f'pdf_extract/{ids}-out.png', 'PNG')


with st.sidebar.expander("Load image files and detect the layout", expanded=True):
    types = ('samples/*.png', 'samples/*.jpeg', 'pdf_extract/*.png')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))

    file_grabbed = st.selectbox('Select Images: ', files_grabbed)
    st.write(f'File Selected:')
    st.write(file_grabbed)

with st.sidebar.expander('Select Models to perform layout detection', expanded=True):
    select_model = st.selectbox('Select Models: ', ['Magazine', 'Newspaper', 'AcademicPapers'])
    ocr_selected = st.checkbox('Activate OCR')
with st.sidebar.expander('Hyperparameter Selection, use with care'):
    score_thres = st.slider('Score Threshold', 0.5, 0.99, 0.85)
    nms_thres = st.slider('NMS Threshold', 0.5, 0.99, 0.75)


models = {'Magazine':lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x",
                                        label_map = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"},
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thres,
                                                      "MODEL.ROI_HEADS.NMS_THRESH_TEST", nms_thres]),
          'Newspaper':lp.AutoLayoutModel("lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config",
                                        label_map = {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon", 4: "Editorial Cartoon", 5: "Headline", 6: "Advertisement"},
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thres,
                                                      "MODEL.ROI_HEADS.NMS_THRESH_TEST", nms_thres]),
          'AcademicPapers':lp.AutoLayoutModel("lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
                                        label_map = {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thres,
                                                      "MODEL.ROI_HEADS.NMS_THRESH_TEST", nms_thres])}


@st.experimental_singleton
def load_model(select_model=None):
    if not select_model:
        model = models['Magazine']
    else:
        model = models[select_model]
    return model

## Title.
st.title('Layout Detection')

if file_grabbed is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)
    st.info("Input file Not available ")
else:
    # User-selected image.
    image = Image.open(file_grabbed)
    image = cv2.imread(file_grabbed)
    image = image[..., ::-1]

# Display image.

original, detected = st.columns(2)

with original:

    st.image(image, caption='Original Image', use_column_width=True)

model = load_model()

layout = model.detect(image)

text_blocks = lp.Layout([b for b in layout if b.type==primaLayout[1]])
image_blocks = lp.Layout([b for b in layout if b.type==primaLayout[2]])
table_blocks = lp.Layout([b for b in layout if b.type==primaLayout[3]])
text_blocks = lp.Layout([b for b in text_blocks \
                         if not any(b.is_in(b_fig) for b_fig in image_blocks)])
#detected_info = pd.DataFrame(vars(c) for c in layout)
#st.write(detected_info)
if ocr_selected:
    ocr_agent = lp.TesseractAgent(languages='eng')
    #text_list = detect_text_list(ocr_agent, layout, image)
layout_collections = list()

st.markdown('### Image Information')
st.metric(label="Image Size(Height, Width)", value=f'{image.shape[1]}*{image.shape[0]}')


for ob in layout:
    layout_dic = dict()
    layout_dic['id'] = ob.id
    layout_dic['detect_type'] = ob.type
    if ocr_selected:
        layout_dic['text'] = detect_text(ocr_agent, ob, image)
    layout_dic['parent'] = ob.parent
    layout_dic['rect_left'] = ob.block.coordinates[0]
    layout_dic['rect_right'] = ob.block.coordinates[1]

    layout_dic['detect_score'] = ob.score
    #layout_dic['']
    layout_collections.append(layout_dic)
detected_info = pd.DataFrame(layout_collections)
st.table(detected_info)

#for ob in layout:
#
#    st.write(ob)
with detected:
    #image2 = lp.draw_box(image, layout, box_width=5, color_map=color_map)
    image2 = lp.draw_box(image, [b.set(id=f'{b.type}') for b in layout],
              color_map=color_map,
              box_width=5,
              show_element_id=True, id_font_size=16,
              id_text_background_color='black',
              id_text_color='white')
    st.image(image2, caption='All Detected Regions', use_column_width=True)


st.markdown('### Detected Texts and Figure Regions Separately')

text_col, image_col = st.columns(2)

with text_col:
    st.metric(label="Number of Text Boxes Detected", value=len(text_blocks))
    text_image = lp.draw_box(image, text_blocks, box_width=3, show_element_id=True,  color_map=color_map)
    st.image(text_image, caption='Detected Text Blocks', use_column_width=True)

with image_col:
    st.metric(label="Number of Image Boxes Detected", value=len(image_blocks))
    image2 = lp.draw_box(image, image_blocks, box_width=5, show_element_id=True,  color_map=color_map)
    st.image(image2, caption='Detected Image Blocks', use_column_width=True)




