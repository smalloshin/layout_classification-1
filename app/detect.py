import layoutparser as lp
import pandas as pd
import cv2


magazineLayout = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion",
                  4:"MathsRegion",5:"SeparatorRegion", 6:"OtherRegion"}
newspaperLayout = {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon",
                   4: "Editorial Cartoon", 5: "Headline", 6: "Advertisement"}
models = {'magazine': lp.AutoLayoutModel("lp://detectron2/PrimaLayout/mask_rcnn_R_50_FPN_3x")}


def detect_text(ocr_agent, text_block, image):
    segment_image = (text_block
                     .pad(left=5, right=5, top=5, bottom=5)
                     .crop_image(image))
    # add padding in each image segment can help
    # improve robustness

    text = ocr_agent.detect(segment_image)
    text_block.set(text=text, inplace=True)
    return text


def detect_text_list(ocr_agent, text_blocks, image):
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


def parse_layout_api(layout, label_mapping=False):
    layout_collections = list()
    for ob, index in zip(layout, range(len(layout))):
        layout_dic = dict()
        layout_dic['id'] = index
        if label_mapping:
            layout_dic['type'] = label_mapping[ob.type]
        else:
            layout_dic['type'] = ob.type

        layout_dic['rect_left'] = ob.block.coordinates[0]
        layout_dic['rect_right'] = ob.block.coordinates[1]

        layout_dic['confidence'] = ob.score
        layout_collections.append(layout_dic)
    return layout_collections


def parse_layout(layout, ocr_agent, image, ocr_selected=False):
    layout_collections = list()
    for ob in layout:
        layout_dic = dict()
        layout_dic['id'] = ob.id
        layout_dic['detect_type'] = ob.type
        if ocr_selected:
            layout_dic['text'] = detect_text(ocr_agent, ob, image)
        layout_dic['parent'] = ob.parent
        layout_dic['rect_left'] = ob.block.coordinates[0]
        layout_dic['rect_right'] = ob.block.coordinates[1]

        layout_dic['confidence'] = ob.score
        # layout_dic['']
        layout_collections.append(layout_dic)
    detected_info = pd.DataFrame(layout_collections)

    return detected_info
