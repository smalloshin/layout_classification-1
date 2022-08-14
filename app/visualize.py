import layoutparser as lp


def draw_detection(image, layout, box_width=5):
    image = lp.draw_box(image, layout, box_width=box_width)
    return image
