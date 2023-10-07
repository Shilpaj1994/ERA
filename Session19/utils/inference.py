import numpy as np
import gradio as gr
from PIL import ImageDraw

from utils.tools_gradio import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt


def segment_everything(
        model,
        device,
        input,
        input_size=1024,
        iou_threshold=0.7,
        conf_threshold=0.25,
        better_quality=False,
        withContours=True,
        use_retina=True,
        text="",
        wider=False,
        mask_random_color=True,
):
    input_size = int(input_size)
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size, )

    if len(text) > 0:
        results = format_results(results[0], 0)
        annotations, _ = text_prompt(results, text, input, device=device, wider=wider)
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours, )
    return fig


def segment_with_points(
        model,
        device,
        input,
        input_size=1024,
        iou_threshold=0.7,
        conf_threshold=0.25,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=True,
):
    global global_points
    global global_point_label

    input_size = int(input_size)
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size, )

    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours, )

    global_points = []
    global_point_label = []
    return fig, None


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == 'Add Mask' else 0)

    print(x, y, label == 'Add Mask')

    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image
