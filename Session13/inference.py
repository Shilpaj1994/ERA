"""
Script to perform the inference
Reference: https://huggingface.co/spaces/anantgupta129/PyTorch-YoloV3-PascolVOC-GradCAM/tree/main
"""
import random
from typing import List

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from utils import cells_to_bboxes, non_max_suppression


IMAGE_SIZE = config.IMAGE_SIZE
scaled_anchors = config.SCALED_ANCHORS

_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)


def draw_predictions(image: np.ndarray, boxes: List[List], class_labels: List[str]) -> np.ndarray:
    """Plots predicted bounding boxes on the image"""

    colors = [[random.randint(0, 255) for _ in range(3)] for name in class_labels]

    im = np.array(image)
    height, width, _ = im.shape
    bbox_thick = int(0.6 * (height + width) / 600)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        x1 = int(upper_left_x * width)
        y1 = int(upper_left_y * height)

        x2 = x1 + int(box[2] * width)
        y2 = y1 + int(box[3] * height)

        cv2.rectangle(
            image,
            (x1, y1), (x2, y2),
            color=colors[int(class_pred)],
            thickness=bbox_thick
        )
        text = f"{class_labels[int(class_pred)]}: {conf:.2f}"
        t_size = cv2.getTextSize(text, 0, 0.7, thickness=bbox_thick // 2)[0]
        c3 = (x1 + t_size[0], y1 - t_size[1] - 3)

        cv2.rectangle(image, (x1, y1), c3, colors[int(class_pred)], -1)
        cv2.putText(
            image,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            bbox_thick // 2,
            lineType=cv2.LINE_AA,
        )

    return image


class YoloCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(YoloCAM, self).__init__(model,
                                      target_layers,
                                      use_cuda,
                                      reshape_transform,
                                      uses_gradients=False)

    def forward(self,
                input_tensor: torch.Tensor,
                scaled_anchors: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            bboxes = [[] for _ in range(1)]
            for i in range(3):
                batch_size, A, S, _, _ = outputs[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(
                    outputs[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            nms_boxes = non_max_suppression(
                bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint",
            )
            # target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = [box[0] for box in nms_boxes]
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)


@torch.inference_mode()
def predict(cam,
            model,
            image: np.ndarray,
            iou_thresh: float = 0.5,
            thresh: float = 0.4,
            show_cam: bool = False,
            transparency: float = 0.5,
            ) -> List[np.ndarray]:
    transformed_image = _transforms(image=image)["image"].unsqueeze(0)
    output = model(transformed_image)

    bboxes = [[] for _ in range(1)]
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            output[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
    )
    plot_img = draw_predictions(image.copy(), nms_boxes, class_labels=config.PASCAL_CLASSES)
    if not show_cam:
        return [plot_img]

    grayscale_cam = cam(transformed_image, scaled_anchors)[0, :, :]
    img = cv2.resize(image, (416, 416))
    img = np.float32(img) / 255
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=transparency)
    return [plot_img, cam_image]

