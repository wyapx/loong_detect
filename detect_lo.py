import os
import sys
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Tuple, Any, List, Union, TypedDict

import cv2
import numpy
import torch

from .utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from .models.common import DetectMultiBackend
from .utils.general import (LOGGER, Profile, check_img_size,
                           non_max_suppression, scale_boxes)
from .utils.torch_utils import select_device, smart_inference_mode


class PredictedResult(TypedDict):
    x: float
    y: float
    width: float
    height: float
    confidence: float


def load_image(io: Union[str, BinaryIO]) -> Tuple[numpy.ndarray, Any]:
    if isinstance(io, str):
        rd = cv2.imread(io)
    else:
        s = numpy.asarray(bytearray(io.read()), dtype=numpy.uint8)
        rd = cv2.imdecode(s, cv2.IMREAD_COLOR)
    im = letterbox(rd, 640, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    return numpy.ascontiguousarray(im), rd  # contiguous


@smart_inference_mode()
def run_infer(
        sources: List[Union[str, BinaryIO]],  # file
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
) -> List[List[PredictedResult]]:
    dataset: List[Any] = []
    for d in sources:
        if isinstance(d, (str, Path)):
            dt = load_image(str(d))
        elif isinstance(d, BytesIO):
            dt = load_image(d)
        else:
            raise TypeError(d)
        dataset.append(dt)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, dt = 0, (Profile(), Profile(), Profile())

    result: List[List[PredictedResult]] = []
    for im, im0s in dataset:
        rs: List[PredictedResult] = []
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)

            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    print(n)
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    x, y, w, h = torch.tensor(xyxy).tolist()
                    rs.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": conf.tolist()
                    })
            result.append(rs)

            # Print time (inference-only)
            LOGGER.info(f"{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return result


if __name__ == '__main__':
    print(run_infer(["sample.jpg"], Path("best.pt")))
