import torch
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops  # for postprocess
from pathlib import Path
import cv2
import numpy as np


# .pt files contains names in there but exported onnx/tflite don't have them.
yolo_default_label_names = {0: 'tennis-ball', 1: 'player', 2: 'n/a', 3: 'n/a'}


class YoloDetectorTFLite:
    def __init__(self, model_path):
        self.name = model_path.name

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        self.interpreter.allocate_tensors()

    def predict(self, frame, conf) -> list[Results]:
        orig_imgs = [frame]

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']

        # TODO check shape of input_shape and frame.shape
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        _, w, h, _ = input_shape

        # check width and height
        if frame.shape[0] != h or frame.shape[1] != w:
            input_img = cv2.resize(frame, (w, h))
        else:
            input_img = frame
        input_img = input_img[np.newaxis, ...]  # add batch dim

        input_img = input_img.astype(np.float32) / 255.  # change to float img
        self.interpreter.set_tensor(input_details[0]['index'], input_img)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(output_details[0]['index'])  
        
        ######################################################################
        # borrowed from ultralytics\models\yolo\detect\predict.py #postprocess

        # convert to torch to use ops.non_max_suppression
        # ultralytics is working on none-deeplearning based non_max_suppression
        # https://github.com/ultralytics/ultralytics/issues/1777
        # maybe someday, but for now, just workaround
        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(preds,
                                        conf,
                                        0.7,  # todo, make into arg
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)  # hack. just copied values from execution of yolov8n.pt

        results = []
        for i, pred in enumerate(preds):
            if pred.ndim == 1:
                pred = pred.reshape(-1, 6)
            if pred.ndim == 2 and pred.shape[1] != 6:
                pred = pred[:, :6]
            orig_img = orig_imgs[i]

            # tflite result are in [0, 1]
            # scale them by width (w == h)
            pred[:, :4] *= w

            pred[:, :4] = ops.scale_boxes(input_img.shape[1:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(Results(orig_img, path=img_path, names=yolo_default_label_names, boxes=pred))

        return results

    def get_label_names(self):
        return yolo_default_label_names


class YoloDetectorWrapper:
    def __init__(self, model_path):
        model_path = Path(model_path)

        if model_path.suffix == '.tflite':
            self.detector = YoloDetectorTFLite(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

    def predict(self, frame, conf=0.5):
        return self.detector.predict(frame, conf=conf)

    def get_label_names(self):
        return self.detector.get_label_names()