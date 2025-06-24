# tyre_defect_detector.py

import cv2
import os
import numpy as np
from ultralytics import YOLO

class TyreDefectDetector:
    def __init__(self, model_path, patch_size=691, num_parts=5, conf=0.6, iou=0.7):
        self.model = YOLO(model_path)
        self.patch_size = patch_size
        self.num_parts = num_parts
        self.conf = conf
        self.iou = iou
        self.output_dir = "test_patches"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def split_image_vertically(self, image):
        H, W = image.shape[:2]
        part_width = W // self.num_parts
        parts = [image[:, i * part_width:(i + 1) * part_width if i < self.num_parts - 1 else W] for i in range(self.num_parts)]
        return parts

    def process_patch(self, patch):
        results = self.model.predict(patch, imgsz=self.patch_size, conf=self.conf, iou=self.iou, verbose=False)
        boxes = results[0].boxes
        names = self.model.names

        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0])
            label = names[cls_id] if names else str(cls_id)

            cv2.rectangle(patch, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(patch, f"{label}", (b[0], b[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return patch

    def reconstruct_part(self, part_img, part_index):
        part_h, part_w = part_img.shape[:2]
        canvas = np.zeros_like(part_img)
        patch_idx = 0

        for y in range(0, part_h, self.patch_size):
            patch = part_img[y:min(y + self.patch_size, part_h), :]

            if part_w > self.patch_size:
                x_offset = (part_w - self.patch_size) // 2
                patch = patch[:, x_offset:x_offset + self.patch_size]
            elif part_w < self.patch_size:
                continue

            annotated_patch = self.process_patch(patch)

            if part_w > self.patch_size:
                canvas[y:y + self.patch_size, x_offset:x_offset + self.patch_size] = annotated_patch
            else:
                canvas[y:y + self.patch_size, :] = annotated_patch

            cv2.imwrite(f"{self.output_dir}/part{part_index+1}_patch{patch_idx+1}.jpg", annotated_patch)
            patch_idx += 1

        return canvas

    def run_detection_pipeline(self, image_path, output_image="reconstructed_result.jpg"):
        img = self.load_image(image_path)
        parts = self.split_image_vertically(img)
        reconstructed_img = np.zeros_like(img)

        for i, part in enumerate(parts):
            reconstructed_part = self.reconstruct_part(part, i)
            x_start = i * (img.shape[1] // self.num_parts)
            x_end = x_start + reconstructed_part.shape[1]
            reconstructed_img[:, x_start:x_end] = reconstructed_part

        cv2.imwrite(output_image, reconstructed_img)
        return output_image
