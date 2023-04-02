import os

import cv2

HOME_ROOT = "/home2/phuc/itabqa"
DATA_ROOT = "/disks/strg16-176/VQAonBD2023/data"

DATA_VQA = f"{DATA_ROOT}/vqaondb2023"
DATA_PUBTAB = f"{DATA_ROOT}/TR_output"

# Visulization
FONT_PATH = os.path.join("PaddleOCR", "doc", "fonts", "latin.ttf")
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
COLOR_RED, COLOR_BLUE = (255, 0, 0), (0, 0, 255)
