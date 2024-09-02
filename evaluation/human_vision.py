import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def generate_color_palette(num_classes):
    """
    生成包含 num_classes 种不同颜色的调色盘
    """
    np.random.seed(0)  # 固定随机种子以确保颜色一致
    palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    return palette


def human_vision(pic,path):

    p = pic.copy()
    pic = np.stack([p,p,p],axis=2)
    Pal = generate_color_palette(41)
    for id in range(41):
        color = Pal[id]
        pic[np.where(p==id)]=color
    img = Image.fromarray(pic)
    img.save(path)