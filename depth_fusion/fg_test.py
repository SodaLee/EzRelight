import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

#read the RBGA png file and convert it to a numpy array
def read_rgba(file):
    img = Image.open(file)
    img = img.convert("RGBA")
    img = np.array(img)
    return img

#read foreground and background images
def read_depth_images(foreground_rgba_file, background_file):
    foreground = read_rgba(foreground_rgba_file)
    background = Image.open(background_file)
    background = np.array(background)

    return foreground, background

if __name__ == "__main__":
    #read the foreground and background images
    foreground, background = read_depth_images("0.png", "bg.png")

    #Centering on the foreground, zero-padding it to match the background size.
    #This is a simple way to align the images.
    h, w = background.shape[:2]
    fh, fw = foreground.shape[:2]
    top = (h - fh) // 2
    bottom = h - fh - top
    left = (w - fw) // 2
    right = w - fw - left
    foreground = np.pad(foreground, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)

    #overlay the foreground on the background
    alpha = foreground[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    integrated = alpha * foreground[:, :, :3] + (1 - alpha) * background[:, :, :3]

    colorer_img = Image.fromarray(integrated.astype(np.uint8))
    colorer_img.save("cover.png")