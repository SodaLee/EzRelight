import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

#read the RBGA png file and convert it to a numpy array
def read_rgba(file):
    img = Image.open(file)
    img = img.convert("RGBA")
    img = np.array(img)
    return img

#integrate the foreground depth image with the background depth image
#both images are numpy arrays
#foreground image is RGBA image
def integrate_depth(foreground_rgba, foreground_depth, background_depth):
    #get the alpha channel from the foreground image
    alpha = foreground_rgba[:, :, 3] / 255.0

    #detect the largest and smallest line index of non-zero alpha value
    min_line = 0
    max_line = foreground_rgba.shape[0] - 1
    for i in range(foreground_rgba.shape[0]):
        if alpha[i].max() > 0:
            min_line = i
            break
    for i in range(foreground_rgba.shape[0] - 1, -1, -1):
        if alpha[i].max() > 0:
            max_line = i
            break
    height = max_line - min_line + 1

    start_line = max_line - max(height // 100, 1)

    pixel_num = 0
    fg_avg = 0.0
    bg_avg = 0.0
    for i in range(max_line, max(0, start_line), -1):
        for j in range(foreground_rgba.shape[1]):
            if alpha[i, j] > 0:
                pixel_num += 1
                fg_avg += foreground_depth[i, j]
                bg_avg += background_depth[i, j]
    fg_avg /= pixel_num
    bg_avg /= pixel_num

    offset =  bg_avg - fg_avg    

    #integrate the depth images
    integrated_depth = alpha * (foreground_depth + offset) + (1 - alpha) * background_depth

    return integrated_depth

#get colored depth image
def get_colored_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    return colored_depth

#read foreground and background depth images
def read_depth_images(foreground_rgba_file, foreground_file, background_file):
    foreground_rgba = read_rgba(foreground_rgba_file)
    foreground = cv2.imread(foreground_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    background = cv2.imread(background_file, cv2.IMREAD_UNCHANGED).astype(np.float32)

    return foreground_rgba, foreground, background

if __name__ == "__main__":
    #read the foreground and background depth images
    foreground_rgba, foreground, background = read_depth_images("0.png", "0_depth.png", "bg_depth.png")

    #Centering on the foreground, zero-padding it to match the background size.
    #This is a simple way to align the images.
    h, w = background.shape
    fh, fw = foreground.shape
    top = (h - fh) // 2
    bottom = h - fh - top
    left = (w - fw) // 2
    right = w - fw - left
    foreground = np.pad(foreground, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    foreground_rgba = np.pad(foreground_rgba, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)

    #integrate the depth images
    integrated_depth = integrate_depth(foreground_rgba, foreground, background)

    #get colored depth image
    colored_depth = get_colored_depth(integrated_depth)

    colorer_img = Image.fromarray(colored_depth)
    colorer_img.save("integrated_depth.png")