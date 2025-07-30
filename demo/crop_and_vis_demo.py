import os
import sys
import numpy as np
import cv2
from torchvision.transforms.v2.functional import crop
from PIL import Image

# TopCenterCrop实现
class TopCenterCrop:
    def __init__(self, resolution=1024):
        self.resolution = resolution

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError('只支持numpy.ndarray输入')
        h, w = img.shape[:2]
        res = self.resolution
        # 先等比缩放，短边到res
        scale = res / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if img.ndim == 2:
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 再做TopCenterCrop
        h, w = img_resized.shape[:2]
        if h > w:
            top = max(0, h // 4 - res // 2)
            left = max(0, (w - res) // 2)
        else:
            top = max(0, (h - res) // 2)
            left = max(0, (w - res) // 2)
        if img_resized.ndim == 2:
            cropped = img_resized[top:top+res, left:left+res]
        else:
            cropped = img_resized[top:top+res, left:left+res, :]
        return cropped

def vis_depth(depth):
    # 归一化到0-255，近白远黑
    d_min, d_max = np.nanmin(depth), np.nanmax(depth)
    norm = (depth - d_min) / (d_max - d_min + 1e-8)
    vis = (1.0 - norm) * 255
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    return vis

def main(image_id, resolution=1024):
    base_dir = '/data/lihaochen/datasets/BGRelight/imgs'
    found = False
    for i in range(5):
        dir_path = os.path.join(base_dir, f'0000{i}', image_id)
        if os.path.isdir(dir_path):
            found = True
            break
    if not found:
        print(f'未找到{image_id}文件夹')
        return
    fg_path = os.path.join(dir_path, 'relight.png')
    bg_path = os.path.join(dir_path, 'inpainted.png')
    depth_path = os.path.join(dir_path, 'fused_depth.npy')
    img_depth_path = os.path.join(dir_path, 'img_depth.npy')
    bg_depth_path = os.path.join(dir_path, 'bg_depth.npy')
    if not (os.path.exists(fg_path) and os.path.exists(bg_path) and os.path.exists(depth_path) and os.path.exists(img_depth_path) and os.path.exists(bg_depth_path)):
        print('缺少必要文件')
        return
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    depth = np.load(depth_path)
    img_depth = np.load(img_depth_path)
    bg_depth = np.load(bg_depth_path)
    cropper = TopCenterCrop(resolution)
    fg_crop = cropper(fg)
    bg_crop = cropper(bg)
    depth_crop = cropper(depth)
    img_depth_crop = cropper(img_depth)
    bg_depth_crop = cropper(bg_depth)
    depth_vis = vis_depth(depth_crop)
    img_depth_vis = vis_depth(img_depth_crop)
    bg_depth_vis = vis_depth(bg_depth_crop)
    save_dir = os.path.join(os.path.dirname(__file__))
    out_dir = os.path.join(save_dir, 'demo')
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f'{image_id}_relight_crop.png'), fg_crop)
    cv2.imwrite(os.path.join(out_dir, f'{image_id}_inpainted_crop.png'), bg_crop)
    cv2.imwrite(os.path.join(out_dir, f'{image_id}_depth_vis.png'), depth_vis)
    cv2.imwrite(os.path.join(out_dir, f'{image_id}_img_depth_vis.png'), img_depth_vis)
    cv2.imwrite(os.path.join(out_dir, f'{image_id}_bg_depth_vis.png'), bg_depth_vis)
    print('保存完成:', out_dir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python crop_and_vis_demo.py <image_id> [resolution]')
    else:
        image_id = sys.argv[1]
        resolution = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        main(image_id, resolution) 