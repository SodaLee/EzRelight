import os
import torch
import tqdm
from PIL import Image
import numpy as np
import depth_pro

# Load model and preprocessing transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

'''
# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
'''

log_file = '/data3/lihaochen/datasets/BGRelight/depth.log'
f = open(log_file, 'w+')

for i in range(5):
    img_dir = os.path.join('/data3/lihaochen/datasets/BGRelight/imgs', f'{i:05d}')
    img_list = sorted(os.listdir(img_dir))

    t = len(img_list)
    loop = tqdm.tqdm(enumerate(img_list), total=len(img_list))
    for j, img_file in loop:
        if not os.path.exists(os.path.join(img_dir, img_file, 'relight_0.png')):
            f.write(f'Folder: {i:05d}, image: {img_file}, no relight_0.png skip\n')
            continue
        if os.path.exists(os.path.join(img_dir, img_file, 'img_depth_0.npy')) and os.path.exists(os.path.join(img_dir, img_file, 'bg_depth_0.npy')):
            continue

        img_path = os.path.join('/data3/lihaochen/datasets/CosmicManHQ-1.0/LAION-5B/laion1B-nolang', f'{i:05d}', img_file)
        if os.path.exists(img_path + '.webp'):
            img_path += '.webp'
        elif os.path.exists(img_path + '.jpg'):
            img_path += '.jpg'
        else:
            f.write(f'Folder: {i:05d}, image: {img_file}, no img file skip\n')
            continue

        bg_path = os.path.join(img_dir, img_file, 'refined_0.png')
        mask = Image.open(os.path.join(img_dir, img_file, 'mask_0.png'))
        w, h = mask.size
        mask = np.array(mask)

        loop.set_description(f'Folder: {i:05d}, image: {img_file}, width: {w}, height: {h}')
        if w * h > 6000000:
            f.write(f'Folder: {i:05d}, image: {img_file}, width: {w}, height: {h}, too large\n')
            f.flush()
            continue

        # estimate image depth
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(img_path)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        depth = depth.squeeze().cpu().numpy()
        depth = np.where(mask > 0, depth, 0)
        np.save(os.path.join(img_dir, img_file, 'img_depth_0.npy'), depth)

        # estimate bg depth
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(bg_path)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        focallength_px = prediction["focallength_px"]
        depth = depth.squeeze().cpu().numpy()
        np.save(os.path.join(img_dir, img_file, 'bg_depth_0.npy'), depth)