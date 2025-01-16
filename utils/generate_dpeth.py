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


video_dir = os.path.join('/data/lihaochen/datasets/TikTok_dataset/')
video_list = sorted(os.listdir(video_dir))

loop = tqdm.tqdm(video_list, total=len(video_list))
for video in loop:
    if not os.path.exists(os.path.join(video_dir, video, 'inpainted')):
        continue
    if os.path.exists(os.path.join(video_dir, video, 'img_depth')) and os.path.exists(os.path.join(video_dir, video, 'bg_depth.npy')):
        continue

    loop.set_description(f'video: {video}')
    # if w * h > 6000000:
    #     continue

    if not os.path.exists(os.path.join(video_dir, video, 'img_depth')):
        os.makedirs(os.path.join(video_dir, video, 'img_depth'))

    img_list = sorted(os.listdir(os.path.join(video_dir, video, 'images')))
    mask_list = sorted(os.listdir(os.path.join(video_dir, video, 'masks')))
    inner_loop = tqdm.tqdm(range(len(img_list)), total=len(img_list))
    for i in inner_loop:
        img_file = img_list[i]
        mask_file = mask_list[i]
        inner_loop.set_description(f'img: {img_file}')

        img_path = os.path.join(video_dir, video, 'images', img_file)
        mask_path = os.path.join(video_dir, video, 'masks', mask_file)
        mask = Image.open(mask_path)
        mask = np.array(mask)
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
        np.save(os.path.join(video_dir, video, 'img_depth', img_file.replace('.png', '.npy')), depth)

    bg_path = os.path.join(video_dir, video, 'refined.png')

    # estimate bg depth
    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(bg_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]
    focallength_px = prediction["focallength_px"]
    depth = depth.squeeze().cpu().numpy()
    np.save(os.path.join(video_dir, video, 'bg_depth.npy'), depth)