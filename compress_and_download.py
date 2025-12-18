import os
import zipfile
from glob import glob
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 源文件夹路径
# base_dir = "/data1/lihaochen/Relight/video"
base_dir = "/data/lihaochen"
folders_avg = ["val_depthattnmask"]
# folders_avg = ["val_attnmask"]
folders_unique = ["val_harmonizer", "val_relightvid"]

# 输出目录
output_dir = str(Path.home())

def average_and_save_images(folder, use_average=True):
    folder_path = os.path.join(base_dir, folder)
    temp_dir = os.path.join(output_dir, f"{folder}_avg_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    # 分组
    groups = defaultdict(list)
    for img_path in glob(os.path.join(folder_path, "*.png")):
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[1]}"
        groups[key].append(img_path)

    def process_group(item):
        key, paths = item
        if use_average:
            # 平均处理
            imgs = [np.array(Image.open(p)).astype(np.float32) for p in paths]
            avg_img = np.mean(imgs, axis=0).astype(np.uint8)
            out_path = os.path.join(temp_dir, f"{key}.png")
            Image.fromarray(avg_img).save(out_path)
        else:
            # 选择特定repeat_num的图片
            parts = key.split("_")
            frame_id = int(parts[1]) + 3
            target_repeat = frame_id % 4
            target_filename = f"{key}_{target_repeat}.png"
            target_path = None
            for path in paths:
                if os.path.basename(path) == target_filename:
                    target_path = path
                    break
            if target_path:
                out_path = os.path.join(temp_dir, f"{key}.png")
                Image.open(target_path).save(out_path)
        return key

    # 多线程处理
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_group, item): item[0] for item in groups.items()}
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {folder}"):
            pass

    # 打包
    zip_path = os.path.join(output_dir, f"{folder}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_file in glob(os.path.join(temp_dir, "*.png")):
            arcname = os.path.join(folder, os.path.basename(img_file))
            zipf.write(img_file, arcname)
    print(f"打包完成: {zip_path}")
    # 清理临时文件
    for img_file in glob(os.path.join(temp_dir, "*.png")):
        os.remove(img_file)
    os.rmdir(temp_dir)

def unique_and_rename_images(folder):
    folder_path = os.path.join(base_dir, folder)
    temp_dir = os.path.join(output_dir, f"{folder}_unique_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    seen = set()
    img_paths = glob(os.path.join(folder_path, "*.png"))
    key_to_path = {}
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[1]}"
        if key not in seen:
            seen.add(key)
            key_to_path[key] = img_path

    def process_rename(item):
        key, img_path = item
        out_path = os.path.join(temp_dir, f"{key}.png")
        Image.open(img_path).save(out_path)
        return key

    # 多线程处理
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_rename, item): item[0] for item in key_to_path.items()}
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"重命名 {folder}"):
            pass

    # 打包
    zip_path = os.path.join(output_dir, f"{folder}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_file in glob(os.path.join(temp_dir, "*.png")):
            arcname = os.path.join(folder, os.path.basename(img_file))
            zipf.write(img_file, arcname)
    print(f"打包完成: {zip_path}")
    # 清理临时文件
    for img_file in glob(os.path.join(temp_dir, "*.png")):
        os.remove(img_file)
    os.rmdir(temp_dir)

if __name__ == "__main__":
    for folder in folders_avg:
        average_and_save_images(folder, False)
    # for folder in folders_unique:
        # unique_and_rename_images(folder)
    print("全部打包完成！")