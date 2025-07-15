import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
import cv2
import gc
import numpy as np
import argparse
from PIL import Image
from utils import preprocess, tools

from safetensors.torch import load_file, save_file
from args import parse_args
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from transformers import AutoTokenizer, PretrainedConfig
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import crop
import tqdm

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 
os.environ["OPENCV_IMGCODECS_USE_OPENEXR"] = "1"

def log_validation(args, weight_dtype, dataloader, device='cuda'):
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    pipeline = tools.get_pipeline(
        args.pretrained_model_name_or_path,
        args.pretrained_unet_model_name_or_path,
        args.controlnext_model_name_or_path,
        args.lightenc_model_name_or_path,
        args.depth_fusion_model_name_or_path,
        vae_model_name_or_path=vae_path,
        lora_path=None,
        load_weight_increasement=args.load_weights_increaments,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        revision=args.revision,
        variant=args.variant,
        hf_cache_dir=None,
        use_safetensors=args.use_safetensors,
        device=device,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # image_logs = []
    inference_ctx = torch.autocast(device)

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        validation_image = batch["source"].to(dtype=weight_dtype)
        validation_image = (validation_image + 1) / 2.0
        validation_prompt = batch["caption"][0]
        gt = batch["target"].to(dtype=weight_dtype)
        gt = (gt + 1) / 2.0
        bg = batch["bg"].to(dtype=weight_dtype)
        bg = (bg + 1) / 2.0
        source = batch["source"].to(dtype=weight_dtype)
        source = (source + 1) / 2.0
        validation_image = validation_image*batch["soft_mask"] + bg*(1-batch["soft_mask"])
        inputs = (batch["source"]*batch["soft_mask"]).to(dtype=weight_dtype)

        images = []
        control_image = (batch["source"]*batch["soft_mask"]).to(device, dtype=torch.float32)
        control_image_2 = batch["bg"].to(device, dtype=torch.float32)

        lighting = batch["lighting"].to(device, dtype=torch.float32)

        depth = batch["depth"].to(device, dtype=torch.float32)

        controlnext_image = depth
        ref_image = (batch["source"]*batch["soft_mask"]).to(device, dtype=torch.float32)
        controlnext_image = torch.cat([controlnext_image, ref_image], dim=1)
        # controlnext_image = None

        with inference_ctx:
            image = pipeline(
                prompt=validation_prompt,
                image=inputs,
                guidance_scale=1.0,
                strength=1.0,
                control_image=control_image,
                control_image_2=control_image_2,
                light_image=lighting,
                controlnext_image=controlnext_image,
                controlnet_scale=args.controlnext_scale_factor,
                num_inference_steps=20,
                generator=generator,
                negative_prompt=None,
                width=1024,
                height=1024,
                output_type='pt',
            ).images
        image = image*batch["soft_mask"] + bg*(1-batch["soft_mask"])

        images.append(image[0])

        log = {"validation_image": validation_image[0], "images": images, "validation_prompt": validation_prompt, "gt": gt[0]}

        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]
        gt = log["gt"]

        for image in images:
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            save_dir_path = '/data1/lihaochen/Relight/stage2/test_attnmask_soft'
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            person = batch["person"][0].split('/')[-1]
            file_path = os.path.join(save_dir_path, f"{person}.png")
            cv2.imwrite(file_path, image)

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return []

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def get_train_dataset(args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    while True:
        try:
            if args.dataset_name is not None:
                # Downloading and loading a dataset from the hub.
                dataset = load_dataset(
                    args.dataset_name,
                    #args.dataset_config_name,
                    data_files=args.data_files,
                    cache_dir=args.cache_dir,
                )
            else:
                if args.train_data_dir is not None:
                    dataset = load_dataset(
                        args.train_data_dir,
                        cache_dir=args.cache_dir,
                    )
                # See more about loading custom images at
                # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            break
        except Exception as e:
            continue

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names

    train_dataset = dataset["train"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset

def get_top_center_crop_params(img_shape, resolution):
    height, width = img_shape
    if height > width:
        top = max(0, height // 4 - resolution // 2)
        left = max(0, (width - resolution) // 2)
    else:
        top = max(0, (height - resolution) // 2)
        left = max(0, (width - resolution) // 2)
    return top, left, resolution, resolution

def align_and_crop_bg(bg, fg_crop_params, bg_crop_params, fg_shape, fill_mode='edge'):
    """
    根据前景的裁剪参数和完整尺寸，对背景进行对齐、裁剪和填充。

    该函数的核心逻辑是：
    1. 创建一个与前景原始图像 (fg_shape) 等大的空白背景画布 (bg_processed)。
    2. 计算前景裁剪框与背景裁剪框之间的相对偏移量。
    3. 根据此偏移量，将背景图像 (bg) 的相关部分复制到新的画布 (bg_processed) 上的正确位置。
       这确保了裁剪区域的相对位置保持一致。
    4. 对画布上未被背景图像覆盖的区域进行边缘填充。

    参数:
        bg (np.ndarray): 背景图像。
        fg_crop_params (tuple): 前景的裁剪参数 (top, left, h, w)。
                                h 和 w 在此函数中不直接使用，但为了接口一致性而保留。
        bg_crop_params (tuple): 背景的裁剪参数 (top, left, h, w)。
                                h 和 w 在此函数中不直接使用。
        fg_shape (tuple): 前景原始图像的完整形状 (height, width)。
        fill_mode (str): 填充模式，目前支持 'edge'。

    返回:
        np.ndarray: 处理后与前景对齐、大小相同的背景。
    """
    fg_top, fg_left, _, _ = fg_crop_params
    bg_top, bg_left, _, _ = bg_crop_params
    fg_h, fg_w = fg_shape
    bg_h, bg_w = bg.shape[:2]

    # 1. 创建一个与前景原始图像大小相同的空白画布
    bg_processed = np.zeros((fg_h, fg_w), dtype=bg.dtype)

    # 2. 计算背景相对于前景画布的偏移量
    # 这个偏移量描述了 bg 的 (0,0) 点相对于 bg_processed 的 (0,0) 点的位置
    offset_y = - (bg_top - fg_top)
    offset_x = - (bg_left - fg_left)

    # 3. 计算源区域 (在 bg 中) 和目标区域 (在 bg_processed 中) 的重叠部分
    
    # 源区域的起始坐标 (在 bg 坐标系中)
    src_top = max(0, -offset_y)
    src_left = max(0, -offset_x)
    # 源区域的结束坐标 (在 bg 坐标系中)
    src_bottom = min(bg_h, fg_h - offset_y)
    src_right = min(bg_w, fg_w - offset_x)

    # 目标区域的起始坐标 (在 bg_processed 坐标系中)
    dst_top = max(0, offset_y)
    dst_left = max(0, offset_x)
    # 目标区域的结束坐标 (在 bg_processed 坐标系中)
    dst_bottom = min(fg_h, bg_h + offset_y)
    dst_right = min(fg_w, bg_w + offset_x)

    # 只有当源区域和目标区域都有效时才进行复制
    if src_right > src_left and src_bottom > src_top:
        bg_processed[dst_top:dst_bottom, dst_left:dst_right] = \
            bg[src_top:src_bottom, src_left:src_right]

    # 4. 根据 fill_mode 对空白区域进行填充
    if fill_mode == 'edge':
        # 填充顶部
        if dst_top > 0:
            bg_processed[:dst_top, :] = bg_processed[dst_top, :]
        # 填充底部
        if dst_bottom < fg_h:
            bg_processed[dst_bottom:, :] = bg_processed[dst_bottom-1, :]
        # 填充左侧
        if dst_left > 0:
            bg_processed[:, :dst_left] = bg_processed[:, dst_left:dst_left+1]
        # 填充右侧
        if dst_right < fg_w:
            bg_processed[:, dst_right:] = bg_processed[:, dst_right-1:dst_right]
            
    return bg_processed

def adjust_and_fuse_depth(
    foreground_depth, background_depth, foreground_mask,
    fg_crop_params, bg_crop_params, bottom_rows=5, fill_mode='edge'
):
    fg_h, fg_w = foreground_depth.shape
    bg_crop = align_and_crop_bg(background_depth, fg_crop_params, bg_crop_params, (fg_h, fg_w), fill_mode=fill_mode)
    bottom_mask = np.zeros_like(foreground_mask, dtype=bool)
    rows, cols = np.where(foreground_mask == 1)
    if len(rows) == 0:
        return bg_crop
    unique_cols = np.unique(cols)
    for col in unique_cols:
        col_rows = rows[cols == col]
        if len(col_rows) > 0:
            bottom_row = np.max(col_rows)
            bottom_mask[col_rows[col_rows >= (bottom_row - bottom_rows + 1)], col] = True
    foreground_bottom_depth = np.mean(foreground_depth[bottom_mask])
    background_bottom_depth = np.mean(bg_crop[bottom_mask])
    depth_diff = background_bottom_depth - foreground_bottom_depth
    adjusted_fg = foreground_depth.copy()
    adjusted_fg[foreground_mask == 1] += depth_diff
    fused = bg_crop.copy()
    fused[foreground_mask == 1] = adjusted_fg[foreground_mask == 1]
    return fused

def prepare_train_dataset(dataset):
    # 自定义裁剪函数
    class TopCenterCrop:
        def __init__(self, resolution):
            self.resolution = resolution

        def __call__(self, img):
            _, height, width = img.shape  # 获取图像宽高
            if height > width:  # 竖图
                top = max(0, height // 4 - self.resolution // 2)  # 向上偏移裁剪
                left = max(0, (width - self.resolution) // 2)  # 水平居中裁剪
            else:  # 横图，保持中心裁剪
                top = max(0, (height - self.resolution) // 2)
                left = max(0, (width - self.resolution) // 2)

            return crop(img, top, left, self.resolution, self.resolution)
    
    image_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
            TopCenterCrop(args.resolution),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    
    conditioning_image_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
            TopCenterCrop(args.resolution),
        ]
    )

    bg_image_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(size=32, max_size=64, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(32),
            v2.functional.horizontal_flip,
        ]
    )

    def preprocess_train(examples):
        source = [cv2.imread(source, cv2.IMREAD_UNCHANGED) for source in examples['source']]
        source = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in source]
        # 获取前景和背景的原图shape
        img_depth_raw = [np.load(depth) for depth in examples['img_depth']]
        bg_depth_raw = [np.load(depth) for depth in examples['bg_depth']]
        mask_raw = [cv2.imread(mask, cv2.IMREAD_UNCHANGED) for mask in examples['mask']]
        mask_bin = [np.where(m > 0, 1, 0).astype(np.float32) for m in mask_raw]
        # 计算crop参数
        fg_crop_params = [get_top_center_crop_params(d.shape, args.resolution) for d in img_depth_raw]
        bg_crop_params = [get_top_center_crop_params(d.shape, args.resolution) for d in bg_depth_raw]
        # 融合
        depth = [adjust_and_fuse_depth(fg, bg, m, fg_p, bg_p) for fg, bg, m, fg_p, bg_p in zip(img_depth_raw, bg_depth_raw, mask_bin, fg_crop_params, bg_crop_params)]
        # 后续处理
        source = [image_transforms(s) for s in source]
        target = [cv2.imread(target, cv2.IMREAD_UNCHANGED) for target in examples['target']]
        target = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in target]
        target = [image_transforms(t) for t in target]
        mask = [np.expand_dims(m, axis=-1) for m in mask_bin]
        mask = [conditioning_image_transforms(m) for m in mask]
        soft_mask = [np.expand_dims(m, axis=-1) for m in mask_raw]
        soft_mask = [conditioning_image_transforms(m) for m in soft_mask]
        img_depth = [np.expand_dims(d, axis=-1) for d in img_depth_raw]
        img_depth = [conditioning_image_transforms(d) for d in img_depth]
        bg_depth = [np.expand_dims(d, axis=-1) for d in bg_depth_raw]
        bg_depth = [conditioning_image_transforms(d) for d in bg_depth]
        depth = [np.expand_dims(d, axis=-1) for d in depth]
        depth = [conditioning_image_transforms(d) for d in depth]
        lighting = [cv2.imread(lighting, cv2.IMREAD_UNCHANGED) for lighting in examples['lighting']]
        lighting = [cv2.cvtColor(l, cv2.COLOR_BGR2RGB) for l in lighting]
        lighting = [np.roll(l, l.shape[1] // 2, 1) for l in lighting]
        lighting = [bg_image_transforms(l) for l in lighting]
        bg = [cv2.imread(bg, cv2.IMREAD_UNCHANGED) for bg in examples['bg']]
        bg = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bg]
        bg = [image_transforms(b) for b in bg]
        examples['source'] = source
        examples['target'] = target
        examples['mask'] = mask
        examples['soft_mask'] = soft_mask
        examples['depth'] = depth
        examples['lighting'] = lighting
        examples['fg_depth'] = img_depth
        examples['bg_depth'] = bg_depth
        examples['bg'] = bg
        return examples

    dataset = dataset.with_transform(preprocess_train)

    # dataset = dataset.remove_columns(["person"])
    
    return dataset


def collate_fn(examples):
    source = torch.stack([example["source"] for example in examples])
    source = source.to(memory_format=torch.contiguous_format).float()

    target = torch.stack([example["target"] for example in examples])
    target = target.to(memory_format=torch.contiguous_format).float()

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    soft_mask = torch.stack([example["soft_mask"] for example in examples])
    soft_mask = soft_mask.to(memory_format=torch.contiguous_format).float()

    depth = torch.stack([example["depth"] for example in examples])
    depth = depth.to(memory_format=torch.contiguous_format).float()

    fg_depth = torch.stack([example["fg_depth"] for example in examples])
    fg_depth = fg_depth.to(memory_format=torch.contiguous_format).float()

    bg_depth = torch.stack([example["bg_depth"] for example in examples])
    bg_depth = bg_depth.to(memory_format=torch.contiguous_format).float()

    lighting = torch.stack([example["lighting"] for example in examples])
    lighting = lighting.to(memory_format=torch.contiguous_format).float()

    bg = torch.stack([example["bg"] for example in examples])
    bg = bg.to(memory_format=torch.contiguous_format).float()

    caption = [example["caption"] for example in examples]

    person = [example["person"] for example in examples]

    return {
        "source": source,
        "target": target,
        "mask": mask,
        "soft_mask": soft_mask,
        "depth": depth,
        "fg_depth": fg_depth,
        "bg_depth": bg_depth,
        "lighting": lighting,
        "bg": bg,
        "caption": caption,
        "person": person,
    }

def main(args):
    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32

    train_dataset = get_train_dataset(args)

    # Then get the training dataset ready to be passed to the dataloader.
    val_dataset = prepare_train_dataset(train_dataset)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    image_logs = log_validation(
        args=args,
        weight_dtype=weight_dtype,
        dataloader=val_dataloader,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)