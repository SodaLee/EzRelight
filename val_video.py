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
from einops import rearrange

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
    pipeline = tools.get_pipeline_vid(
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

    inference_ctx = torch.autocast(device)

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        b, _, f, h, w = batch["source"].shape
        validation_image = batch["source"].to(dtype=weight_dtype)
        validation_image = (validation_image + 1) / 2.0
        validation_prompt = batch["caption"][0][0]
        gt = batch["target"].to(dtype=weight_dtype)
        gt = (gt + 1) / 2.0
        bg = batch["bg"].to(dtype=weight_dtype)
        bg = (bg + 1) / 2.0
        validation_image = validation_image*batch["mask"] + bg*(1-batch["mask"])
        validation_image = rearrange(validation_image, "b c f h w -> (b f) c h w")
        gt = rearrange(gt, "b c f h w -> (b f) c h w")

        inputs = (batch["source"]*batch["mask"]).to(dtype=weight_dtype)

        # images = []
        control_image = (batch["source"]*batch["mask"]).to(device, dtype=torch.float32)
        control_image_2 = batch["bg"].to(device, dtype=torch.float32)

        lighting = batch["lighting"].to(device, dtype=torch.float32)

        depth = batch["depth"].to(device, dtype=torch.float32)

        controlnext_image = depth
        ref_image = (batch["source"]*batch["mask"]).to(device, dtype=torch.float32)
        controlnext_image = torch.cat([controlnext_image, ref_image], dim=1)

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
        image = image*batch["mask"] + bg*(1-batch["mask"])
        image = rearrange(image, "b c f h w -> (b f) c h w")

        log = {"validation_image": validation_image, "images": image, "validation_prompt": validation_prompt, "gt": gt}
        log_images = log["images"]
        # validation_prompt = log["validation_prompt"]
        validation_images = log["validation_image"]
        gts = log["gt"]

        # for j in range(f):
        #     formatted_images = []
        #     formatted_images.append(np.asarray(validation_images[j].permute(1, 2, 0).cpu()))
        #     formatted_images.append(np.asarray(log_images[j].permute(1, 2, 0).cpu()))
        #     formatted_images.append(np.asarray(gts[j].permute(1, 2, 0).cpu()))
        #     formatted_images = np.concatenate(formatted_images, 1)

        #     file_path = os.path.join(save_dir_path, f"image_{i}_{j}.png")
        #     formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(file_path, formatted_images * 255)

        file_names = batch["file_names"][0]
        for j in range(f):
            img = log_images[j].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # '/data/lihaochen/datasets/TikTok_dataset/{video_name}/relight/{frame_name}.png'
            f_name = file_names[j]
            video_name = f_name.split('/')[-3]
            frame_name = f_name.split('/')[-1].split('.')[0]
            save_dir_path = '/data1/lihaochen/Relight/video/val_attnmask'
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            file_path = os.path.join(save_dir_path, f"{video_name}_{frame_name}_{j}.png")
            cv2.imwrite(file_path, img)
        
        # for j in range(f):
        #     gt = gts[j].permute(1, 2, 0).cpu().numpy()
        #     gt = (gt * 255).astype(np.uint8)
        #     gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        #     # '/data/lihaochen/datasets/TikTok_dataset/{video_name}/relight/{frame_name}.png'
        #     f_name = file_names[j]
        #     video_name = f_name.split('/')[-3]
        #     frame_name = f_name.split('/')[-1].split('.')[0]
        #     save_dir_path = '/data1/lihaochen/Relight/video/gt'
        #     if not os.path.exists(save_dir_path):
        #         os.makedirs(save_dir_path)
        #     file_path = os.path.join(save_dir_path, f"{video_name}_{frame_name}_{j}.png")
        #     cv2.imwrite(file_path, gt)

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

    def adjust_and_fuse_depth(foreground_depth, background_depth, foreground_mask, bottom_rows=5):
        """
        将前景深度图与背景深度图融合，确保前景物体（如人）的脚部深度与背景一致。

        参数:
            foreground_depth (np.ndarray): 前景深度图。
            background_depth (np.ndarray): 背景深度图。
            foreground_mask (np.ndarray): 前景掩码，1表示前景，0表示背景。
            bottom_rows (int): 取前景物体底部的行数，默认为5。

        返回:
            np.ndarray: 融合后的深度图。
        """
        # 确保输入数组的尺寸一致
        assert foreground_depth.shape == background_depth.shape == foreground_mask.shape, "输入数组的尺寸必须一致"
        
        # 找到前景物体的底部区域（取底部多行）
        bottom_mask = np.zeros_like(foreground_mask, dtype=bool)
        rows, cols = np.where(foreground_mask == 1)  # 找到所有前景像素的行和列
        if len(rows) == 0:
            return background_depth  # 如果没有前景物体，直接返回背景深度图
        
        # 找到每一列的前景物体的最底部行
        unique_cols = np.unique(cols)  # 所有有前景物体的列
        for col in unique_cols:
            col_rows = rows[cols == col]  # 当前列的所有前景行
            if len(col_rows) > 0:
                bottom_row = np.max(col_rows)  # 当前列的最底部行
                # 取底部多行
                bottom_mask[col_rows[col_rows >= (bottom_row - bottom_rows + 1)], col] = True
        
        # 计算前景物体底部的平均深度
        foreground_bottom_depth = np.mean(foreground_depth[bottom_mask])
        
        # 计算背景对应区域的平均深度
        background_bottom_depth = np.mean(background_depth[bottom_mask])
        
        # 计算深度差异
        depth_diff = background_bottom_depth - foreground_bottom_depth
        
        # 调整前景物体的深度值
        adjusted_foreground_depth = foreground_depth.copy()
        adjusted_foreground_depth[foreground_mask == 1] += depth_diff
        
        # 将调整后的前景深度信息融合到背景深度图中
        fused_depth = background_depth.copy()
        fused_depth[foreground_mask == 1] = adjusted_foreground_depth[foreground_mask == 1]
        
        return fused_depth

    def preprocess_train(examples):
        f = len(examples['source'])
        bs = len(examples['source'][0])

        file_names = examples['source']

        source = []
        for sources in examples['source']:
            for s in sources:
                source.append(cv2.imread(s, cv2.IMREAD_UNCHANGED))
        source = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in source]
        source = [image_transforms(s) for s in source]

        target = []
        for targets in examples['target']:
            for t in targets:
                target.append(cv2.imread(t, cv2.IMREAD_UNCHANGED))
        target = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in target]
        target = [image_transforms(t) for t in target]

        mask = []
        for masks in examples['mask']:
            for m in masks:
                mask.append(cv2.imread(m, cv2.IMREAD_UNCHANGED))

        img_depth = []
        for img_depths in examples['img_depth']:
            for d in img_depths:
                img_depth.append(np.load(d))

        bg_depth = []
        for bg_depths in examples['bg_depth']:
            for d in bg_depths:
                bg_depth.append(np.load(d))

        depth = [adjust_and_fuse_depth(img, bg, m) for img, bg, m in zip(img_depth, bg_depth, mask)]
        
        mask = [np.expand_dims(m, axis=-1) for m in mask]
        mask = [conditioning_image_transforms(m) for m in mask]

        depth = [np.expand_dims(d, axis=-1) for d in depth]
        depth = [conditioning_image_transforms(d) for d in depth]

        lighting = []
        for lightings in examples['lighting']:
            for l in lightings:
                lighting.append(cv2.imread(l, cv2.IMREAD_UNCHANGED))
        lighting = [cv2.cvtColor(l, cv2.COLOR_BGR2RGB) for l in lighting]
        lighting = [np.roll(l, l.shape[1] // 2, 1) for l in lighting]
        lighting = [bg_image_transforms(l) for l in lighting]

        bg = []
        for bgs in examples['bg']:
            for b in bgs:
                bg.append(cv2.imread(b, cv2.IMREAD_UNCHANGED))
        bg = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bg]
        bg = [image_transforms(b) for b in bg]

        # group by batch size
        source = [source[i:i+bs] for i in range(0, len(source), bs)]
        target = [target[i:i+bs] for i in range(0, len(target), bs)]
        mask = [mask[i:i+bs] for i in range(0, len(mask), bs)]
        depth = [depth[i:i+bs] for i in range(0, len(depth), bs)]
        lighting = [lighting[i:i+bs] for i in range(0, len(lighting), bs)]
        bg = [bg[i:i+bs] for i in range(0, len(bg), bs)]

        # stack along dim 1
        source = [torch.stack(s, dim=1) for s in source]
        target = [torch.stack(t, dim=1) for t in target]
        mask = [torch.stack(m, dim=1) for m in mask]
        depth = [torch.stack(d, dim=1) for d in depth]
        lighting = [torch.stack(l, dim=1) for l in lighting]
        bg = [torch.stack(b, dim=1) for b in bg]

        examples['source'] = source
        examples['target'] = target
        examples['mask'] = mask
        examples['depth'] = depth
        examples['lighting'] = lighting
        examples['bg'] = bg
        examples['file_names'] = file_names

        return examples

    dataset = dataset.with_transform(preprocess_train)

    dataset = dataset.remove_columns(["person", "phi"])
    
    return dataset


def collate_fn(examples):
    source = torch.stack([example["source"] for example in examples])
    source = source.to(memory_format=torch.contiguous_format).float()

    target = torch.stack([example["target"] for example in examples])
    target = target.to(memory_format=torch.contiguous_format).float()

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    depth = torch.stack([example["depth"] for example in examples])
    depth = depth.to(memory_format=torch.contiguous_format).float()

    lighting = torch.stack([example["lighting"] for example in examples])
    lighting = lighting.to(memory_format=torch.contiguous_format).float()

    bg = torch.stack([example["bg"] for example in examples])
    bg = bg.to(memory_format=torch.contiguous_format).float()

    caption = [example["caption"] for example in examples]

    file_names = [example["file_names"] for example in examples]

    return {
        "source": source,
        "target": target,
        "mask": mask,
        "mask": mask,
        "depth": depth,
        "lighting": lighting,
        "bg": bg,
        "caption": caption,
        "file_names": file_names,
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