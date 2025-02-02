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

    image_logs = []
    inference_ctx = torch.autocast(device)

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        validation_image = batch["source"].to(dtype=weight_dtype)
        validation_prompt = batch["caption"][0]
        gt = batch["target"].to(dtype=weight_dtype)
        inputs = (batch["source"]*batch["mask"]).to(dtype=weight_dtype)

        images = []
        control_image = (batch["source"]*batch["mask"]).to(device, dtype=torch.float32)
        control_image_2 = torch.zeros_like(control_image).to(device, dtype=torch.float32)

        lighting = batch["lighting"].to(device, dtype=torch.float32)

        controlnext_image = batch["depth"].to(device, dtype=torch.float32)
        ref_image = (batch["source"]*batch["mask"]).to(device, dtype=torch.float32)
        controlnext_image = torch.cat([controlnext_image, ref_image], dim=1)

        with inference_ctx:
            image = pipeline(
                prompt=validation_prompt,
                image=inputs,
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
                output_type='np',
            ).images[0]

        images.append(image)

        image_logs.append(
            {"validation_image": validation_image[0], "images": images, "validation_prompt": validation_prompt, "gt": gt[0]}
        )

        log = image_logs[-1]
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]
        gt = log["gt"]

        formatted_images = []
        formatted_images.append(np.asarray(validation_image.permute(1, 2, 0).cpu()))
        for image in images:
            # formatted_images.append(np.asarray(image.permute(1, 2, 0).cpu()) / 2 + 0.5)
            formatted_images.append(np.asarray(image))
        formatted_images.append(np.asarray(gt.permute(1, 2, 0).cpu()))
        formatted_images = np.concatenate(formatted_images, 1)

        file_path = os.path.join(save_dir_path, "image_{}.png".format(i))
        formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, formatted_images * 255)

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image_logs

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
    image_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(args.resolution),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    
    conditioning_image_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(args.resolution),
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

    def ACESToneMapping(color, adapted_lum):
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14

        color *= adapted_lum
        return (color * (A * color + B)) / (color * (C * color + D) + E)

    def preprocess_train(examples):
        source = [cv2.imread(source, cv2.IMREAD_UNCHANGED) for source in examples['source']]
        source = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in source]
        if args.enable_acestonemapping:
            source = [ACESToneMapping(s, 1.0) for s in source]
        source = [image_transforms(s) for s in source]

        target = [cv2.imread(target, cv2.IMREAD_UNCHANGED) for target in examples['target']]
        target = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in target]
        if args.enable_acestonemapping:
            target = [ACESToneMapping(t, 1.0) for t in target]
        target = [image_transforms(t) for t in target]

        mask = [cv2.imread(mask, cv2.IMREAD_UNCHANGED) for mask in examples['mask']]
        mask = [1 - np.all(m == [191,191,191], axis=-1).astype(np.float32) for m in mask]
        mask = [np.expand_dims(m, axis=-1) for m in mask]
        mask = [conditioning_image_transforms(m) for m in mask]

        depth = [np.load(depth) for depth in examples['depth']]
        depth = [np.expand_dims(d, axis=-1) for d in depth]
        depth = [conditioning_image_transforms(d) for d in depth]

        lighting = [cv2.imread(lighting, cv2.IMREAD_UNCHANGED) for lighting in examples['lighting']]
        lighting = [cv2.cvtColor(l, cv2.COLOR_BGR2RGB) for l in lighting]
        lighting = [np.roll(l, l.shape[1] // 2 - int(l.shape[1] * phi / 2), 1) for l, phi in zip(lighting, examples['phi'])]
        lighting = [bg_image_transforms(l) for l in lighting]

        # phi = [torch.tensor(phi) for phi in examples['phi']]

        examples['source'] = source
        examples['target'] = target
        examples['mask'] = mask
        examples['depth'] = depth
        examples['lighting'] = lighting
        
        return examples

    dataset = dataset.with_transform(preprocess_train)

    dataset = dataset.remove_columns(["person"])
    
    return dataset


def collate_fn(examples):
    source = torch.stack([example["source"] for example in examples])
    source = source.to(memory_format=torch.contiguous_format).float()

    target = torch.stack([example["target"] for example in examples])
    target = target.to(memory_format=torch.contiguous_format).float()

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    depth = torch.stack([example["depth"] for example in examples])
    depth = depth.to(memory_format=torch.contiguous_format).float()

    lighting = torch.stack([example["lighting"] for example in examples])
    lighting = lighting.to(memory_format=torch.contiguous_format).float()

    caption = [example["caption"] for example in examples]

    return {
        "source": source,
        "target": target,
        "mask": mask,
        "depth": depth,
        "lighting": lighting,
        "caption": caption,
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
    )

    image_logs = log_validation(
        args=args,
        weight_dtype=weight_dtype,
        dataloader=val_dataloader,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)