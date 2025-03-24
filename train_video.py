#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import functools
import gc
import re
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import cv2
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file, save_file
from pipeline.pipeline_controlnext_img2img import StableDiffusionXLControlNeXtImg2ImgPipeline
from models.controlnext import ControlNetModel as ControlNext
from models.unet import UNet2DConditionModel
from models.lightenc import LightEnc, MLP5, DepthFusion
from args import parse_args

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 
os.environ["OPENCV_IMGCODECS_USE_OPENEXR"] = "1"

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        # 计算预测值与真实值之间的平方差
        squared_diff = (pred - target) ** 2
        # 应用掩码
        masked_squared_diff = squared_diff * mask
        # 计算平均损失
        loss = masked_squared_diff.sum() / mask.sum()
        return loss

def save_models(unet, controlnext, lightenc, consistency_mlp, output_dir, args, orig_unet_sd=None):
    os.makedirs(output_dir, exist_ok=True)
    unet_sd = unet.state_dict()
    pattern = re.compile(args.unet_trainable_param_pattern)
    extra_save = ["conv_in.weight", "conv_in.bias"]
    unet_sd = {k: v for k, v in unet_sd.items() if pattern.match(k) or k in extra_save or "temporal" in k}
    if args.save_weights_increaments:
        for k, v in unet_sd.items():
            unet_sd[k] = unet_sd[k].detach().cpu() - orig_unet_sd[k]
    save_file(unet_sd, os.path.join(output_dir, "unet_weight_increasements.safetensors"))
    save_file(controlnext.state_dict(), os.path.join(output_dir, "controlnext.safetensors"))
    save_file(lightenc.state_dict(), os.path.join(output_dir, "lightenc.safetensors"))
    save_file(consistency_mlp.state_dict(), os.path.join(output_dir, "consistency_mlp.safetensors"))
    # save_file(depth_fusion.state_dict(), os.path.join(output_dir, "depth_fusion.safetensors"))


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


class LossRecorder:
    r"""
    Class to record better losses.
    """

    def __init__(self, gamma=0.9, max_window=None):
        self.losses = []
        self.gamma = gamma
        self.ema = 0
        self.t = 0
        self.max_window = max_window

    def add(self, *, loss: float) -> None:
        self.losses.append(loss)
        if self.max_window is not None and len(self.losses) > self.max_window:
            self.losses.pop(0)
        self.t += 1
        ema = self.ema * self.gamma + loss * (1 - self.gamma)
        ema_hat = ema / (1 - self.gamma ** self.t) if self.t < 500 else ema
        self.ema = ema_hat

    def moving_average(self, *, window: int) -> float:
        if len(self.losses) < window:
            window = len(self.losses)
        return sum(self.losses[-window:]) / window

def get_train_dataset(args, accelerator):
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
            logger.error(f"Error loading dataset: {e}")
            logger.error("Retry...")
            continue

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    logger.info(f"image column defaulting to 'target'")
    logger.info(f"mask column defaulting to 'mask'")
    logger.info(f"depth column defaulting to 'depth'")
    logger.info(f"lighting column defaulting to 'lighting'")
    logger.info(f"source column defaulting to 'source'")
    logger.info(f"caption column defaulting to 'caption'")

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def prepare_train_dataset(dataset, accelerator):
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

    source_transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
            TopCenterCrop(args.resolution),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
        # source = [image_transforms(s) for s in source]
        source = [source_transforms(s) for s in source]

        target = [cv2.imread(target, cv2.IMREAD_UNCHANGED) for target in examples['target']]
        target = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in target]
        target = [image_transforms(t) for t in target]

        mask = [cv2.imread(mask, cv2.IMREAD_UNCHANGED) for mask in examples['mask']]
        # mask = [np.where(m > 0, 1, 0).astype(np.float32) for m in mask]
        # img_depth = [np.load(depth) for depth in examples['img_depth']]
        # bg_depth = [np.load(depth) for depth in examples['bg_depth']]
        # depth = [np.where(m != 0, d1, d2) for m, d1, d2 in zip(mask, img_depth, bg_depth)]
        depth = [np.load(depth) for depth in examples['fused_depth']]
        
        mask = [np.expand_dims(m, axis=-1) for m in mask]
        mask = [conditioning_image_transforms(m) for m in mask]

        # img_depth = [np.expand_dims(d, axis=-1) for d in img_depth]
        # img_depth = [conditioning_image_transforms(d) for d in img_depth]
        # bg_depth = [np.expand_dims(d, axis=-1) for d in bg_depth]
        # bg_depth = [conditioning_image_transforms(d) for d in bg_depth]
        depth = [np.expand_dims(d, axis=-1) for d in depth]
        depth = [conditioning_image_transforms(d) for d in depth]

        lighting = [cv2.imread(lighting, cv2.IMREAD_UNCHANGED) for lighting in examples['lighting']]
        lighting = [cv2.cvtColor(l, cv2.COLOR_BGR2RGB) for l in lighting]
        lighting = [np.roll(l, l.shape[1] // 2, 1) for l in lighting]
        lighting = [bg_image_transforms(l) for l in lighting]

        bg = [cv2.imread(bg, cv2.IMREAD_UNCHANGED) for bg in examples['bg']]
        bg = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bg]
        bg = [image_transforms(b) for b in bg]

        # phi = [torch.tensor(phi) for phi in examples['phi']]

        examples['source'] = source
        examples['target'] = target
        examples['mask'] = mask
        examples['depth'] = depth
        examples['lighting'] = lighting
        # examples['fg_depth'] = img_depth
        # examples['bg_depth'] = bg_depth
        examples['bg'] = bg
        
        return examples

    with accelerator.main_process_first():
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

    # fg_depth = torch.stack([example["fg_depth"] for example in examples])
    # fg_depth = fg_depth.to(memory_format=torch.contiguous_format).float()

    # bg_depth = torch.stack([example["bg_depth"] for example in examples])
    # bg_depth = bg_depth.to(memory_format=torch.contiguous_format).float()

    lighting = torch.stack([example["lighting"] for example in examples])
    lighting = lighting.to(memory_format=torch.contiguous_format).float()

    bg = torch.stack([example["bg"] for example in examples])
    bg = bg.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "source": source,
        "target": target,
        "mask": mask,
        "depth": depth,
        # "fg_depth": fg_depth,
        # "bg_depth": bg_depth,
        "lighting": lighting,
        "bg": bg,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision if args.pretrained_vae_model_name_or_path is None else None,
        variant=args.variant if args.pretrained_vae_model_name_or_path is None else None,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, use_safetensors=args.use_safetensors,
    )

    # if args.load_weights_increaments or args.save_weights_increaments:
    import copy
    orig_unet_sd = copy.deepcopy(unet.state_dict())

    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    torch.nn.init.zeros_(new_conv_in.weight)
    new_conv_in.weight.data[:, :4, :, :] = unet.conv_in.weight.data
    new_conv_in.bias.data = unet.conv_in.bias.data
    unet.conv_in = new_conv_in

    if args.pretrained_unet_model_name_or_path:
        logger.info("Loading existing unet weights")
        unet_sd = load_file(args.pretrained_unet_model_name_or_path)
        if args.load_weights_increaments:
            logger.info("Loading unet weights in increaments")
            for k in orig_unet_sd.keys():
                if k in unet_sd:
                    unet_sd[k] += orig_unet_sd[k]
                else:
                    unet_sd[k] = orig_unet_sd[k]
        else:
            logger.info("Loading unet weights")
            for k in orig_unet_sd.keys():
                if k not in unet_sd:
                    unet_sd[k] = orig_unet_sd[k]
        unet.load_state_dict(unet_sd)
    else:
        logger.info("Initializing unet weights from scratch")
        pass

    controlnext = ControlNext()
    if args.controlnext_model_name_or_path:
        logger.info("Loading existing controlnext weights")
        controlnext.load_state_dict(load_file(args.controlnext_model_name_or_path))
    else:
        logger.info("Initializing controlnext weights from scratch")

    lightenc = LightEnc()
    if args.lightenc_model_name_or_path:
        logger.info("Loading existing lightenc weights")
        lightenc.load_state_dict(load_file(args.lightenc_model_name_or_path))
    else:
        logger.info("Initializing lightenc weights from scratch")

    consistency_mlp = MLP5(128)
    if args.consistency_mlp_model_name_or_path:
        logger.info("Loading existing consistency_mlp weights")
        consistency_mlp.load_state_dict(load_file(args.consistency_mlp_model_name_or_path))
    else:
        logger.info("Initializing consistency_mlp weights from scratch")
    consistency_loss_fn = MaskedMSELoss()

    # depth_fusion = DepthFusion(128)
    # if args.depth_fusion_model_name_or_path and os.path.exists(args.depth_fusion_model_name_or_path):
    #     logger.info("Loading existing depth_fusion weights")
    #     depth_fusion.load_state_dict(load_file(args.depth_fusion_model_name_or_path))
    # else:
    #     logger.info("Initializing depth_fusion weights from scratch")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnext.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnext.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    
    if unwrap_model(controlnext).dtype != torch.float32:
        raise ValueError(
            f"Controlnext loaded as datatype {unwrap_model(controlnext).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_controlnet = (
            args.learning_rate_controlnet * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_controlnext = (
            args.learning_rate_controlnext * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.optimizer_type.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer_kwargs = dict(
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer_type.lower() == "adafactor":
        optimizer_class = transformers.optimization.Adafactor
        optimizer_kwargs = dict(
            relative_step=args.adafactor_relative_step,
            scale_parameter=args.adafactor_scale_parameter,
            warmup_init=args.adafactor_warmup_init,
        )
    else:
        raise ValueError(f"Optimizer type {args.optimizer_type} not supported.")

    # Optimizer creation
    controlnext.train()
    controlnext.requires_grad_(True)
    params_to_optimize = [{'params': list(controlnext.parameters()), 'lr': args.learning_rate_controlnext}]
    logger.info(f"Number of trainable parameters in controlnext: {sum(p.numel() for p in controlnext.parameters() if p.requires_grad)}")

    lightenc.train()
    lightenc.requires_grad_(True)
    params_to_optimize.append({'params': list(lightenc.parameters()), 'lr': args.learning_rate_controlnext})
    logger.info(f"Number of trainable parameters in lightenc: {sum(p.numel() for p in lightenc.parameters() if p.requires_grad)}")

    consistency_mlp.train()
    consistency_mlp.requires_grad_(True)
    params_to_optimize.append({'params': list(consistency_mlp.parameters()), 'lr': args.learning_rate_controlnext})
    logger.info(f"Number of trainable parameters in consistency_mlp: {sum(p.numel() for p in consistency_mlp.parameters() if p.requires_grad)}")

    # depth_fusion.train()
    # depth_fusion.requires_grad_(True)
    # params_to_optimize.append({'params': list(depth_fusion.parameters()), 'lr': args.learning_rate_controlnext})
    # logger.info(f"Number of trainable parameters in depth_fusion: {sum(p.numel() for p in depth_fusion.parameters() if p.requires_grad)}")

    unet.train()
    unet.requires_grad_(True)
    unet_params = []
    pattern = re.compile(args.unet_trainable_param_pattern)
    extra_save = ["conv_in.weight", "conv_in.bias"]
    for name, param in unet.named_parameters():
        if pattern.match(name) or name in extra_save or "temporal" in name:
            param.requires_grad = True
            unet_params.append(param)
        else:
            param.requires_grad = False
    logger.info(f"Number of trainable parameters in unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    params_to_optimize.append({'params': unet_params, 'lr': args.learning_rate})
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        **optimizer_kwargs,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    # if args.pretrained_vae_model_name_or_path is not None:
    #     vae.to(accelerator.device, dtype=weight_dtype)
    # else:
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnext = controlnext.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, controlnext, lightenc, consistency_mlp, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnext, lightenc, consistency_mlp, optimizer, train_dataloader, lr_scheduler
    )

    patch_accelerator_for_fp16_training(accelerator)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        tracker_config.pop("bg_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    loss_recorder = LossRecorder(gamma=0.9)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, controlnext, lightenc, consistency_mlp):
                # Convert images to latent space
                pixel_values = batch["target"]*batch["mask"]
                latents = vae.encode(pixel_values).latent_dist.mode()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                pixel_values = batch["source"]*batch["mask"]
                latents_source = vae.encode(pixel_values).latent_dist.mode()
                latents_source = latents_source * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents_source = latents_source.to(weight_dtype)

                bg = batch["bg"]
                latents_bg = vae.encode(bg).latent_dist.mode()
                latents_bg = latents_bg * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents_bg = latents_bg.to(weight_dtype)

                # latents = torch.cat([latents, latents_source, latents_bg], dim=1)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = torch.cat([noisy_latents, latents_source, latents_bg], dim=1)

                lighting = batch["lighting"].to(accelerator.device, dtype=torch.float32)

                # random 4x4 mask resize to 32x32
                l_mask = torch.rand((bsz, 1, 4, 4), device=accelerator.device, dtype=torch.float32)
                l_mask = F.interpolate(l_mask, size=(32, 32), mode='bilinear', align_corners=False)
                l1 = lighting * l_mask
                l2 = lighting * (1 - l_mask)
                lighting = torch.reshape(lighting, (bsz, -1))
                l1 = torch.reshape(l1, (bsz, -1))
                l2 = torch.reshape(l2, (bsz, -1))
                lighting = lightenc(lighting)
                l1 = lightenc(l1)
                l2 = lightenc(l2)
                lighting = torch.reshape(lighting, (bsz, 3, 2048))
                l1 = torch.reshape(l1, (bsz, 3, 2048))
                l2 = torch.reshape(l2, (bsz, 3, 2048))

                # fg_depth = batch["fg_depth"].to(accelerator.device, dtype=torch.float32)
                # bg_depth = batch["bg_depth"].to(accelerator.device, dtype=torch.float32)
                raw_depth = batch["depth"].to(accelerator.device, dtype=torch.float32)

                # depth = torch.cat([fg_depth, bg_depth], dim=1)
                # depth = depth_fusion(depth)

                # depth_loss = F.mse_loss(depth.float(), raw_depth.float(), reduction="mean")
                # ControlNext conditioning.
                controlnext_image = raw_depth
                ref_image = (batch["source"]*batch["mask"]).to(accelerator.device, dtype=torch.float32)
                controlnext_image = torch.cat([controlnext_image, ref_image], dim=1)
                controls = controlnext(
                    controlnext_image,
                    timesteps,
                )
                controls['scale'] *= args.controlnext_scale_factor

                added_conditions = batch["unet_added_conditions"]
                enc_hid = torch.cat([batch["prompt_ids"], lighting], dim=1)

                # print(batch["prompt_ids"].shape) # [2, 77, 2048]

                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=enc_hid,
                        added_cond_kwargs=added_conditions,
                        controls=controls,
                        return_dict=False,
                    )[0][:, :4, :, :]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # process l1 & l2
                enc_hid = torch.cat([batch["prompt_ids"], l1], dim=1)
                with accelerator.autocast():
                    model_pred_l1 = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=enc_hid,
                        added_cond_kwargs=added_conditions,
                        controls=controls,
                        return_dict=False,
                    )[0][:, :4, :, :]
                
                enc_hid = torch.cat([batch["prompt_ids"], l2], dim=1)
                with accelerator.autocast():
                    model_pred_l2 = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=enc_hid,
                        added_cond_kwargs=added_conditions,
                        controls=controls,
                        return_dict=False,
                    )[0][:, :4, :, :]

                mlp_out = consistency_mlp(torch.cat([model_pred_l1, model_pred_l2], dim=1))
                # resize mask to the same size as the model output
                mask = batch["mask"].to(accelerator.device, dtype=torch.float32)
                mask = F.interpolate(mask, size=(model_pred.shape[2], model_pred.shape[3]), mode='bilinear', align_corners=False)
                
                loss_c = consistency_loss_fn(mlp_out, target, mask)
                # loss_c = F.mse_loss(mlp_out.float(), target.float(), reduction="mean")

                loss = noise_loss + 0.1 * loss_c  #+ 0.01 * depth_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = []
                    for p in params_to_optimize:
                        params_to_clip.extend(p["params"])
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                        save_models(
                            accelerator.unwrap_model(unet),
                            accelerator.unwrap_model(controlnext),
                            accelerator.unwrap_model(lightenc),
                            accelerator.unwrap_model(consistency_mlp),
                            save_path,
                            args,
                            orig_unet_sd if args.save_weights_increaments else None,
                        )
                        logger.info(f"Saved state to {save_path}")

            loss = loss.detach().item()
            loss_recorder.add(loss=loss)
            loss_avr: float = loss_recorder.moving_average(window=1000)
            loss_ema: float = loss_recorder.ema
            logs = {"loss/step": loss, 'loss_avr/step': loss_avr, 'loss_ema/step': loss_ema, 'lr/step': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            tlogs = logs.copy()
            tlogs['noise_loss'] = noise_loss.detach().item()
            tlogs['loss_c'] = 0.1 * loss_c.detach().item()
            accelerator.log(tlogs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "final")
        save_models(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(controlnext),
            accelerator.unwrap_model(lightenc),
            accelerator.unwrap_model(consistency_mlp),
            save_path,
            args,
            orig_unet_sd if args.save_weights_increaments else None,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)