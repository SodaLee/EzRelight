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

def save_models(depth_fusion, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_file(depth_fusion.state_dict(), os.path.join(output_dir, "depth_fusion.safetensors"))

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
        ]
    )

    def preprocess_train(examples):
        img_depth = [np.load(depth) for depth in examples['img_depth']]
        bg_depth = [np.load(depth) for depth in examples['bg_depth']]
        depth = [np.load(depth) for depth in examples['fused_depth']]
        
        mask = [np.expand_dims(m, axis=-1) for m in mask]
        mask = [conditioning_image_transforms(m) for m in mask]


        img_depth = [np.expand_dims(d, axis=-1) for d in img_depth]
        img_depth = [conditioning_image_transforms(d) for d in img_depth]
        bg_depth = [np.expand_dims(d, axis=-1) for d in bg_depth]
        bg_depth = [conditioning_image_transforms(d) for d in bg_depth]
        depth = [np.expand_dims(d, axis=-1) for d in depth]
        depth = [conditioning_image_transforms(d) for d in depth]

        # phi = [torch.tensor(phi) for phi in examples['phi']]

        examples['mask'] = mask
        examples['depth'] = depth
        examples['fg_depth'] = img_depth
        examples['bg_depth'] = bg_depth
        
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    dataset = dataset.remove_columns(["person"])
    
    return dataset


def collate_fn(examples):
    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    depth = torch.stack([example["depth"] for example in examples])
    depth = depth.to(memory_format=torch.contiguous_format).float()

    fg_depth = torch.stack([example["fg_depth"] for example in examples])
    fg_depth = fg_depth.to(memory_format=torch.contiguous_format).float()

    bg_depth = torch.stack([example["bg_depth"] for example in examples])
    bg_depth = bg_depth.to(memory_format=torch.contiguous_format).float()

    return {
        "mask": mask,
        "depth": depth,
        "fg_depth": fg_depth,
        "bg_depth": bg_depth,
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

    depth_fusion = DepthFusion(128)
    if args.depth_fusion_model_name_or_path and os.path.exists(args.depth_fusion_model_name_or_path):
        logger.info("Loading existing depth_fusion weights")
        depth_fusion.load_state_dict(load_file(args.depth_fusion_model_name_or_path))
    else:
        logger.info("Initializing depth_fusion weights from scratch")

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

    depth_fusion.train()
    depth_fusion.requires_grad_(True)
    params_to_optimize = {'params': list(depth_fusion.parameters()), 'lr': args.learning_rate_controlnext}
    logger.info(f"Number of trainable parameters in depth_fusion: {sum(p.numel() for p in depth_fusion.parameters() if p.requires_grad)}")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        **optimizer_kwargs,
    )

    train_dataset = get_train_dataset(args, accelerator)

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
    depth_fusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        depth_fusion, optimizer, train_dataloader, lr_scheduler
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
            with accelerator.accumulate(depth_fusion):
                fg_depth = batch["fg_depth"].to(accelerator.device, dtype=torch.float32)
                bg_depth = batch["bg_depth"].to(accelerator.device, dtype=torch.float32)
                raw_depth = batch["depth"].to(accelerator.device, dtype=torch.float32)

                depth = torch.cat([fg_depth, bg_depth], dim=1)
                depth = depth_fusion(depth)

                depth_loss = F.mse_loss(depth.float(), raw_depth.float(), reduction="mean")
                
                loss = depth_loss

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
                            accelerator.unwrap_model(depth_fusion),
                            save_path,
                        )
                        logger.info(f"Saved state to {save_path}")

            loss = loss.detach().item()
            loss_recorder.add(loss=loss)
            loss_avr: float = loss_recorder.moving_average(window=1000)
            loss_ema: float = loss_recorder.ema
            logs = {"loss/step": loss, 'loss_avr/step': loss_avr, 'loss_ema/step': loss_ema, 'lr/step': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "final")
        save_models(
            accelerator.unwrap_model(depth_fusion),
            save_path,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)