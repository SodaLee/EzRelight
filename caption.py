import os
from PIL import Image, ImageStat
import numpy as np
import torch
import cv2
import tqdm
from transformers import AutoModel, AutoTokenizer

def ACESToneMapping(color, adapted_lum):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14

    color *= adapted_lum
    return (color * (A * color + B)) / (color * (C * color + D) + E)

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 
os.environ["OPENCV_IMGCODECS_USE_OPENEXR"] = "1"
torch.manual_seed(0)

model = AutoModel.from_pretrained('/data3/lihaochen/datasets/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/data3/lihaochen/datasets/MiniCPM-V-2_6', trust_remote_code=True)

question = "Please describe the only person in the image. Use simple and clear language. Do not describe the background. No more than 30 words."

dataset_root = '/data3/lihaochen/datasets/synthetic_human_pp/'
person = os.listdir(dataset_root)
person = [p for p in person if p != 'lighting' and os.path.isdir(os.path.join(dataset_root, p))]
person.sort()

for p in person:
    relight = os.path.join(dataset_root, p, 'relight')
    relight = os.listdir(relight)
    relight = [r for r in relight if os.path.isdir(os.path.join(dataset_root, p, 'relight', r))]
    relight.sort()
    d = {r:{} for r in relight}
    angle = os.path.join(dataset_root, p, 'relight', relight[0], 'rendering')
    angle = os.listdir(angle)
    angle = [a for a in angle if os.path.isfile(os.path.join(dataset_root, p, 'relight', relight[0], 'rendering', a))]
    angle.sort()
    mask = os.path.join(dataset_root, p, 'relight', relight[0], 'normal')
    mask = os.listdir(mask)
    mask_f = [m for m in mask if os.path.isfile(os.path.join(dataset_root, p, 'relight', relight[0], 'normal', m))]
    mask_f.sort()

    loop = tqdm.tqdm(range(len(relight) * len(angle)))
    for r in relight:
        for a, m in zip(angle, mask_f):
            loop.set_description(f'{p} {r} {a}')
            loop.update(1)
            source = os.path.join(dataset_root, p, 'relight', r, 'rendering', a)
            mask = os.path.join(dataset_root, p, 'relight', r, 'normal', m)

            img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
            mask = 1 - np.all(mask == [191,191,191], axis=-1).astype(np.uint8)
            mask = np.expand_dims(mask, axis=-1)
            img = img * mask
            img = ACESToneMapping(img, 1.0).clip(0, 1)
            img = Image.fromarray((img * 255).astype(np.uint8))

            msgs = [{'role': 'user', 'content': [img, question]}]
            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
            
            d[r][a] = answer
    
    np.save(f'/data3/lihaochen/datasets/synthetic_human_pp/{p}_prompt.npy', d)