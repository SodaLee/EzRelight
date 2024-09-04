import cv2
import numpy as np
from PIL import Image


def get_extractor(extractor_name):
    if extractor_name is None:
        return None
    if extractor_name not in EXTRACTORS:
        raise ValueError(f"Extractor {extractor_name} is not supported.")
    return EXTRACTORS[extractor_name]


def canny_extractor(image: Image.Image, threshold1=None, threshold2=None) -> Image.Image:
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)

    sigma = 0.33
    threshold1 = threshold1 or int(max(0, (1.0 - sigma) * v))
    threshold2 = threshold2 or int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, threshold1, threshold2)
    edges = Image.fromarray(edges).convert("RGB")
    return edges


def depth_extractor(image: Image.Image, model):
    """
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    """
    
    # raw_img = cv2.imread('your/image/path')
    raw_img = np.array(image)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth = Image.fromarray(depth.astype('uint16'))
    return 


def pose_extractor(image: Image.Image):
    raise NotImplementedError("Pose extractor is not implemented yet.")


EXTRACTORS = {
    "canny": canny_extractor,
    "depth": depth_extractor,
}