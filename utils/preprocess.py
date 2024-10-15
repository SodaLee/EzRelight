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


def depth_extractor(image, f_px, model):
    """
    from PIL import Image
    import depth_pro

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    """
    
    prediction = model.infer(image, f_px=f_px)
    return prediction


def pose_extractor(image: Image.Image):
    raise NotImplementedError("Pose extractor is not implemented yet.")


EXTRACTORS = {
    "canny": canny_extractor,
    "depth": depth_extractor,
}