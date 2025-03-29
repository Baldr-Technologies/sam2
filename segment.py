import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class CarNotFoundError(Exception):
    """Raised when no cars are detected in an image."""
    pass

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def apply_alpha_blending(img):
    """Applies alpha blending to an image with an alpha channel onto a checkerboard background."""
    display_img = None
    # Check if image is valid and has an alpha channel
    if img is None:
        return None # Return None if the input image is invalid

    if len(img.shape) < 3 or img.shape[2] != 4:
        print("Image does not have an alpha channel or is not BGRA. Displaying as is.")
        return img # Return original image if no alpha channel

    print("Image has an alpha channel. Applying alpha blending...")

    # Separate the BGR channels and the alpha channel
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    # Create a checkerboard background
    h, w = img.shape[:2]
    tile_size = 10 # Size of the checkerboard tiles
    background = np.zeros((h, w, 3), dtype=np.uint8)
    c1, c2 = (200, 200, 200), (255, 255, 255) # Light gray and white tiles
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                color = c1
            else:
                color = c2
            background[y:y + tile_size, x:x + tile_size, :] = color

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha_normalized = alpha / 255.0
    alpha_normalized = alpha_normalized[:, :, np.newaxis] # Add channel dimension for broadcasting

    # Perform alpha blending: foreground * alpha + background * (1 - alpha)
    foreground = bgr.astype(float)
    background_float = background.astype(float)
    blended = alpha_normalized * foreground
    blended += (1.0 - alpha_normalized) * background_float

    display_img = blended.astype(np.uint8)
    return display_img

def post_process(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the largest contour (assuming it's the object)
    contour = max(contours, key=cv2.contourArea)

    # Create an empty mask
    object_mask = np.zeros_like(mask)

    # Fill the contour to create a new mask
    cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    processed = cv2.bitwise_and(image, image, mask=object_mask)
    # Convert to RGBA format (add an alpha channel)
    rgba_image = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)

    # Set alpha channel: 0 for background, 255 for object
    rgba_image[:, :, 3] = (object_mask > 0).astype(np.uint8) * 255  # Transparent background
    return rgba_image

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "bmw3.jpeg"

# Run inference on the source
results = model(source)  # list of Results objects

# No batch processing:
result = results[0] 
bounding_box = None
for box in result.boxes:
    class_name = result.names[int(box.cls)]
    if class_name in ("truck", "car"):
        bounding_box = box.xyxy.squeeze().cpu()
if bounding_box is None:
    raise CarNotFoundError

print("Found car box: ", bounding_box)

image = Image.open(source)
image = np.array(image.convert("RGB"))

checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bounding_box[None,:],
        multimask_output=False,
    )

mask = masks[0].astype(np.uint8)
processed = post_process(image,mask)
processed = Image.fromarray(processed)
processed.save("extracted_bmw3.png",format="PNG", quality=100)