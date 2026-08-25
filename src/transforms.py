import torchvision.transforms as T
from PIL import Image

IMAGE_SIZE = 224

def resize_with_padding(image: Image.Image, target_size: int = IMAGE_SIZE) -> Image.Image:
    """Resize an image to fit within target_size x target_size,
    preserving aspect ratio, padding the rest with black."""
    width, height = image.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = image.resize((new_width, new_height), Image.BILINEAR)

    padded = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    padded.paste(resized, (paste_x, paste_y))

    return padded

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

normalize_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def preprocess_image(image: Image.Image, target_size: int = IMAGE_SIZE):
    """Full preprocessing pipeline: pad-resize, then normalize to a tensor."""
    padded = resize_with_padding(image, target_size)
    tensor = normalize_transform(padded)
    return tensor


train_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.15, contrast=0.15),
])

def preprocess_image_train(image: Image.Image, target_size: int = IMAGE_SIZE):
    """Training pipeline: augment, then pad-resize, then normalize."""
    augmented = train_augmentation(image)
    padded = resize_with_padding(augmented, target_size)
    tensor = normalize_transform(padded)
    return tensor

def preprocess_cached(image: Image.Image):
    """For already-resized cached images: just normalize, no resize needed."""
    return normalize_transform(image)

def preprocess_cached_train(image: Image.Image):
    """For already-resized cached images: augment (on the small image), then normalize."""
    augmented = train_augmentation(image)
    return normalize_transform(augmented)