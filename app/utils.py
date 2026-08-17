import torch
from torchvision import transforms
from PIL import Image


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess_frame(frame):

    image = Image.fromarray(frame).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    return image