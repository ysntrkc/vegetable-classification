import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import streamlit as st
from PIL import Image

from src.model.CNN import CNN

classes = [
    "Bean",
    "Bitter Gourd",
    "Bottle Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]


def load_image(image):
    image = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(image)
    image = image.unsqueeze(0)
    return image


def make_prediction(model, image):
    image = load_image(image)
    model.eval()
    with torch.no_grad():
        pred = model(image)
        pred = torch.argmax(pred, dim=1)
    return pred.item()


st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="centered")
st.title("Vegetable Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

cnn_model = CNN(15)

resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
features = list(resnet_model.fc.parameters())[0].shape[1]
resnet_model.fc = torch.nn.Linear(features, 15)

cnn_model.load_state_dict(torch.load("./models/cnn.pt"))
resnet_model.load_state_dict(torch.load("./models/resnet.pt"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    cnn_result = make_prediction(cnn_model, image)
    resnet_result = make_prediction(resnet_model, image)

    print(cnn_result)
    print(resnet_result)

    st.write(f"**CNN:** {classes[cnn_result]}")
    st.write(f"**ResNet:** {classes[resnet_result]}")
