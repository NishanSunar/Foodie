import torch
from transformers import pipeline
import streamlit as st


FOOD_MODEL = "vishnudas08/food101-vit-model"


@st.cache_resource
def load_food_model():

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "image-classification",
        model=FOOD_MODEL,
        device=device
    )

    return classifier


def recognize_food(image, top_k=5):

    classifier = load_food_model()

    predictions = classifier(
        image,
        top_k=top_k
    )

    return predictions