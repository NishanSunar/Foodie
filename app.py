import streamlit as st
from PIL import Image

from food_recognition import recognize_food
from recipe_generator import generate_recipe


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Foodie",
    page_icon=None,
    layout="wide"
)


# ==========================================================
# HEADER
# ==========================================================

st.title("Foodie")

st.markdown(
    """
### AI Food Recognition & Recipe Generation

Upload an image of food. Foodie identifies the food
using a Vision Transformer and generates a customized
recipe using a Transformer-based language model.
"""
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Recipe Settings")

servings = st.sidebar.selectbox(
    "Number of servings",
    [1, 2, 3, 4, 5, 6, 8, 9, 10],
    index=1
)

dietary_preference = st.sidebar.selectbox(
    "Preference",
    [
        "Regular",
        "Vegetarian",
        "Non-vegetarian"
    ]
)


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a food image to begin."
    )

    st.stop()


# ==========================================================
# LOAD IMAGE
# ==========================================================

image = Image.open(
    uploaded_file
).convert("RGB")


# ==========================================================
# DISPLAY IMAGE
# ==========================================================

image_column, result_column = st.columns(
    [1, 1]
)


with image_column:

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width="stretch"
    )


# ==========================================================
# FOOD RECOGNITION
# ==========================================================

with st.spinner(
    "Analyzing food image..."
):

    predictions = recognize_food(
        image,
        top_k=5
    )


food_name = predictions[0]["label"]

confidence = (
    predictions[0]["score"] * 100
)


# ==========================================================
# RECOGNITION RESULT
# ==========================================================

with result_column:

    st.subheader(
        "Recognition Result"
    )

    st.metric(
        "Detected Food",
        food_name.replace(
            "_",
            " "
        ).title()
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


# ==========================================================
# CONFIDENCE WARNING
# ==========================================================

if confidence < 60:

    st.warning(
        "The model has low confidence in this prediction. "
        "The generated recipe may not be accurate."
    )


st.divider()


# ==========================================================
# RECIPE SECTION
# ==========================================================

st.subheader(
    "Recipe Generator"
)

st.write(
    f"""
Generate a {dietary_preference.lower()} recipe for
**{food_name.replace("_", " ")}** for
**{servings} people**.
"""
)


if st.button(
    "Generate Recipe",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Generating recipe..."
    ):

        recipe = generate_recipe(
            food_name=food_name,
            servings=servings,
            dietary_preference=dietary_preference
        )


    st.success(
        "Recipe generated successfully."
    )


    st.markdown(
        recipe
    )