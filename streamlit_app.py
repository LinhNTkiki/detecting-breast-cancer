import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit.runtime.uploaded_file_manager import UploadedFile

IMG_SIZE = 224
SAMPLE_IMG_DIR = Path("sample_images")

title = "Breast Cancer Detector"
st.set_page_config(page_title=title)
st.header(title)
st.markdown(
    "Predict whether breast tumours in [histopathological][hp] images are"
    " *benign*, *malignant (cancerous)*, or *normal*.\n\n"
    "[hp]: https://en.wikipedia.org/wiki/Histopathology"
)


def load_image(
    image: Path | UploadedFile, resize: bool = False
) -> Image.Image | tf.Tensor:
    """Convert an input image into the form expected by the model.

    Args:
        image (Path | UploadedFile): Image input.
        resize (bool): Whether or not to resize the image.

    Returns:
        PIL.image.Image | tensorflow.Tensor: A PIL Image. Or a 3D tensor, if
        resize is True.
    """
    img = Image.open(image)
    if resize:
        img = img.convert("RGB")  # Ensure image is in RGB mode
        img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img


@st.cache_data
def get_sample_image_files() -> dict[str, list]:
    """Fetch processed sample images, grouped by label.

    Returns:
        dict: Keys are labels ("benign", "malignant", "normal"). Values are lists of
        images.
    """
    return {
        dir.name: [load_image(file) for file in dir.glob("*.jpg")]
        for dir in SAMPLE_IMG_DIR.iterdir()
    }


@st.cache_resource
def load_model() -> tf.keras.Model:
    """Fetch pretrained model.

    Returns:
        tf.keras.Model: Trained convolutional neural network.
    """
    return tf.keras.models.load_model("cnn_model.h5")


def get_prediction(image: Image.Image | tf.Tensor) -> None:
    """Obtain a prediction for the supplied image, and format the results for
    display.

    Args:
        image (Image | Tensor): An image (PIL Image or 3D tensor).
    """
    pred = model.predict(np.expand_dims(image, 0), verbose=0)[0]
    class_names = ["benign", "malignant", "normal"]
    predicted_class = class_names[np.argmax(pred)]
    st.success(f"Prediction: {predicted_class}")
    st.markdown(f"Predicted probabilities: {dict(zip(class_names, pred))}")
    st.caption(
        "The model's output node has *softmax activation*, providing probabilities "
        "for each class: 'benign', 'malignant', and 'normal'. The class with the highest "
        "probability is the predicted class."
    )


sample_images = get_sample_image_files()
model = load_model()

upload_tab, sample_tab = st.tabs(["Upload an image", "Use a sample image"])

with upload_tab:
    with st.form("image-input-form", clear_on_submit=True):
        file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("submit")
        if file:
            img = load_image(file, resize=True)
            st.image(img.numpy().astype("uint8"))
            get_prediction(img)

with sample_tab:
    if st.button("Get sample image", type="primary"):
        # Randomly select a sample image
        label = np.random.choice(["benign", "malignant", "normal"])
        image_list = sample_images[label]
        idx = np.random.choice(len(image_list))
        st.image(image_list[idx], caption=f"{label} sample")
        get_prediction(image_list[idx])

st.caption(
    "Exploratory data analysis and model training were performed in "
    "[this Kaggle notebook][nb].\n\n"
    "[nb]: https://www.kaggle.com/code/timothyabwao/detecting-breast-cancer"
    "-with-computer-vision"
)
