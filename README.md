# Breast Cancer Detector

Predict whether breast tumour tissue samples in [histopathological][hp] images are *benign* or *malignant (cancerous)*.

The model used here was trained in [this notebook][nb].

This project is part of my submission as a mentee in the [KaggleX BIPOC Mentorship Program][bipoc].

[hp]: https://en.wikipedia.org/wiki/Histopathology
[nb]: https://www.kaggle.com/code/timothyabwao/detecting-breast-cancer-with-computer-vision
[bipoc]: https://www.kaggle.com/kagglex-bipoc-mentorship-program

<https://breast-cancer-kagglex-project.streamlit.app/>

![screencast](screencast.gif)

## Running locally

1. Fetch the code:

    ```bash
    git clone https://github.com/Tim-Abwao/detecting-breast-cancer.git
    cd detecting-breast-cancer
    ```

2. Create a virtual environment, and install dependencies:

   >**NOTE:** Requires *python3.10* and above.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Launch the app:

    ```bash
    streamlit run streamlit_app.py
    ```
