# Breast Cancer Classifier

Predict whether [histopathological][hp] images of breast tissue contain malignant or benign cancer.

Powered by [streamlit][st]. The data used to train the model can be found [here][data].

[hp]: https://en.wikipedia.org/wiki/Histopathology
[st]: https://streamlit.io/
[data]: https://www.kaggle.com/datasets/forderation/breakhis-400x

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
