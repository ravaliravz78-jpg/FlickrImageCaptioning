import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

st.title("🧠 AI-Based Image Caption Generator")
st.write("Upload an image and let the CNN-LSTM model generate a caption.")

# ----------------------------
# Load Model and Files
# ----------------------------
@st.cache_resource
def load_caption_model():
    model = load_model("caption_model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

@st.cache_resource
def load_max_length():
    with open("max_length.pkl", "rb") as f:
        max_length = pickle.load(f)
    return max_length

@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

model = load_caption_model()
tokenizer = load_tokenizer()
max_length = load_max_length()
feature_extractor = load_feature_extractor()

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(img):
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features

# ----------------------------
# Word Mapping
# ----------------------------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ----------------------------
# Caption Generation (Greedy)
# ----------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    final_caption = in_text.replace("startseq", "").replace("endseq", "")
    return final_caption.strip()

# ----------------------------
# UI Section
# ----------------------------
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption..."):
            features = extract_features(img)
            caption = generate_caption(model, tokenizer, features, max_length)
        st.success("Caption Generated!")
        st.markdown(f"### 📝 {caption}")