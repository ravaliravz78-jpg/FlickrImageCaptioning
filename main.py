# =====================================================
# IMPORTS
# =====================================================
import os
import pickle
import numpy as np
from tqdm import tqdm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

from PIL import UnidentifiedImageError

# =====================================================
# LOAD CNN MODEL (Feature Extractor)
# =====================================================
cnn_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# =====================================================
# BATCH FEATURE EXTRACTION (FASTER)
# =====================================================
def extract_features_batch(directory, batch_size=32):
    features = {}
    batch_imgs, batch_names = [], []

    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        try:
            img = image.load_img(img_path, target_size=(299, 299))
            img = image.img_to_array(img)
            batch_imgs.append(img)
            batch_names.append(img_name)

            if len(batch_imgs) == batch_size:
                batch_array = preprocess_input(np.array(batch_imgs))
                batch_features = cnn_model.predict(batch_array, verbose=0)
                for name, feature in zip(batch_names, batch_features):
                    features[name] = feature
                batch_imgs, batch_names = [], []

        except UnidentifiedImageError:
            continue
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    # Process remaining images
    if batch_imgs:
        batch_array = preprocess_input(np.array(batch_imgs))
        batch_features = cnn_model.predict(batch_array, verbose=0)
        for name, feature in zip(batch_names, batch_features):
            features[name] = feature

    return features

# =====================================================
# LOAD OR EXTRACT FEATURES
# =====================================================
if os.path.exists("features.pkl"):
    print("Loading saved features...")
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    print("Features loaded successfully!")
else:
    print("Extracting features (batch mode)...")
    features = extract_features_batch("dataset/Images")
    with open("features.pkl", "wb") as f:
        pickle.dump(features, f)
    print("Features saved successfully!")

print("Total extracted features:", len(features))

# =====================================================
# LOAD CAPTIONS
# =====================================================
captions_dict = {}
captions_path = "dataset/captions.txt"

with open(captions_path, 'r', encoding='utf-8') as file:
    next(file)  # skip header
    for line in file:
        line = line.strip()
        if not line:
            continue
        image_name, caption = line.split(",", 1)
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)

print("Total images with captions:", len(captions_dict))

# =====================================================
# TOKENIZER AND MAX LENGTH
# =====================================================
all_captions = [cap for caps in captions_dict.values() for cap in caps]

if os.path.exists("tokenizer.pkl"):
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully!")
else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer created and saved successfully!")

if os.path.exists("max_length.pkl"):
    with open("max_length.pkl", "rb") as f:
        max_length = pickle.load(f)
    print("Max length loaded successfully!")
else:
    max_length = max(len(c.split()) for c in all_captions)
    with open("max_length.pkl", "wb") as f:
        pickle.dump(max_length, f)
    print("Max length computed and saved!")

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)
print("Max Caption Length:", max_length)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================
model = load_model("caption_model.h5")
print("Model loaded successfully!")

# =====================================================
# HELPER FUNCTION
# =====================================================
def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# =====================================================
# GREEDY SEARCH
# =====================================================
def generate_caption_greedy(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word(yhat, tokenizer)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return " ".join(in_text.split()[1:])

# =====================================================
# BEAM SEARCH
# =====================================================
def generate_caption_beam_search(model, tokenizer, photo, max_length, beam_width=3):
    start_seq = tokenizer.texts_to_sequences(["startseq"])[0]
    sequences = [[start_seq, 0.0]]
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo, padded], verbose=0)[0]
            top_indices = np.argsort(yhat)[-beam_width:]
            for idx in top_indices:
                prob = yhat[idx]
                candidate = seq + [idx]
                candidate_score = score - np.log(prob + 1e-10)
                all_candidates.append([candidate, candidate_score])
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
    best_seq = sequences[0][0]
    final_caption = []
    for idx in best_seq:
        word = index_to_word(idx, tokenizer)
        if word is None or word == "startseq":
            continue
        if word == "endseq":
            break
        final_caption.append(word)
    if len(final_caption) == 0:
        return generate_caption_greedy(model, tokenizer, photo, max_length)
    return " ".join(final_caption)

# =====================================================
# PREDICT CAPTION WRAPPER
# =====================================================
def predict_caption(model, tokenizer, photo, max_length, method="beam"):
    if method == "greedy":
        return generate_caption_greedy(model, tokenizer, photo, max_length)
    elif method == "beam":
        return generate_caption_beam_search(model, tokenizer, photo, max_length, beam_width=3)
    else:
        raise ValueError("Method must be 'greedy' or 'beam'")

# =====================================================
# GENERATE CAPTIONS FOR ALL IMAGES
# =====================================================
output_file = "captions_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for img_name, photo in tqdm(list(features.items())[:1500], desc="Generating captions"):
        caption = predict_caption(model, tokenizer, photo, max_length, method="greedy")
        f.write(f"{img_name}: {caption}\n")

print(f"\nAll captions saved to {output_file}")