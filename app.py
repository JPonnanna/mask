import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Load the model once
maskmodel = load_model('epoch130black.h5')

# Color map for decoding the mask
COLOR_MAP = {
    0: [0, 0, 0],         # background - black
    1: [0, 0, 255],       # ripe - blue
    2: [0, 255, 0],       # unripe - green
    3: [255, 255, 0],     # semi-ripe - yellow
}

# Decode model output to RGB mask
def decode_mask(mask):
    if mask.ndim == 3 and mask.shape[-1] == 4:
        mask = np.argmax(mask, axis=-1)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        color_mask[mask == class_idx] = color
    return color_mask

# Color match helper
def color_match(mask, color, tolerance=10):
    return np.all(np.abs(mask - color) <= tolerance, axis=-1)

# Generate individual masks for ripe, unripe, and semi-ripe
def generate_class_masks(mask):
    ripe = np.zeros_like(mask)
    semi = np.zeros_like(mask)
    unripe = np.zeros_like(mask)

    ripe[color_match(mask, [0, 0, 255])] = [0, 0, 255]
    semi[color_match(mask, [255, 255, 0])] = [255, 255, 0]
    unripe[color_match(mask, [0, 255, 0])] = [0, 255, 0]

    return ripe, semi, unripe

# Count tomatoes using watershed
def count_tomatoes(mask):
    rgb = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (3, 3), 0)

    local_max_coords = peak_local_max(
        distance,
        min_distance=15,
        labels=binary,
        footprint=np.ones((3, 3))
    )
    local_max_mask = np.zeros_like(distance, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    markers = ndi.label(local_max_mask)[0]
    labels = watershed(-distance, markers, mask=binary)

    count = 0
    for l in range(1, np.max(labels) + 1):
        if np.sum(labels == l) >= 200:
            count += 1
    return count

# Predict mask using model
def predict_mask(image_np):
    resized = cv2.resize(image_np, (256, 256))
    norm = resized / 255.0
    input_tensor = np.expand_dims(norm, axis=0)
    prediction = maskmodel.predict(input_tensor)[0]
    decoded = decode_mask(prediction)
    return decoded

# Streamlit UI
st.title("🍅 Tomato Ripeness Detector & Counter")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Predict
    predicted_mask = predict_mask(image_np)

    # Display mask
    st.subheader("Predicted Mask")
    st.image(predicted_mask, use_column_width=True)

    # Separate masks
    ripe, semi, unripe = generate_class_masks(predicted_mask)

    # Display separated masks
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(ripe, caption="Ripe")
    with col2:
        st.image(semi, caption="Semi-Ripe")
    with col3:
        st.image(unripe, caption="Unripe")

    # Count each class
    ripe_count = count_tomatoes(ripe)
    semi_count = count_tomatoes(semi)
    unripe_count = count_tomatoes(unripe)

    st.markdown("### 🍅 Tomato Counts")
    st.write(f"✅ Ripe: **{ripe_count}**")
    st.write(f"🟡 Semi-Ripe: **{semi_count}**")
    st.write(f"🟢 Unripe: **{unripe_count}**")
