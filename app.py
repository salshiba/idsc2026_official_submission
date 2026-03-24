from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# Image processing imports
from skimage.filters import frangi
from skimage.filters.rank import entropy
from skimage.morphology import disk

# TDA imports
from gtda.homology import CubicalPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy, BettiCurve, PersistenceImage

# ==========================================
# 1. PAGE SETUP & GLOBAL VARIABLES
# ==========================================
st.set_page_config(page_title="Topo-Vision AI", layout="wide")
DECISION_BOUNDARY = 0.6


# Initialize TDA Extractors globally
@st.cache_resource
def initialize_tda_extractors():
    """Initialize TDA extractors once and cache them."""
    cubical = CubicalPersistence(homology_dimensions=[0, 1])
    metrics = ["bottleneck", "wasserstein", "betti", "landscape", "heat"]
    extractors = {f"amp_{m}": Amplitude(metric=m) for m in metrics}
    extractors["entropy"] = PersistenceEntropy()
    extractors["betti_curve"] = BettiCurve(n_bins=10)
    extractors["pi"] = PersistenceImage(sigma=0.1, n_bins=5)
    return cubical, extractors


cubical, extractors = initialize_tda_extractors()


# ==========================================
# 2. LOAD MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    """Loads the Random Forest and the YOLO model."""
    rf = joblib.load("models/our_tda_rf_model.joblib")
    yolo = YOLO("models/our_yolo_model.pt")
    return rf, yolo


rf_model, yolo_cropper = load_models()


# ==========================================
# 3. PREPROCESSING & EXTRACTION FUNCTIONS
# ==========================================
def preprocess_for_tda_vascular(img_bgr):
    _, green, _ = cv2.split(img_bgr)
    g_blur = cv2.GaussianBlur(green, (5, 5), 0)
    frangi_map = frangi(g_blur, sigmas=range(2, 6, 1), black_ridges=True)
    norm = cv2.normalize(frangi_map, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    small = cv2.resize(norm, (64, 64), interpolation=cv2.INTER_AREA)
    return ((1.0 - small) * 255).astype(np.uint8)


def preprocess_for_tda_cup(img_bgr):
    _, _, red = cv2.split(img_bgr)
    r_blur = cv2.GaussianBlur(red, (21, 21), 0)
    norm = cv2.normalize(r_blur, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    small = cv2.resize(norm, (64, 64), interpolation=cv2.INTER_AREA)
    return ((1.0 - small) * 255).astype(np.uint8)


def preprocess_for_tda_texture(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    texture_map = entropy(gray, disk(5))
    norm = cv2.normalize(texture_map, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    small = cv2.resize(norm, (64, 64), interpolation=cv2.INTER_AREA)
    return ((1.0 - small) * 255).astype(np.uint8)


def extract_features_from_landscape(img_array, prefix):
    """Takes a processed 64x64 numpy array and extracts TDA features."""
    diagrams = cubical.fit_transform(img_array[None, :, :])

    features = {}
    for name, extractor in extractors.items():
        values = extractor.fit_transform(diagrams)[0]
        if values.ndim == 1 and len(values) == 2:
            features[f"{prefix}_{name}_H0"] = values[0]
            features[f"{prefix}_{name}_H1"] = values[1]
        elif values.ndim == 2 and len(values) == 2:
            for bin_idx in range(values.shape[1]):
                features[f"{prefix}_{name}_H0_bin{bin_idx}"] = values[0][bin_idx]
                features[f"{prefix}_{name}_H1_bin{bin_idx}"] = values[1][bin_idx]
        elif values.ndim == 3 and len(values) == 2:
            h0_flat = values[0].flatten()
            h1_flat = values[1].flatten()
            for px_idx in range(len(h0_flat)):
                features[f"{prefix}_{name}_H0_px{px_idx}"] = h0_flat[px_idx]
                features[f"{prefix}_{name}_H1_px{px_idx}"] = h1_flat[px_idx]
    return features


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("👁️ TopoMedic AI: Topological Data Driven Diagnostics")
st.markdown("""
Using image processing and topological representations, this tool isolates the biological structure of the eye. 
By measuring the topological complexity of the **Optic Cup**, **Vascular Bayoneting**, and **Peripapillary Atrophy (PPA)**, 
we provide an objective, mathematically explainable Glaucoma diagnosis.
""")

uploaded_file = st.file_uploader(
    "Upload image of Optic Disc", type=["jpg", "png", "jpeg"]
)

st.markdown("### 📈 Quality Score")
scan_quality = st.slider(
    "Please choose correct quality score value:",
    min_value=1,
    max_value=10,
    value=5,
    help="Higher values indicate sharper focus and better illumination.",
)

if scan_quality < 5:
    st.warning(
        "⚠️ **Low Quality Scan:** AI confidence may be degraded. Clinical correlation strongly advised."
    )

analyze_button = st.button("Analyze Image", type="primary", width="stretch")

if uploaded_file is not None and analyze_button:
    st.markdown("---")

    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    # Yolo cropper
    results = yolo_cropper(img_bgr, verbose=False)
    if not results or len(results[0].boxes) == 0:
        st.error("❌ No object detected in the image. Please upload a different image.")
        st.stop()

    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    padding = 20
    y1 = max(0, y1 - padding)
    y2 = min(img_bgr.shape[0], y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(img_bgr.shape[1], x2 + padding)
    roi_img = img_bgr[y1:y2, x1:x2]

    # Apply processing
    annotated_frame = results[0].plot()
    lab = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    img_rgb_enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Uploaded Scan")
        st.image(img_rgb_enhanced, width="stretch")

    with col2:
        with st.spinner("Extracting Topological Landscapes & Betti Curves..."):
            # 1. Preprocess
            vessel_img = preprocess_for_tda_vascular(roi_img)
            cup_img = preprocess_for_tda_cup(roi_img)
            texture_img = preprocess_for_tda_texture(roi_img)

            # 2. Extract Features
            patient_features = {}
            patient_features.update(
                extract_features_from_landscape(vessel_img, "vessel")
            )
            patient_features.update(extract_features_from_landscape(cup_img, "cup"))
            patient_features.update(
                extract_features_from_landscape(texture_img, "texture")
            )

            # 3. Add Quality Score from the slider
            patient_features["Quality Score"] = scan_quality

            # 4. Prepare DataFrame for Scikit-Learn
            df_patient = pd.DataFrame([patient_features])
            df_patient = df_patient[rf_model.feature_names_in_]

            # 5. Predict
            probabilities = rf_model.predict_proba(df_patient)[0]
            prob_healthy = probabilities[0]
            prob_glaucoma = probabilities[1]

            # Display Prediction
            st.markdown("### 📊 AI Diagnosis")
            st.markdown(f"**Glaucoma Risk Score:** {prob_glaucoma * 100:.1f}%")
            st.caption(f"Decision threshold: {DECISION_BOUNDARY * 100:.1f}%")
            st.progress(float(prob_glaucoma))

            if prob_glaucoma > DECISION_BOUNDARY:
                st.error("🩺 **VERDICT:** High Risk (GON+)")
            else:
                st.success("🩺 **VERDICT:** Low Risk (GON-)")

    st.markdown("---")
    st.markdown("### 🧠 AI Decision Breakdown")
    st.markdown(
        "This SHAP chart shows how much each feature category contributed to the final Glaucoma risk score."
    )

    with st.spinner("Calculating Topological Feature Importance..."):
        # Calculate SHAP values for the Positive Class (Glaucoma)
        explainer = shap.TreeExplainer(rf_model)
        shap_values_obj = explainer(df_patient)[:, :, 1]

        # Define mapping based on your TDA extractors
        prefix_map = {
            "vessel_": "Vascular Features",
            "cup_": "Optic Cup Features",
            "texture_": "Peripapillary Texture Features",
            "Quality Score": "Image Quality Score",
        }

        # Group the features
        category_map = {label: [] for label in prefix_map.values()}

        for col in df_patient.columns:
            for prefix, label in prefix_map.items():
                if col.startswith(prefix):
                    category_map[label].append(col)
                    break

        # 3. Aggregate the SHAP values
        group_names = list(category_map.keys())
        aggregated_values = []

        for cat in group_names:
            indices = [
                df_patient.columns.get_loc(f)
                for f in category_map[cat]
                if f in df_patient.columns
            ]
            if len(indices) > 0:
                cat_sum = shap_values_obj.values[:, indices].sum(axis=1)
            else:
                cat_sum = np.array([0.0])
            aggregated_values.append(cat_sum)

        val_1d = np.array(aggregated_values).flatten()
        bv = shap_values_obj.base_values[0]
        data_1d = np.full(len(val_1d), "")

        final_explanation = shap.Explanation(
            values=val_1d, base_values=bv, data=data_1d, feature_names=group_names
        )
        max_idx = np.argmax(val_1d)
        highest_risk_factor = group_names[max_idx]
        highest_risk_value = val_1d[max_idx]

        # Find the category with the biggest negative impact
        min_idx = np.argmin(val_1d)
        highest_protective_factor = group_names[min_idx]
        highest_protective_value = val_1d[min_idx]

        st.markdown("### 📝 Clinical AI Summary")

        if prob_glaucoma > DECISION_BOUNDARY:
            st.warning(
                f"The AI flagged this patient as **High Risk**. The primary topological driver for this decision was **{highest_risk_factor}**."
            )
            if highest_protective_value < 0:
                st.info(
                    f"Interestingly, the patient's **{highest_protective_factor}** showed healthy topological characteristics, but it was not enough to override the risk factors."
                )
        else:
            st.success(
                f"The AI flagged this patient as **Healthy**. The strongest evidence supporting this was the patient's **{highest_protective_factor}**."
            )
            if highest_risk_value > 0:
                st.info(
                    f"Note: The **{highest_risk_factor}** showed some abnormal topology, but the overall geometry remains within healthy AI parameters."
                )

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(final_explanation, show=False)
        st.pyplot(fig)
        plt.clf()
        plt.close(fig)

elif analyze_button and uploaded_file is None:
    st.error("Please upload an image first!")
