import streamlit as st
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

st.set_page_config(page_title="Infrastructure Surveillance", layout="centered")

st.title("🛰️ Critical Infrastructure Detection")
st.caption("RF-DETR • Drone & Aerial Surveillance")

@st.cache_resource
def load_model(dataset_dir, checkpoint):
    model = RFDETRBase()
    model.train(
        dataset_dir=dataset_dir,
        epochs=0,
        resume=checkpoint
    )
    return model

DATASET_DIR = "local_rfdetr_data_1"
CHECKPOINT = "checkpoints/checkpoint_best_ema.pth"

model = load_model(DATASET_DIR, CHECKPOINT)

uploaded = st.file_uploader("Upload aerial image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    detections = model.predict(image, threshold=0.25)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{cls}:{conf:.2f}"
        for cls, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(image.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    st.image(annotated, caption="Detected Infrastructure", use_column_width=True)
