import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import yaml
import os
from datetime import datetime
from pathlib import Path
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="AI Safety Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Custom CSS Styling
# ------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #0d6efd;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: bold;
        color: #212529;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0d6efd;
        padding-bottom: 0.3rem;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Cached Functions
# ------------------------
@st.cache_resource
def load_model():
    model_path = "/Users/syedasif/duality_ai/runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return YOLO(model_path)

@st.cache_data
def load_config():
    config_path = "/Users/syedasif/duality_ai/Hackathon2_scripts/yolo_params.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

@st.cache_data
def load_training_results():
    results_path = "/Users/syedasif/duality_ai/runs/detect/train/results.csv"
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None

@st.cache_data
def get_class_names():
    config = load_config()
    return config.get('names', [
        'OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm',
        'SafetySwitchPanel', 'EmergencyPhone', 'EscapeHatch'
    ])

def run_inference(model, image, conf_threshold=0.5):
    results = model.predict(image, conf=conf_threshold, verbose=False)
    return results[0]

def create_detection_summary(detections, class_names):
    if detections.boxes is None or len(detections.boxes) == 0:
        return pd.DataFrame(columns=['Class', 'Detected Count', 'Status'])
    class_counts = {}
    for box in detections.boxes:
        cls_id = int(box.cls)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    summary_data = []
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        status = "✅" if count > 0 else "⚠️"
        summary_data.append({
            'Class': class_name,
            'Detected Count': count,
            'Status': status
        })
    return pd.DataFrame(summary_data)

def generate_pdf_report(detection_summary, model_metrics, detection_image_path=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=24,
        spaceAfter=30, alignment=TA_CENTER, textColor=colors.HexColor("#0d6efd")
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=16,
        spaceAfter=12, textColor=colors.HexColor("#0d6efd")
    )

    # Cover Page
    story.append(Paragraph("AI Safety Compliance Detection Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    project_info = [
        ["Project Title:", "AI Safety Compliance Detection System"],
        ["Team Name:", "Team Dominators"],
        ["Hackathon:", "AI Safety Compliance Challenge"],
        ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Model:", "YOLOv8 (Safety Equipment Detection)"]
    ]
    project_table = Table(project_info, colWidths=[2*inch, 4*inch])
    project_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (1, 0), (1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(project_table)
    story.append(Spacer(1, 0.4*inch))

    # Results
    story.append(Paragraph("Detection Results", heading_style))
    table_data = [['Class', 'Detected Count', 'Status']]
    for _, row in detection_summary.iterrows():
        table_data.append([row['Class'], str(row['Detected Count']), row['Status']])
    results_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))

    # Performance Metrics
    if model_metrics:
        story.append(Paragraph("Model Performance Metrics", heading_style))
        metrics_data = [
            ["Metric", "Value"],
            ["Precision", f"{model_metrics.get('precision', 'N/A'):.3f}"],
            ["Recall", f"{model_metrics.get('recall', 'N/A'):.3f}"],
            ["mAP@0.5", f"{model_metrics.get('map50', 'N/A'):.3f}"],
            ["mAP@0.5:0.95", f"{model_metrics.get('map50_95', 'N/A'):.3f}"]
        ]
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#198754")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)

    if detection_image_path and os.path.exists(detection_image_path):
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Sample Detection Result", heading_style))
        img = RLImage(detection_image_path, width=5*inch, height=3.5*inch)
        story.append(img)

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Generated by AI Safety Compliance Dashboard | Team Dominators",
        ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9, textColor=colors.grey)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------
# Main App
# ------------------------
def main():
    st.markdown('<h1 class="main-header">🛡️ AI Safety Compliance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Team Dominators | YOLOv8-powered Safety Detection</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        return
    class_names = get_class_names()
    training_results = load_training_results()

    # Sidebar Configuration
    st.sidebar.header("⚙️ Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    st.sidebar.header("📷 Input Options")
    use_camera = st.sidebar.radio("Choose input method:", ["Upload Image", "Use Camera"])

    if use_camera == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    else:
        uploaded_file = st.camera_input("Capture from Camera")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown('<h2 class="section-header">🔍 Detection Preview</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Detection Results")
            with st.spinner("Running detection..."):
                results = run_inference(model, image, conf_threshold)
                summary = create_detection_summary(results, class_names)
                annotated = results.plot()
                st.image(annotated, use_column_width=True)

        # Results
        st.markdown('<h2 class="section-header">📊 Detection Results</h2>', unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True)

        # Compliance Status
        missing = summary[summary['Detected Count'] == 0]
        if len(missing) > 0:
            st.warning(f"⚠️ Missing equipment: {', '.join(missing['Class'].tolist())}")
        else:
            st.success("✅ All required safety equipment detected.")

        # PDF Report Button
        st.markdown('<h2 class="section-header">📋 Generate Compliance Report</h2>', unsafe_allow_html=True)
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, "detection.jpg")
                cv2.imwrite(image_path, annotated)
                metrics = None
                if training_results is not None and not training_results.empty:
                    latest = training_results.iloc[-1]
                    metrics = {
                        'precision': latest['metrics/precision(B)'],
                        'recall': latest['metrics/recall(B)'],
                        'map50': latest['metrics/mAP50(B)'],
                        'map50_95': latest['metrics/mAP50-95(B)']
                    }
                pdf = generate_pdf_report(summary, metrics, image_path)
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf.getvalue(),
                    file_name=f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

    st.markdown('<div class="footer">🛡️ AI Safety Compliance Dashboard | Team Dominators</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()