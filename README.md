# AI Safety Compliance Dashboard

A comprehensive Streamlit dashboard for safety compliance detection using trained YOLOv8 models.

## Features

- **Real-time Detection**: Upload images and get instant safety equipment detection
- **Analytics Dashboard**: Visualize detection results with interactive charts
- **Compliance Reporting**: Generate professional PDF reports
- **Model Performance**: View training metrics and model performance
- **Safety Equipment Detection**: Detects 7 types of safety equipment:
  - Oxygen Tank
  - Nitrogen Tank
  - First Aid Box
  - Fire Alarm
  - Safety Switch Panel
  - Emergency Phone
  - Escape Hatch

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run safety_compliance_dashboard.py
```

## Usage

1. **Upload Image**: Use the sidebar to upload an image for detection
2. **Adjust Settings**: Set confidence threshold for detection
3. **View Results**: See detection results in the main area
4. **Analytics**: Explore detection statistics and model performance
5. **Generate Report**: Create PDF compliance reports

## Project Structure

- `safety_compliance_dashboard.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `runs/detect/train/weights/best.pt` - Trained YOLOv8 model
- `Hackathon2_scripts/yolo_params.yaml` - Model configuration
- `Hackathon2_scripts/classes.txt` - Class definitions

## Model Information

- **Architecture**: YOLOv8n
- **Training Epochs**: 25
- **Input Size**: 640x640
- **Classes**: 7 safety equipment types
- **Performance**: See training results in `runs/detect/train/results.csv`

## Dashboard Sections

1. **Detection Preview**: Side-by-side original and annotated images
2. **Detection Results**: Tabular summary of detected equipment
3. **Analytics Dashboard**: Interactive charts and visualizations
4. **Compliance Report**: PDF report generation

## Technical Details

- Built with Streamlit for web interface
- Uses Ultralytics YOLOv8 for object detection
- Plotly for interactive visualizations
- ReportLab for PDF generation
- OpenCV for image processing
- Pandas for data manipulation

## Team

Duality AI Team - AI Safety Compliance Challenge


