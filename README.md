# Retinal A/V Classification Streamlit App

A web application for automatic artery and vein segmentation in retinal fundus images using a U-Net model with attention gates.

## Features

- 🖼️ Upload retinal fundus images in various formats (PNG, JPG, TIFF)
- 🔴🔵 Automatic segmentation of arteries (red) and veins (blue)
- 📊 Statistical analysis including A/V ratio
- 💾 Download segmentation results
- ⚙️ Adjustable overlay transparency
- 🚀 GPU acceleration support (if available)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the trained model file at `models/unet_av.pth`

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

4. Upload a retinal image and view the segmentation results!

## Model Information

The model uses an enhanced U-Net architecture with:
- Attention gates for better feature selection
- Residual connections for improved gradient flow
- CLAHE preprocessing on the green channel for enhanced vessel visibility
- 3-class segmentation: Background (0), Artery (1), Vein (2)

## File Structure

```
working/
├── app.py                  # Streamlit application
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── models/
    └── unet_av.pth       # Trained model weights
```

## A/V Ratio Interpretation

The Artery/Vein (A/V) ratio is an important clinical metric:
- **Normal**: ~0.67 (2:3 ratio)
- **Higher values**: May indicate arterial narrowing (hypertensive retinopathy)
- **Lower values**: May indicate venous dilation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- OpenCV 4.8+
- CUDA (optional, for GPU acceleration)

## Troubleshooting

### Model not found
Make sure the model file exists at `models/unet_av.pth`. You can change the path in the sidebar settings.

### Out of memory error
If you encounter GPU memory issues, the app will automatically fall back to CPU processing.

### Image format not supported
Supported formats: PNG, JPG, JPEG, TIF, TIFF. Make sure your image is in one of these formats.

## License

This project is for educational and research purposes.

