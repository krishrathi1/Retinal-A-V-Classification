"""
Streamlit App for Retinal A/V Classification
Upload a retinal image and get artery/vein segmentation
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title="Retinal A/V Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------
# Model Architecture (same as training)
# --------------------
class AttentionGate(nn.Module):
    """Attention gate for better feature selection"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_residual=False):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if self.use_residual:
            self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_residual:
            out = out + self.residual(x)
        return out

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=3, base=48):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base, use_residual=False)
        self.enc2 = DoubleConv(base, base*2, use_residual=True)
        self.enc3 = DoubleConv(base*2, base*4, use_residual=True)
        self.enc4 = DoubleConv(base*4, base*8, use_residual=True)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bott = DoubleConv(base*8, base*16, use_residual=True)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.att4 = AttentionGate(F_g=base*8, F_l=base*8, F_int=base*4)
        self.dec4 = DoubleConv(base*16, base*8, use_residual=True)
        
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.att3 = AttentionGate(F_g=base*4, F_l=base*4, F_int=base*2)
        self.dec3 = DoubleConv(base*8, base*4, use_residual=True)
        
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.att2 = AttentionGate(F_g=base*2, F_l=base*2, F_int=base)
        self.dec2 = DoubleConv(base*4, base*2, use_residual=True)
        
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.att1 = AttentionGate(F_g=base, F_l=base, F_int=base//2)
        self.dec1 = DoubleConv(base*2, base, use_residual=True)

        # Final classification head with dropout
        self.head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(base, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bott(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        return self.head(d1)

# --------------------
# Preprocessing Functions
# --------------------
def apply_green_CLAHE(img_rgb: np.ndarray) -> np.ndarray:
    """Enhance small arteries via CLAHE on G channel."""
    g = img_rgb[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ge = clahe.apply(g)
    out = img_rgb.copy()
    out[:,:,1] = ge
    return out

def circle_crop_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Mask out black surround to avoid red rim."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, circ = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    circ = cv2.medianBlur(circ, 31)
    return circ

def preprocess_image(image_rgb: np.ndarray, target_size=512):
    """Preprocess image for model input."""
    # Apply CLAHE enhancement
    img_enhanced = apply_green_CLAHE(image_rgb)
    
    # Resize
    img_resized = cv2.resize(img_enhanced, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Get circle mask
    circle_mask = circle_crop_mask(img_resized)
    
    # Convert to tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, circle_mask, img_resized

def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """Convert mask {0,1,2} to RGB (black, red, blue)."""
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[mask_np == 1] = (255, 0, 0)   # artery red
    out[mask_np == 2] = (0, 0, 255)   # vein blue
    return out

# --------------------
# Model Loading
# --------------------
@st.cache_resource
def load_model(model_path, device):
    """Load the trained model."""
    model = UNet(in_ch=3, num_classes=3, base=48)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

# --------------------
# Main App
# --------------------
def main():
    # Title and description
    st.title("👁️ Retinal Artery/Vein Classification")
    st.markdown("""
    Upload a retinal fundus image to get automatic artery and vein segmentation.
    - **Red**: Arteries
    - **Blue**: Veins
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"**Device**: {device.upper()}")
        
        # Model path
        model_path = st.text_input(
            "Model Path",
            value="models/unet_av.pth",
            help="Path to the trained model file"
        )
        
        st.markdown("---")
        st.markdown("""
        ### About
        This app uses a U-Net model with attention gates to segment 
        arteries and veins in retinal fundus images.
        
        **Model Features:**
        - Attention-based U-Net architecture
        - CLAHE preprocessing for enhanced vessel visibility
        - 3-class segmentation (Background, Artery, Vein)
        """)
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: `{model_path}`")
        st.info("Please ensure the model file exists at the specified path.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model, checkpoint = load_model(model_path, device)
        
        st.success(f"✅ Model loaded successfully!")
        
        # Display model info in expander
        with st.expander("📊 Model Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epoch", checkpoint.get('epoch', 'N/A'))
            with col2:
                st.metric("Val Loss", f"{checkpoint.get('val_loss', 0):.4f}")
            with col3:
                st.metric("Image Size", f"{checkpoint.get('img_size', 512)}×{checkpoint.get('img_size', 512)}")
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # File uploader
    st.markdown("---")
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a retinal image...",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload a retinal fundus image for A/V classification"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Display original image
        st.markdown("---")
        st.header("🖼️ Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
            st.caption(f"Size: {image_rgb.shape[1]}×{image_rgb.shape[0]} pixels")
        
        # Process image
        with st.spinner("Processing image..."):
            try:
                # Preprocess
                img_tensor, circle_mask, img_resized = preprocess_image(
                    image_rgb, 
                    target_size=checkpoint.get('img_size', 512)
                )
                
                # Predict
                with torch.no_grad():
                    img_tensor = img_tensor.to(device)
                    logits = model(img_tensor)
                    pred_mask = logits.softmax(1).argmax(1)[0].cpu().numpy()
                
                # Apply circle mask to remove border artifacts
                pred_mask[circle_mask == 0] = 0
                
                # Create visualization
                colored_mask = colorize_mask(pred_mask)
                
                # Resize back to original size for better visualization
                original_h, original_w = image_rgb.shape[:2]
                colored_mask_full = cv2.resize(colored_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                
                with col2:
                    st.subheader("Segmentation Result")
                    st.image(colored_mask_full, use_container_width=True)
                    st.caption("Segmentation mask (Red: Arteries, Blue: Veins)")
                
                # Download button
                st.markdown("---")
                
                # Convert mask to bytes
                mask_bgr = cv2.cvtColor(colored_mask_full, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.png', mask_bgr)
                mask_bytes = buffer.tobytes()
                
                st.download_button(
                    label="💾 Download Segmentation Mask",
                    data=mask_bytes,
                    file_name="av_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)
    
    else:
        st.info("👆 Please upload a retinal image to get started.")
        
        # Show example
        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Upload a retinal fundus image using the file uploader above
        2. Wait for the model to process the image
        3. View the segmentation results with arteries (red) and veins (blue)
        4. Download the segmentation mask using the download button
        
        ### 🎯 Supported Formats
        - PNG, JPG, JPEG, TIF, TIFF
        - Color retinal fundus images
        - Any resolution (will be resized to 512×512 for processing)
        """)

if __name__ == "__main__":
    main()

