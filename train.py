# train.py  — Enhanced U-Net for 3-class AV segmentation (BG=0, Artery=1, Vein=2)
# Folders (Kaggle):
#   /kaggle/input/av-classification/data/images (.tif/.png/.jpg)
#   /kaggle/input/av-classification/data/artery (.png)
#   /kaggle/input/av-classification/data/vein (.png)
#   /kaggle/input/av-classification/data/av (.png colored, optional)
#
# Outputs (in working dir):
#   models/unet_av.pth, outputs_train_debug/debug_*.png

import os, glob, cv2, math, random, numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# --------------------
# Config (UPDATED FOR KAGGLE DATASET)
# --------------------
# Point directly to your Kaggle dataset
DATA_DIR   = "/kaggle/input/av-classification/data"

IMG_DIR    = os.path.join(DATA_DIR, "images")
ART_DIR    = os.path.join(DATA_DIR, "artery")
VEIN_DIR   = os.path.join(DATA_DIR, "vein")
AV_DIR     = os.path.join(DATA_DIR, "av")       # optional colored AV masks

# Save models & debug outputs in the working directory
MODEL_DIR  = "models"
OUT_DIR    = "outputs_train_debug"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE   = 512
BATCH_SIZE = 2
EPOCHS     = 75  # train for 75 epochs
LR         = 1e-3
NUM_CLASSES= 3   # 0=bg, 1=artery, 2=vein

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------
# Utils
# --------------------
def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_gray(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m

def apply_green_CLAHE(img_rgb: np.ndarray) -> np.ndarray:
    """Enhance small arteries via CLAHE on G channel."""
    g = img_rgb[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ge = clahe.apply(g)
    out = img_rgb.copy()
    out[:,:,1] = ge
    return out

def build_mask_from_av(av_img: np.ndarray) -> np.ndarray:
    """av colored: red=artery, blue=vein, green=crossing(ignored). Return mask {0,1,2}."""
    R, G, B = av_img[:,:,0], av_img[:,:,1], av_img[:,:,2]
    artery = (R > 160) & (B < 100)
    vein   = (B > 160) & (R < 100)
    mask = np.zeros(artery.shape, dtype=np.uint8)
    mask[artery] = 1
    mask[vein]   = 2
    return mask

def compose_mask(artery_path: Optional[str], vein_path: Optional[str],
                 av_path: Optional[str], shape: Tuple[int,int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if av_path and os.path.exists(av_path):
        av = read_rgb(av_path)
        if (av.shape[0], av.shape[1]) != (H, W):
            av = cv2.resize(av, (W, H), interpolation=cv2.INTER_NEAREST)
        return build_mask_from_av(av)
    if artery_path and os.path.exists(artery_path):
        a = read_gray(artery_path)
        if a.shape != (H, W): a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[a > 127] = 1
    if vein_path and os.path.exists(vein_path):
        v = read_gray(vein_path)
        if v.shape != (H, W): v = cv2.resize(v, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[v > 127] = 2
    return mask

def circle_crop_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Mask out black surround to avoid red rim."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, circ = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    circ = cv2.medianBlur(circ, 31)
    return circ

# --------------------
# Dataset
# --------------------
class AVDataset(Dataset):
    def __init__(self, img_dir, art_dir, vein_dir, av_dir, split="train", img_size=512):
        # Support multiple extensions: .tif/.png/.jpg/.jpeg
        exts = ("*.tif", "*.png", "*.jpg", "*.jpeg")
        img_paths = []
        for e in exts:
            img_paths.extend(glob.glob(os.path.join(img_dir, e)))
        self.img_paths = sorted(img_paths)

        assert len(self.img_paths)>0, f"No images found in {img_dir} with {exts}"
        rng = np.random.RandomState(SEED)
        idx = np.arange(len(self.img_paths))
        rng.shuffle(idx)
        split_at = int(0.8*len(idx))
        self.ids = idx[:split_at] if split=="train" else idx[split_at:]

        self.all = self.img_paths
        self.art_dir  = art_dir
        self.vein_dir = vein_dir
        self.av_dir   = av_dir
        self.img_size = img_size
        self.split = split

    def __len__(self):
        return len(self.ids)

    def _augment(self, img, mask):
        if self.split == "train":
            # Geometric augmentations
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=0))
                mask = np.ascontiguousarray(np.flip(mask, axis=0))
            if random.random() < 0.3:
                k = random.choice([1,2,3])
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            
            # Enhanced color augmentations
            img_f = img.astype(np.float32)
            # Brightness/Contrast
            if random.random() < 0.4:
                gain = 0.8 + random.random()*0.6   # 0.8 - 1.4
                img_f = np.clip(img_f * gain, 0, 255)
            # Gamma correction
            if random.random() < 0.3:
                gamma = 0.7 + random.random()*0.6  # 0.7 - 1.3
                img_f = np.clip((img_f/255.0) ** gamma * 255.0, 0, 255)
            # Color jitter (per channel)
            if random.random() < 0.3:
                for c in range(3):
                    jitter = 0.9 + random.random()*0.2  # 0.9 - 1.1
                    img_f[:,:,c] = np.clip(img_f[:,:,c] * jitter, 0, 255)
            # Gaussian noise
            if random.random() < 0.2:
                noise = np.random.normal(0, 5, img_f.shape).astype(np.float32)
                img_f = np.clip(img_f + noise, 0, 255)
            
            img = img_f.astype(np.uint8)
        return img, mask

    def __getitem__(self, i):
        path = self.all[self.ids[i]]
        base = Path(path).stem
        img  = read_rgb(path)

        # preprocess: CLAHE on green channel
        img = apply_green_CLAHE(img)

        # resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        a_path = os.path.join(self.art_dir,  base + ".png")
        v_path = os.path.join(self.vein_dir, base + ".png")
        av_path= os.path.join(self.av_dir,   base + ".png") if os.path.isdir(self.av_dir) else None

        mask = compose_mask(a_path, v_path, av_path, (img.shape[0], img.shape[1]))

        # suppress border
        circ = circle_crop_mask(img)
        mask[circ==0] = 0

        img, mask = self._augment(img, mask)

        # to tensors
        img  = TF.to_tensor(img)            # [0,1], CxHxW
        mask = torch.from_numpy(mask).long()# HxW
        return img, mask, base

# --------------------
# Enhanced U-Net with Attention and Residual Connections
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
    def __init__(self, in_ch=3, num_classes=3, base=48):  # Increased base from 32 to 48
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
# Enhanced Loss Functions
# --------------------
class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        total = 0.0
        for c in range(self.num_classes):
            pc = probs[:, c]
            tc = (targets == c).float()
            inter = (pc * tc).sum(dim=(1,2))
            denom = pc.sum(dim=(1,2)) + tc.sum(dim=(1,2)) + self.smooth
            dice = (2*inter + self.smooth) / denom
            loss_c = 1 - dice
            if self.weight is not None:
                loss_c = loss_c * self.weight[c]
            total += loss_c.mean()
        return total / self.num_classes

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TverskyLoss(nn.Module):
    """Tversky Loss - better for imbalanced segmentation"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # false positive weight
        self.beta = beta    # false negative weight
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        total = 0.0
        for c in range(1, logits.size(1)):  # Skip background
            pc = probs[:, c]
            tc = (targets == c).float()
            tp = (pc * tc).sum(dim=(1,2))
            fp = (pc * (1 - tc)).sum(dim=(1,2))
            fn = ((1 - pc) * tc).sum(dim=(1,2))
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            total += (1 - tversky).mean()
        return total / (logits.size(1) - 1)

# --------------------
# Training / Validation
# --------------------
def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """mask {0,1,2} -> RGB (black, red, blue)."""
    h,w = mask_np.shape
    out = np.zeros((h,w,3), np.uint8)
    out[mask_np==1] = (255,0,0)   # artery red
    out[mask_np==2] = (0,0,255)   # vein blue
    return out

def train():
    ds_tr = AVDataset(IMG_DIR, ART_DIR, VEIN_DIR, AV_DIR, split="train", img_size=IMG_SIZE)
    ds_va = AVDataset(IMG_DIR, ART_DIR, VEIN_DIR, AV_DIR, split="val",   img_size=IMG_SIZE)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=1,           shuffle=False, num_workers=0)

    # Improved class weights - balanced for better learning
    class_w = torch.tensor([0.3, 1.5, 0.8], dtype=torch.float32, device=DEVICE)
    focal_alpha = torch.tensor([0.3, 1.5, 0.8], dtype=torch.float32, device=DEVICE)

    base_model = UNet(in_ch=3, num_classes=NUM_CLASSES, base=48)
    n_params = sum(p.numel() for p in base_model.parameters())/1e6

    # -------- MULTI-GPU SUPPORT (uses all visible GPUs) --------
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model = model.to(DEVICE)

    # Combined loss functions
    ce = nn.CrossEntropyLoss(weight=class_w)
    dice = DiceLoss(num_classes=NUM_CLASSES, weight=class_w)
    focal = FocalLoss(alpha=focal_alpha, gamma=2.0)
    tversky = TverskyLoss(alpha=0.7, beta=0.3)
    
    # Optimizer with gradient clipping
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Cosine annealing with warm restarts
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val = math.inf
    patience_counter = 0
    max_patience = 10
    
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {n_params:.2f}M")
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for img, msk, _ in dl_tr:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            logits = model(img)
            
            # Combined loss: CE + Dice + Focal + Tversky
            loss = (0.3 * ce(logits, msk) + 
                    0.3 * dice(logits, msk) + 
                    0.2 * focal(logits, msk) + 
                    0.2 * tversky(logits, msk))
            
            optim.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tr_loss += loss.item() * img.size(0)
        tr_loss /= len(ds_tr)

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for img, msk, base in dl_va:
                img, msk = img.to(DEVICE), msk.to(DEVICE)
                logits = model(img)
                loss = (0.3 * ce(logits, msk) + 
                        0.3 * dice(logits, msk) + 
                        0.2 * focal(logits, msk) + 
                        0.2 * tversky(logits, msk))
                va_loss += loss.item()
            va_loss /= max(1, len(ds_va))

        sched.step()
        lr_now = optim.param_groups[0]['lr']
        print(f"[Epoch {epoch:02d}/{EPOCHS}] train={tr_loss:.4f}  val={va_loss:.4f}  lr={lr_now:.2e}")

        # Save debug prediction periodically
        if epoch % 5 == 0 or epoch == EPOCHS:
            try:
                img, msk, base = next(iter(dl_va))
                img = img.to(DEVICE)
                with torch.no_grad():
                    pr = model(img).softmax(1).argmax(1)[0].cpu().numpy()
                vis = colorize_mask(pr)
                cv2.imwrite(os.path.join(OUT_DIR, f"debug_{epoch:02d}_{base[0]}.png"),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            except StopIteration:
                pass

        # Early stopping with patience
        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0

            # Handle DataParallel when saving
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            path = os.path.join(MODEL_DIR, "unet_av.pth")
            torch.save({"model": state_dict,
                        "img_size": IMG_SIZE,
                        "epoch": epoch,
                        "val_loss": best_val}, path)
            print(f"  -> saved best: {path} (val={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    train()
