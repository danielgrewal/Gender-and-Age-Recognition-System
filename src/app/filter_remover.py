import cv2
import torch
import numpy as np
import sys
from torch import nn

#####################################################
# 1) Define the EXACT same U-Net architecture used
#    in training (1 input channel -> 2 output channels).
#####################################################
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv0 = self.double_conv(1, 64)
        self.pool0 = nn.MaxPool2d(2)
        self.enc_conv1 = self.double_conv(64, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = self.double_conv(128, 256)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.double_conv(256, 512)

        # Decoder
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(256, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = self.double_conv(128, 64)

        # Final: 2 channels (ab)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e0 = self.enc_conv0(x)
        p0 = self.pool0(e0)
        e1 = self.enc_conv1(p0)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec_conv2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec_conv1(cat1)

        u0 = self.up0(d1)
        cat0 = torch.cat([u0, e0], dim=1)
        d0 = self.dec_conv0(cat0)

        out = self.out_conv(d0)
        return out


#####################################################
# 2) Default / Forced Options (no CLI arguments)
#####################################################
MODEL_PATH    = "colorization_unet.pt"                  # Pretrained model file
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HAAR_CASCADE  = "haarcascade_frontalface_default.xml"    # Haar cascade path
FACE_SIZE     = 200                                      # We trained on 200×200

#####################################################
# 3) Load the pretrained model
#####################################################
def load_model(model_path, device='cpu'):
    model = UNet()
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


#####################################################
# 4) Colorization helper (using True Lab L)
#####################################################
def colorize_face_lab(face_bgr, model, device='cpu'):
    """
    face_bgr:   np.array (H,W,3) in BGR, range [0..255], shape (FACE_SIZE, FACE_SIZE)
    model:      the trained U-Net
    Returns:    colorized face (H,W,3) BGR, [0..255]
    Pipeline:
      1) face_bgr -> Lab (OpenCV)
      2) scale L to [0..1], ab to [-1..1]
      3) feed L to model -> pred_ab
      4) combine L, pred_ab -> Lab -> BGR
    """
    h, w, _ = face_bgr.shape

    # 1) Convert BGR -> RGB -> Lab
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    L, a, b = cv2.split(face_lab)   # L in [0..100], a,b in [-128..127]

    # 2) Scale L to [0..1], a,b to [-1..1]
    L_norm = L / 100.0
    a_norm = (a - 128.0) / 128.0
    b_norm = (b - 128.0) / 128.0

    # Convert L to torch tensor [1,1,H,W]
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)  # shape [1,1,h,w]

    # 3) Predict ab
    with torch.no_grad():
        pred_ab = model(L_tensor)  # shape [1,2,h,w]

    pred_ab_np = pred_ab.squeeze(0).cpu().numpy()  # [2,h,w]
    # Convert from [-1..1] -> [-128..127]
    a_pred = (pred_ab_np[0] * 128.0) + 128.0
    b_pred = (pred_ab_np[1] * 128.0) + 128.0

    # 4) Combine with original L (in [0..100]) for better correctness
    #    rather than the naive grayscale approach.
    # L was in [0..100], so let's retrieve that from L_norm
    # (or use L directly from face_lab).
    L_scaled_2d = L_norm * 100.0  # back to [0..100]

    lab_pred = np.zeros((h, w, 3), dtype=np.float32)
    lab_pred[..., 0] = L_scaled_2d
    lab_pred[..., 1] = a_pred
    lab_pred[..., 2] = b_pred

    # Convert Lab -> BGR
    lab_uint8 = lab_pred.astype(np.uint8)
    face_bgr_colorized = cv2.cvtColor(lab_uint8, cv2.COLOR_Lab2BGR)

    return face_bgr_colorized


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")

    # Load the pretrained model.
    model = load_model(MODEL_PATH, device=DEVICE)

    # Initialize Haar cascade.
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    if face_cascade.empty():
        print("Error loading Haar cascade.")
        sys.exit(1)

    # Load your JPG image (keep it as uint8).
    image_path = "media/image_gs.jpg"  # Replace with your actual image path
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        print(f"Error loading image from {image_path}")
        sys.exit(1)

    display_frame = frame_bgr.copy()
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces.
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_roi_bgr = frame_bgr[y:y+h, x:x+w]
        face_resized_bgr = cv2.resize(face_roi_bgr, (FACE_SIZE, FACE_SIZE))
        colorized_bgr = colorize_face_lab(face_resized_bgr, model, DEVICE)
        colorized_bgr = cv2.resize(colorized_bgr, (w, h))
        display_frame[y:y+h, x:x+w] = colorized_bgr
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Colorized Image", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
