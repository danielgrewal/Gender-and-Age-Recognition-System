import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "colorization_unet.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAAR_CASCADE  = os.path.join(CURRENT_DIR, "haarcascade_frontalface_default.xml")
FACE_SIZE = 200

class GrayscaleFilterRemover:
    
    def __init__(self, image: Image):
        self.image = image
        self.model = UNet()
        # Load the trained weights
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
    
    def remove_filter(self) -> Image:
        """ Approximates colour image from grayscale """
       
        # Initialize Haar cascade
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        
        if face_cascade.empty():
            print("Error loading Haar cascade.")
            return False
        
        gray_np = np.array(self.image)
        
        # Create a 3-channel version of the grayscale image for display
        display_frame = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)

        # To see the rest of the frame in grayscale:
        #display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        # To see the entire frame in color:
        # display_frame = frame_bgr.copy()

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_np, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Crop face from the original color frame (not grayscale!)
            face_roi_bgr = display_frame[y:y+h, x:x+w]

            # Resize to model input if needed
            face_resized_bgr = cv2.resize(face_roi_bgr, (FACE_SIZE, FACE_SIZE))

            # Colorize using the true Lab pipeline
            colorized_bgr = self.colorize_face_lab(face_resized_bgr, self.model, DEVICE)

            # Resize back to the bounding box size
            colorized_bgr = cv2.resize(colorized_bgr, (w, h))

            # Paste onto display_frame
            display_frame[y:y+h, x:x+w] = colorized_bgr

            # Bounding box
            #cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    
    def colorize_face_lab(self, face_bgr, model, device='cpu'):
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
        # (or use L directly from face_lab)
        L_scaled_2d = L_norm * 100.0  # back to [0..100]

        lab_pred = np.zeros((h, w, 3), dtype=np.float32)
        lab_pred[..., 0] = L_scaled_2d
        lab_pred[..., 1] = a_pred
        lab_pred[..., 2] = b_pred

        # Convert Lab -> BGR
        lab_uint8 = lab_pred.astype(np.uint8)
        face_bgr_colorized = cv2.cvtColor(lab_uint8, cv2.COLOR_Lab2BGR)

        return face_bgr_colorized
    
# Define the same U-Net architecture used in training (1 input channel -> 2 output channels)
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
