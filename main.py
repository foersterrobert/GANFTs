import streamlit as st
import torch.nn as nn
import torch
import numpy as np
from PIL import Image

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, img_size):
        super().__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise | 1 x 1 x 100
            self._block(channels_noise, img_size * 36, 4, 1, 0),  # img: 4x4x864
            self._block(img_size * 36, img_size * 18, 4, 2, 1),  # img: 8x8x432
            self._block(img_size * 18, img_size * 9, 4, 2, 1),  # img: 16x16x216
            self._block(img_size * 9, img_size * 3, 5, 1, 0),  # img: 20x20x72
            nn.ConvTranspose2d(
                img_size * 3, channels_img, kernel_size=5, stride=1, padding=0
            ),
            # Output: N x channels_img | 24x24x4
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

st.experimental_singleton()
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = Generator(256, 4, 24).to(device)
    gan.load_state_dict(torch.load("ganPunk.pth", map_location=device))
    gan.eval()
    return gan

gan = load_model()

st.write("GANFTS")

btn = st.button("Generate")

columns = st.columns(5)

if btn:
    noise = torch.randn(len(columns) * 4, 256, 1, 1)
    fake = gan(noise)
    imgs = fake.detach().cpu().numpy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    imgs = np.moveaxis(imgs, 1, -1)
    imgs = imgs * 255
    for idx, column in enumerate(columns):
        for j in range(4):
            img = imgs[idx*4+j]
            img = Image.fromarray(img.astype(np.uint8))
            img = img.resize((240, 240), resample=Image.NEAREST)
            column.image(img, width=200)