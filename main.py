import streamlit as st
import torch.nn as nn
import torch
import numpy as np
from PIL import Image

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise | 1 x 1 x 100
            self._block(channels_noise, features_g * 36, 4, 1, 0),  # img: 4x4x864
            self._block(features_g * 36, features_g * 18, 4, 2, 1),  # img: 8x8x432
            self._block(features_g * 18, features_g * 9, 4, 2, 1),  # img: 16x16x216
            self._block(features_g * 9, features_g * 3, 5, 1, 0),  # img: 20x20x72
            nn.ConvTranspose2d(
                features_g * 3, channels_img, kernel_size=5, stride=1, padding=0
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

class Critic(nn.Module):
    def __init__(self, channels_img, img_size):
        super().__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img | N = W x H | 24 x 24 x 4
            nn.Conv2d(
                channels_img, img_size * 3, kernel_size=3, stride=1, padding=0
            ), # 20x20
            nn.LeakyReLU(0.2),
            self._block(img_size * 3, img_size * 9, 3, 1, 0), # 16x16
            self._block(img_size * 9, img_size * 18, 2, 2, 1), # 8x8
            self._block(img_size * 18, img_size * 36, 2, 2, 1), # 4x4
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(img_size * 36, 1, kernel_size=2, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

@st.cache()
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
    for column in columns:
        for j in range(8):
            noise = torch.randn(1, 256, 1, 1)
            fake = gan(noise)
            # score = critic(fake).detach().numpy().shape
            img = fake.detach().cpu().numpy()[0]
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((240, 240), resample=Image.NEAREST)
            column.image(img, width=200)
            # column.write(score)