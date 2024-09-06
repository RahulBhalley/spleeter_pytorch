import torch
import torch.nn as nn
from params import params

def encoder_block(out_channel):
    """Create an encoder block with batch normalization and activation."""
    act_fn = nn.LeakyReLU(0.2) if len(params['instruments']) == 2 else nn.ELU()
    return nn.Sequential(
        nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
        act_fn
    )

def decoder_block(in_channel, out_channel, dropout=False):
    """Create a decoder block with transposed convolution, activation, batch normalization, and optional dropout."""
    layers = [
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=5, stride=2),
        nn.ELU(),
        nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """U-Net architecture for audio source separation."""

    def __init__(self, in_channel=2):
        super(UNet, self).__init__()

        self.pad = nn.ZeroPad2d(padding=(1, 2, 1, 2))

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2)
        self.encoder1 = encoder_block(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.encoder2 = encoder_block(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.encoder3 = encoder_block(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.encoder4 = encoder_block(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.encoder5 = encoder_block(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.encoder6 = encoder_block(512)

        # Decoder layers
        self.decoder1 = decoder_block(512, 256, dropout=True)
        self.decoder2 = decoder_block(512, 128, dropout=True)
        self.decoder3 = decoder_block(256, 64, dropout=True)
        self.decoder4 = decoder_block(128, 32)
        self.decoder5 = decoder_block(64, 16)
        self.decoder6 = decoder_block(32, 1)

        # Final layers
        self.mask = nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the U-Net."""
        # Encoder
        skip1 = self.pad(x)
        skip1 = self.conv1(skip1)
        down1 = self.encoder1(skip1)
        
        skip2 = self.pad(down1)
        skip2 = self.conv2(skip2)
        down2 = self.encoder2(skip2)

        skip3 = self.pad(down2)
        skip3 = self.conv3(skip3)
        down3 = self.encoder3(skip3)
        
        skip4 = self.pad(down3)
        skip4 = self.conv4(skip4)
        down4 = self.encoder4(skip4)

        skip5 = self.pad(down4)
        skip5 = self.conv5(skip5)
        down5 = self.encoder5(skip5)

        skip6 = self.pad(down5)
        skip6 = self.conv6(skip6)
        down6 = self.encoder6(skip6)

        # Decoder with skip connections
        up1 = self.decoder1(skip6)
        up1 = up1[:, :, 1:-2, 1:-2]
        merge1 = torch.cat((skip5, up1), 1)
        
        up2 = self.decoder2(merge1)
        up2 = up2[:, :, 1:-2, 1:-2]
        merge2 = torch.cat((skip4, up2), 1)

        up3 = self.decoder3(merge2)
        up3 = up3[:, :, 1:-2, 1:-2]
        merge3 = torch.cat((skip3, up3), 1)

        up4 = self.decoder4(merge3)
        up4 = up4[:, :, 1:-2, 1:-2]
        merge4 = torch.cat((skip2, up4), 1)

        up5 = self.decoder5(merge4)
        up5 = up5[:, :, 1:-2, 1:-2]
        merge5 = torch.cat((skip1, up5), 1)

        up6 = self.decoder6(merge5)
        up6 = up6[:, :, 1:-2, 1:-2]
        
        m = self.mask(up6)
        
        # Apply sigmoid to get final mask
        m = self.sig(m)
        return m


class MultiLoss(nn.Module):
    """Multi-instrument loss module."""

    def __init__(self, model_list, criterion, params):
        super(MultiLoss, self).__init__()
        self.model_list = model_list
        self.num_instrument = params['num_instruments']
        self.criterion = criterion
        self.sum = len(self.num_instrument)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, mix_stft_mag, separate_stft_mag):
        """Compute multi-instrument loss."""
        loss = 0
        pred_stft_mag = []
    
        # Get predictions for each instrument
        for model in self.model_list:
            pred = model(mix_stft_mag)
            pred_stft_mag.append(pred)

        # Compute loss for each instrument
        for i in range(self.sum):
            loss += self.criterion(pred_stft_mag[i] * mix_stft_mag, separate_stft_mag[i])
        loss = loss / self.sum

        return loss


class Softmax_UNet(nn.Module):
    """Apply softmax to U-Net outputs."""

    def __init__(self):
        super(Softmax_UNet, self).__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, masks):
        """Apply softmax to the stacked masks."""
        masks = torch.stack(masks, dim=0)
        return self.softmax(masks)