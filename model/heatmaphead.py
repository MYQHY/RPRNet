import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .IRGSM_SAMPLE import IRGSM_SAMPLE

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, g=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        # self.norm = nn.GroupNorm(g, out_ch)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MultiScaleSegHead(nn.Module):

    def __init__(self, in_ch=768, mid_ch=256, out_ch=1, single_scale_only=False):
        super().__init__()
        self.single_scale_only = single_scale_only

        self.block1 = ConvGNAct(in_ch, mid_ch)
        self.block2 = ConvGNAct(mid_ch, mid_ch)
        self.block3 = ConvGNAct(mid_ch, mid_ch)

        self.up1 = IRGSM_SAMPLE(mid_ch, scale=2)
        self.up2 = IRGSM_SAMPLE(mid_ch, scale=2)
        self.up22 = IRGSM_SAMPLE(mid_ch, scale=2)

        self.up3 = IRGSM_SAMPLE(mid_ch, scale=2)
        self.up33 = IRGSM_SAMPLE(mid_ch, scale=2)
        self.up333 = IRGSM_SAMPLE(mid_ch, scale=2)
        self.out0 = nn.Conv2d(in_ch, out_ch, 1)
        self.out1 = nn.Conv2d(mid_ch, out_ch, 1)
        self.out2 = nn.Conv2d(mid_ch, out_ch, 1)
        self.out3 = nn.Conv2d(mid_ch, out_ch, 1)

        self.fuse = nn.Conv2d(out_ch * 4, out_ch, 1)

        self.gamma = nn.Parameter(torch.zeros(4))

    def _init_weights(self):
        for m in [self.out0, self.out1, self.out2, self.out3, self.fuse]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def set_pos_prior_bias(self, prior_p=0.01):
        b = math.log(prior_p / (1 - prior_p))
        heads = [self.out0, self.out1, self.out2, self.out3, self.fuse]
        with torch.no_grad():
            for h in heads:
                if h.bias is not None:
                    h.bias.fill_(b)
        print(f"[MultiScaleSegHead] set prior bias = logit({prior_p:.2%}) = {b:.3f}")

    def forward(self, x0):
        B, C, H, W = x0.shape
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)



#       # upsample to highest resolution
        x1u = self.up1(x1)
        x2u = self.up22(self.up2(x2))
        x3u = self.up333(self.up33(self.up3(x3)))
        l0u, l1u, l2u, l3u = self.out0(x0), self.out1(x1u), self.out2(x2u), self.out3(x3u)

        gam = torch.softmax(self.gamma, dim=0)
        fused = self.fuse(torch.cat([l0u, l1u, l2u, l3u], dim=1))
        main = fused + (gam[0]*l0u + gam[1]*l1u + gam[2]*l2u + gam[3]*l3u)
        main = torch.sigmoid(main)
        l0u = torch.sigmoid(l0u)
        l1u = torch.sigmoid(l1u)
        l2u = torch.sigmoid(l2u)
        l3u = torch.sigmoid(l3u)
        return {"main_logit": main, "side_logits": [l0u, l1u, l2u, l3u]}


