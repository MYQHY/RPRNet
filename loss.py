import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SPGloss(nn.Module):
    def __init__(self,gamma=0, alpha=[0.1667, 0.8333], size_average=False, MaxClutterNum=120, ProtectedArea=2):
        super(SPGloss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)
        self.HardRatio = 1/2
        self.HardNum = round(MaxClutterNum*self.HardRatio)
        self.EasyNum = MaxClutterNum - self.HardNum

        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma=gamma
        self.alpha=alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)): self.alpha=torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha=torch.Tensor(alpha)


    def forward(self, pred, target):   ## Input: [2,1,512,512]    Target: [2,1,512,512]
        # total_SHAPEloss = shape_align_loss(pred,target)
        
        inputs = [pred['main_logit'],pred['side_logits'][0],pred['side_logits'][1],pred['side_logits'][2],pred['side_logits'][3]]
        total_loss = 0
        if target.ndim == 3:
            target = target.unsqueeze(1)
        for i in range(5):
            loss = 0
            input = inputs[i]
            template = torch.ones(1, 1, 2*self.ProtectedArea+1, 2*self.ProtectedArea+1).to(input.device)    ## [1,1,5,5]
            template_s = torch.ones(1, 1, 2*self.ProtectedArea, 2*self.ProtectedArea).to(input.device) 
            template_ss = torch.ones(1, 1, 2*self.ProtectedArea-1, 2*self.ProtectedArea-1).to(input.device) 
            template_sss = torch.ones(1, 1, 2*self.ProtectedArea-2, 2*self.ProtectedArea-2).to(input.device) 

            target_prot = F.conv2d(target.float(), template, stride=1, padding=self.ProtectedArea)          ## [2,1,512,512]
            target_prot = (target_prot > 0).float()
            target_prot_s = F.conv2d(target.float(), template_s, stride=1, padding="same")  
            target_prot_ss = F.conv2d(target.float(), template_ss, stride=1, padding="same")  
            target_prot_sss = F.conv2d(target.float(), template_sss, stride=1, padding="same")  


            ring1 = (target_prot - target_prot_s)
            ring1 = (ring1>0)
            ring2 = (target_prot_s-target_prot_ss)
            ring2 =  (ring2>0)
            ring3 = (target_prot_ss-target_prot_sss)
            ring3 =  (ring3>0)
            with torch.no_grad():
                loss_wise = self.bce_loss(input, target.float())        ## learning based on result of loss computing
                loss_p = loss_wise * (1 - target_prot)
                idx = torch.randperm(130) + 20

                batch_l = loss_p.shape[0]
                Wgt = torch.zeros(batch_l, 1, 256, 256)
                for ls in range(batch_l):
                    loss_ls = loss_p[ls, :, :, :].reshape(-1)
                    loss_topk, indices = torch.topk(loss_ls, 200)
                    indices_rand = indices[idx[0:self.HardNum]]         ## random select HardNum samples in top [20-150]
                    idx_easy = torch.randperm(len(loss_ls))[0:self.EasyNum].to(input.device)  ## random select EasyNum samples in all image
                    indices_rand = torch.cat((indices_rand, idx_easy), 0)
                    indices_rand_row = indices_rand // 256
                    indices_rand_col = indices_rand % 256
                    Wgt[ls, 0, indices_rand_row, indices_rand_col] = 1


                WgtData_New = Wgt.to(input.device)*(1-target_prot) + target.float()
                WgtData_New[WgtData_New > 1] = 1

            logpt = F.logsigmoid(input)
            logpt_bk = F.logsigmoid(-input)
            pt = logpt.data.exp()
            pt_bk = 1 - logpt_bk.data.exp()
            loss_gt = -self.alpha[1]*(1-pt)**self.gamma*target*logpt 
            loss_bk = - self.alpha[0]*pt_bk**self.gamma*(1-target)*logpt_bk

            WgtData_New_bk = WgtData_New + ring1* 0.5 + ring2 * 0.3 + ring3 * 0.2
            loss = loss_gt *WgtData_New + loss_bk * WgtData_New_bk
            total_loss = total_loss + loss


        return total_loss.sum()


