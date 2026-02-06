import argparse
import time
import os
import cv2
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from metrics import *
from utils import *
import model.Config as config
from torch.utils.tensorboard import SummaryWriter
from model.RPRNet import RPRNet
import torch.nn.functional as F
import torch.nn.init as init
import math
from loss import *
from model.DeformConv3D import *
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['RPRNet'], type=list)
parser.add_argument("--num_frame", default=6, type=int)
parser.add_argument("--dataset_names", default=["TSIRMT"], type=list)
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=30, type=int, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--begin_test", default=500, type=int)
parser.add_argument("--every_test", default=1, type=int)
parser.add_argument("--every_save_pth", default=1, type=int)
parser.add_argument("--every_print", default=1, type=int)
parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=r'./log', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/RPRNet", help='path of log files')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--resume", default=False, type=bool, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)

config_vit = config.get_SCTrans_config()






def init_weights_full(model, prior_p_for_seg=0.01):


    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0.)

        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                init.constant_(m.weight, 1.)
            if m.bias is not None:
                init.constant_(m.bias, 0.)

        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0.0, std=0.02)

        if hasattr(m, "conv_h") and isinstance(getattr(m, "conv_h"), nn.Conv2d):
            init.orthogonal_(m.conv_h.weight)
            if m.conv_h.bias is not None:
                init.constant_(m.conv_h.bias, 0.)

        if isinstance(m, nn.MultiheadAttention):
            if hasattr(m, "in_proj_weight"):
                init.xavier_uniform_(m.in_proj_weight)
            if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
                init.constant_(m.in_proj_bias, 0.)
            if hasattr(m, "out_proj"):
                init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    init.constant_(m.out_proj.bias, 0.)

        if "decoder" in name or "attn" in name:
            if hasattr(m, "linear1"):
                init.xavier_uniform_(m.linear1.weight)
                if m.linear1.bias is not None:
                    init.constant_(m.linear1.bias, 0.)
            if hasattr(m, "linear2"):
                init.xavier_uniform_(m.linear2.weight)
                if m.linear2.bias is not None:
                    init.constant_(m.linear2.bias, 0.)
            if hasattr(m, "norm1"):
                init.constant_(m.norm1.weight, 1.)
                init.constant_(m.norm1.bias, 0.)
            if hasattr(m, "norm2"):
                init.constant_(m.norm2.weight, 1.)
                init.constant_(m.norm2.bias, 0.)
            if hasattr(m, "norm3"):
                init.constant_(m.norm3.weight, 1.)
                init.constant_(m.norm3.bias, 0.)

    if hasattr(model.model, "MultiScaleSegHead"):
        seg_head = getattr(model.model, "MultiScaleSegHead")
        b = math.log(prior_p_for_seg / (1 - prior_p_for_seg))  # logit(p)
        with torch.no_grad():
            for hname in ["out0", "out1", "out2", "out3", "fuse"]:
                if hasattr(seg_head, hname):
                    layer = getattr(seg_head, hname)
                    if isinstance(layer, nn.Conv2d) and layer.bias is not None and layer.out_channels == 1:
                        layer.bias.fill_(b)
        print(f"[init] set prior bias = logit({prior_p_for_seg:.2%}) = {b:.3f} on seg head layers")

def train():
    train_set = seqDataset(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg,num_frame=opt.num_frame, type='train')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    net = Net(model_name=opt.model_name, mode='train').cuda()
    init_weights_full(net, prior_p_for_seg=0.01)  
    macs, params = get_model_complexity_info(net, (3,6,256,256), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
    # net.train()
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    if opt.resume:

        ckpt = torch.load(r'D:\Code\RPRNet\log\TSIRMT_RM\\RPRNet.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
    ### Default settings of SCTransNet
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.0001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-6, 'last_epoch': -1}

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)
    get_loss = SPGloss()
    for idx_epoch in range(epoch_state, opt.nEpochs):
        net.train()
        results1 = (0, 0)
        results2 = (0, 0)
        total_loss_epoch = []
        with tqdm(total=len(train_loader), desc=f"Training Epoch {idx_epoch+1}/{opt.nEpochs}", unit="batch") as pbar:
            for idx_iter, (img, gt_mask) in enumerate(train_loader):
                img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
                if img.shape[0] == 1:   
                    continue
                gt_mask = gt_mask / 255.0
                preds = net.forward(img)
                loss = get_loss.forward(preds,gt_mask)

                total_loss_epoch.append(loss.detach().cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'batchidx': f"{idx_iter:d}",    
                    'loss': f"{loss.item():.8f}",
                    'avg_loss': f"{torch.mean(torch.tensor(total_loss_epoch)):.8f}"
                })
                pbar.update(1)  

            scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f,'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            # Log the scalar values
            writer.add_scalar('loss', total_loss_list[-1], idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        # 
        if (idx_epoch + 1) >= opt.begin_test and (idx_epoch + 1) % opt.every_test == 0:
            test_set = seqDataset(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg,num_frame=opt.num_frame, type='test')
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            net.eval()
            with torch.no_grad():
                eval_mIoU = mIoU()
                eval_PD_FA = PD_FA()
                test_loss = []
                for idx_iter, (img, gt_mask) in enumerate(test_loader):
                    img = Variable(img).cuda()
                    pred = net.forward(img)
                    gt_mask = gt_mask.unsqueeze(1)
                    gt_mask = gt_mask/255

                    loss = get_loss.forward(pred,gt_mask)
                    pred = pred['main_logit']
                    test_loss.append(loss.detach().cpu())
                    eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                    eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], [opt.patchSize,opt.patchSize])
                test_loss.append(float(np.array(test_loss).mean()))
                results1 = eval_mIoU.get()
                results2 = eval_PD_FA.get()
                writer.add_scalar('mIOU', results1[-1], idx_epoch + 1)
                writer.add_scalar('testloss', test_loss[-1], idx_epoch + 1)


        if (idx_epoch + 1) % opt.every_save_pth == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)

        if idx_epoch == epoch_state:
            best_mIOU = results1
            best_Pd = results2

        if results1[1] > best_mIOU[1]:
            best_mIOU = results1
            best_Pd = results2
            print('------save the best model epoch', opt.model_name,'_%d ------' % (idx_epoch + 1))
            opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
            print("pixAcc, mIoU:\t" + str(best_mIOU))
            print("testloss:\t" + str(test_loss[-1]))
            print("PD, FA:\t" + str(best_Pd))

            opt.f.write("pixAcc, mIoU:\t" + str(best_mIOU) + '\n')
            opt.f.write("PD, FA:\t" + str(best_Pd) + '\n')
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '_' + 'best' + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)

        # last epoch
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % opt.every_save_pth != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)


def test(save_pth):
    test_set = seqDataset(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg,num_frame=opt.num_frame, type='test')
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    with torch.no_grad():
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        for idx_iter, (img, gt_mask, [ori_w,ori_h]) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            gt_mask = gt_mask.unsqueeze(1)
            gt_mask = gt_mask/255
            pred = pred['main_logit']
            _,_, w, h = pred.shape      
            scale = min(w / ori_w, h / ori_h)
            nw = int(ori_w * scale)
            nh = int(ori_h * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            pred = pred[:,:,dy:dy+nh, dx:dx+nw]
            pred = F.interpolate(pred, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
            gt_mask = gt_mask[:,:,dy:dy+nh, dx:dx+nw]
            gt_mask = F.interpolate(gt_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)


            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], [ori_w,ori_h])

        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()

        print('== == == == == == == ', opt.model_name, ' == == == == == == ==')
        print("pixAcc, mIoU:\t" + str(results1))
        print("PD, FA:\t" + str(results2))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)

    return save_path





class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([200.0]).to('cuda:0'))
        self.model = RPRNet(3,opt.num_frame)
    def forward(self, img):
        return self.model(img)


    def STDloss(self, output, gt_masks, eps=1e-6):
        main_logit = output["main_logit"]
        sides = output["side_logits"]   
        alpha = 0.8
        gamma = 2.0
        Batchsize,_,_,_ = main_logit.shape  
        reduction='mean'
        loss = 0
        # 正/负样本部分
        for i in range(Batchsize):
            p = main_logit[i,0,:,:].clamp(1e-6, 1 - 1e-6)
            p1 = sides[0][i,0,:,:].clamp(1e-6, 1 - 1e-6)
            p2 = sides[1][i,0,:,:].clamp(1e-6, 1 - 1e-6)
            p3 = sides[2][i,0,:,:].clamp(1e-6, 1 - 1e-6)
            p4 = sides[3][i,0,:,:].clamp(1e-6, 1 - 1e-6)
            gt_mask = gt_masks[i,:,:]
            loss_pos = -alpha * (1 - p) ** gamma * gt_mask * torch.log(p)
            loss_neg = -(1 - alpha) * p ** gamma * (1 - gt_mask) * torch.log(1 - p)

            loss_pos1 = -alpha * (1 - p1) ** gamma * gt_mask * torch.log(p1)
            loss_neg1 = -(1 - alpha) * p1 ** gamma * (1 - gt_mask) * torch.log(1 - p1)

            loss_pos2 = -alpha * (1 - p2) ** gamma * gt_mask * torch.log(p2)
            loss_neg2 = -(1 - alpha) * p2 ** gamma * (1 - gt_mask) * torch.log(1 - p2)

            loss_pos3 = -alpha * (1 - p3) ** gamma * gt_mask * torch.log(p3)
            loss_neg3 = -(1 - alpha) * p3 ** gamma * (1 - gt_mask) * torch.log(1 - p3)

            loss_pos4 = -alpha * (1 - p4) ** gamma * gt_mask * torch.log(p4)
            loss_neg4 = -(1 - alpha) * p4 ** gamma * (1 - gt_mask) * torch.log(1 - p4)

            loss = loss_pos + loss_neg + loss_pos1 + loss_neg1 + loss_pos2 + loss_neg2 + loss_pos3 + loss_neg3 + loss_pos4 + loss_neg4
        loss = loss/Batchsize
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss



if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
