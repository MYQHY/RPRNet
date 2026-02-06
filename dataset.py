from utils import *
import matplotlib.pyplot as plt
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # rnd_bn = np.random.normal(0, 0.03)#0.03
        # img += rnd_bn
        #
        # minm = img.min()
        # rng = img.max() - minm
        # gamma = np.random.uniform(0.5, 1.6)
        # x=((img - minm) / rng)
        # img = np.power(x, gamma)
        # img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class TrainSetLoader02(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        rnd_bn = np.random.normal(0, 0.03)#0.03
        img += rnd_bn
        #
        # minm = img.min()
        # rng = img.max() - minm
        # gamma = np.random.uniform(0.5, 1.6)
        # x=((img - minm) / rng)
        # img = np.power(x, gamma)
        # img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader03(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # rnd_bn = np.random.normal(0, 0.03)#0.03
        # img += rnd_bn

        minm = img.min()
        rng = img.max() - minm
        gamma = np.random.uniform(0.5, 1.6)
        x=((img - minm) / rng)
        img = np.power(x, gamma)
        img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader04(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        rnd_bn = np.random.normal(0, 0.03)#0.03
        img += rnd_bn

        minm = img.min()
        rng = img.max() - minm
        gamma = np.random.uniform(0.5, 1.6)
        x=((img - minm) / rng)
        img = np.power(x, gamma)
        img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class seqDataset(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg, num_frame ,type='train'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_dir +'/'+ dataset_name
        self.img_idx = []
        self.anno_idx = []
        self.image_size = patch_size
        self.num_frame = num_frame
        if type == 'train' :
            self.txt_path = self.dataset_path + '/train.txt'
            self.aug = True
            self.type = 'train'
        elif type == "train_multi_output":
            self.txt_path = self.dataset_path + '/train.txt'
            self.aug = True
            self.type = 'train_multi_output'
        else:
            self.txt_path = self.dataset_path + '/test.txt'
            self.aug = False
            self.type = 'test'
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
        self.tranform = InfraredMultiFrameAugSingleMask(
                            flip_prob=0.5,
                            rotate_prob=0.3,
                            noise_prob=0.4,
                            brightness_prob=0.3,
                            max_angle=20,
                            noise_sigma=8
                        )
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.type == 'train' or self.type == 'train_multi_output':
            images, box  = self.get_data(index)
        else:
            images, box , [ori_w,ori_h] = self.get_data(index)
        images = np.transpose(self.preprocess(images),(3, 0, 1, 2))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        if self.type == 'train' or self.type == 'train_multi_output':
            return images, box
        else:
            return images, box , [ori_w,ori_h]      
        
    def preprocess(self,image):
        image /= 255.0
        image -= np.array([0.28, 0.28, 0.28])
        image /= np.array([0.23, 0.23, 0.23])
        return image
    def cvtColor(self,image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 
    def split_number(n: int):
        if n % 2 == 0:
            return n // 2, n // 2
        else:
            return n // 2 + 1, n // 2
    def get_exist_index(self,image_path,image_id,rel_id1,rel_id2):
       # 获取文件夹中所有存在的图片编号
        files = [f for f in os.listdir(image_path) if f.endswith('.png')]
        existing_ids = sorted([
            int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit()
        ])

        if image_id not in existing_ids:
            raise ValueError(f"中心编号 {image_id} 不存在于 {image_path}")

        result = []
        # 生成完整的相对偏移范围，例如 [-2, -1, 0, 1, 2, 3]
        for offset in range(rel_id1, rel_id2):
            target_id = image_id + offset
            if target_id in existing_ids:
                result.append(target_id)
            else:
                # 向上寻找更大的存在编号
                replacement = target_id
                while replacement <= max(existing_ids) and replacement not in existing_ids:
                    replacement += 1
                # 如果还找不到，反方向找小一点的编号
                if replacement not in existing_ids:
                    replacement = target_id
                    while replacement >= min(existing_ids) and replacement not in existing_ids:
                        replacement -= 1
                # 若都找不到，最后兜底用 center_id
                if replacement not in existing_ids:
                    replacement = image_id
                result.append(replacement)

        return result
    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        mask_path = image_path.replace(image_path.split("/")[-2], '')+'masks/'
        # label_data = self.anno_idx[index]  # 4+1
        did = (self.num_frame + 1) // 2 - 1
        idrange = self.get_exist_index(self.dataset_path+'/'+image_path,image_id,0 - did,self.num_frame - did)
        # id1 = id12[0]
        # id2 = id12[1]
        mask_ok = 0
        for id in idrange :
            img = Image.open(f"{self.dataset_path}/{image_path}{id:05d}.png")


            img = self.cvtColor(img)
            iw, ih = img.size
            
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            
            img = img.resize((nw, nh), Image.BICUBIC)  # 原图等比列缩放
            new_img = Image.new('RGB', (w,h), (128, 128, 128))  # 预期大小的灰色图
            new_img.paste(img, (dx, dy))  # 缩放图片放在正中
            image_data.append(np.array(new_img, np.float32))
            
            if  id == image_id and mask_ok == 0 :
                if self.type == "train_multi_output":
                    mask_data = []
                    id_index = idrange.index(id)  # 当前帧在idrange中的索引
                    neighbor_ids = []
                    if id_index - 1 >= 0:
                        neighbor_ids.append(idrange[id_index - 1])
                    else:
                        neighbor_ids.append(None)
                    neighbor_ids.append(id)
                    if id_index + 1 < len(idrange):
                        neighbor_ids.append(idrange[id_index + 1])
                    else:
                        neighbor_ids.append(None)
                    for mid in neighbor_ids:
                        if mid == None:
                            mask = Image.open(f"{self.dataset_path}/{mask_path}{id:05d}.png")
                        else:
                            mask_file = f"{self.dataset_path}/{mask_path}{mid:05d}.png"
                            mask = Image.open(mask_file)
                        mask = mask.resize((nw, nh), Image.BICUBIC)
                        new_mask = Image.new('L', (w, h), 0)
                        new_mask.paste(mask, (dx, dy))
                        mask_data.append(np.array(new_mask, np.float32))
                    new_mask = mask_data
                else:
                    mask = Image.open(f"{self.dataset_path}/{mask_path}{id:05d}.png")
                    ori_w,ori_h = mask.size
                    mask = mask.resize((nw, nh), Image.BICUBIC)  # 原图等比列缩放
                    new_mask = Image.new('L', (w,h), (0))  # 预期大小的灰色图
                    new_mask.paste(mask, (dx, dy))  # 缩放图片放在正中
                    mask_ok = 1
                    
        image_data = np.array(image_data) # 关键帧在后 # [5,w,h,3]
        mask_data = np.array(new_mask, dtype=np.float32) # [:,5]
        mask_data[mask_data > 0] = 255
        if self.aug is True:
            # image_data, label_data[:,:4] = augmentation(image_data,label_data[:,:4],h,w)
            pass
        if self.type == 'test':
            pass
        else:
            pass
            # image_data, mask_data = self.tranform(image_data, mask_data)  # 数据增强
        if self.type == 'train' or self.type == 'train_multi_output':
            return image_data, mask_data
        else:
            return image_data, mask_data, [ori_w,ori_h]

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
        # with open(r'D:\05TGARS\Upload\datasets\SIRST3\img_idx\val.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open(
            (self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//', '/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target

class seq_augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[:, ::-1, :, :].copy()
            target = target[::-1, :].copy()
        if random.random() < 0.5:  # 垂直反转
            img = cv2.flip(img, 0)  # 0表示上下翻转
            mask = cv2.flip(mask, 0)
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(0, 2, 1, 3).copy()
            target = target.transpose(1, 0).copy()
        return input, target
    



class InfraredMultiFrameAugSingleMask:
    """
    红外小目标多帧检测数据增强模块（输入多帧图像，单帧mask）
    支持输入:
        imgs.shape = (T, H, W, C)
        mask.shape = (H, W)
    """

    def __init__(self,
                 flip_prob=0.5,
                 rotate_prob=0.3,
                 noise_prob=0.3,
                 brightness_prob=0.3,
                 max_angle=15,
                 noise_sigma=5,
                 brightness_factor=0.15):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.brightness_prob = brightness_prob
        self.max_angle = max_angle
        self.noise_sigma = noise_sigma
        self.brightness_factor = brightness_factor

    def __call__(self, imgs, mask):
        """imgs: (T,H,W,C), mask: (H,W)"""
        T, H, W, C = imgs.shape

        # ---- 随机垂直翻转 ----
        if random.random() < self.flip_prob:
            imgs = np.flip(imgs, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # ---- 随机水平翻转 ----
        if random.random() < self.flip_prob:
            imgs = np.flip(imgs, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # ---- 随机旋转 ----
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_angle, self.max_angle)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

            imgs = np.stack([
                cv2.warpAffine(imgs[i], M, (W, H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
                for i in range(T)
            ], axis=0)

            mask = cv2.warpAffine(mask.astype(np.uint8), M, (W, H),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT)

        return imgs, mask