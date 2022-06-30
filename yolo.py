import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as tudata
import os
import pandas as pd
import torchvision as tv
def getgpu():
    return torch.device('cuda:0')

def net1():
    n1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
    n2 = nn.MaxPool2d(kernel_size=2, stride=2)
    n3 = nn.Conv2d(32, 96, kernel_size=3, padding=1)
    n4 = nn.MaxPool2d(kernel_size=2, stride=2)
    return nn.Sequential(n1, n2, n3, nn.BatchNorm2d(96),nn.ReLU(),n4)

def net2():
    n1 = nn.Conv2d(96, 64, kernel_size=1)
    n2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    n3 = nn.Conv2d(128, 128, kernel_size=1)
    n4 = nn.Conv2d(128, 256, kernel_size=3, padding = 1)
    n5 = nn.MaxPool2d(kernel_size=2, stride=2)
    return nn.Sequential(n1, n2, n3, n4, nn.BatchNorm2d(256), nn.ReLU(),n5)

def net3part():
    n1 = nn.Conv2d(256, 128, kernel_size=1)
    n2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    return nn.Sequential(n1,n2)

def net3():
    ls = [net3part(), net3part(), net3part()]
    ls.append(nn.Conv2d(256, 256, kernel_size=1))
    ls.append(nn.Conv2d(256, 512, kernel_size=3,padding=1))
    ls.append(nn.BatchNorm2d(512))
    ls.append(nn.ReLU())
    ls.append(nn.MaxPool2d(2,stride=2))
    return nn.Sequential(*ls)

def net4():
    n1 = nn.Conv2d(512, 256, kernel_size=1)
    n2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    n3 = nn.Conv2d(512, 256, kernel_size=1)
    n4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    n5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    n6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
    n7 = nn.Conv2d(512, 512, kernel_size=1)
    return nn.Sequential(n1, n2, n3, n4, n5, n6, n7)

def net5():
    n1 = nn.Flatten()
    n2 = nn.Linear(8192, 2048)
    n3 = nn.Linear(2048, 588)
    return nn.Sequential(n1, n2, n3)

class Yolo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n1 = net1()
        n2 = net2()
        n3 = net3()
        n4 = net4()
        n5 = net5()
        self.net = nn.Sequential(n1, n2, n3, n4, n5)
    def forward(self, X:torch.Tensor):
        y:torch.Tensor = self.net(X)
        y = y.reshape(-1, 7, 7, 12)
        return y

class BananaDataSet(tudata.Dataset):
    def __init__(self, is_train) -> None:
        super().__init__()
        dir = r'C:\Users\Orange\Desktop\python\deepLearn\banana\banana-detection'
        path = ''
        if is_train:
            path = os.path.join(dir,'bananas_train')
        else:
            path = os.path.join(dir, 'bananas_val')
        labels = pd.read_csv(os.path.join(path,'label.csv'))
        labels = labels.set_index('img_name')
        imgs = []
        targets= []
        for imgName, target in labels.iterrows():
            # print(imgName, target)
            imgs.append(tv.io.read_image(os.path.join(path,'images', imgName)))
            targets.append(list(target))
        self._imgs = imgs
        self._targets = torch.tensor(targets).unsqueeze(1) / 256
        print(self._targets.shape)

    def __getitem__(self, index):
        return self._imgs[index].float(), self._targets[index]

    def __len__(self):
        return len(self._imgs)


class Yolo_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coord = 5
        self.noobj = 0.5
    
    def box_iou(boxes1, boxes2):
        """
        计算交并比，boxes1 shape[M, 4], boxes2 shape[N, 4],生成 shape[M, N]
        """
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                (boxes[:, 3] - boxes[:, 1]))
        #计算面积
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        #找到相交部分的点
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas
    
    def forward(self, preds:torch.Tensor, target:torch.Tensor):
        exobj_mask = preds[:, :, :, 4] > 0
        noobj_mask = preds[:, :, :, 4] <=0
        batch_size = preds.shape[0]
        #将四个角点的形式换成中心点加高宽的形式,(x,y,h,w)
        target_c = target.clone()
        target_c[:, :, 1] = (target[:, :, 1]+target[:, :, 3])/2
        target_c[:, :, 2] = (target[:, :, 2]+target[:, :, 4])/2
        target_c[:, :, 3] = (target[:, :, 3] - target[:, :, 1])
        target_c[:, :, 4] = (target[:, :, 4] - target[:, :, 2])


        pass



net = Yolo()
x = torch.randn((3,3,256,256))
y = net(x)
print(y.shape)


# dataset = BananaDataSet(is_train=False)
# banaDataloader = tudata.DataLoader(dataset, batch_size=10, shuffle=True)
# for img, target in banaDataloader:
#     print(img)
#     print('====================')
#     print(target)
#     break


# # x = x.to(getgpu())

# print(n1.state_dict)
# print(n2.state_dict)

# X = torchvision.io.read_image(r'C:\Users\Orange\Desktop\python\deepLearn\nn\result\banana.jpeg').unsqueeze(0).float()
# img = X.squeeze(0).permute(1, 2, 0).long()
# print(X.shape)
# print(img.shape)