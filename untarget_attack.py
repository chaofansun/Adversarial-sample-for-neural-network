import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from utils.model.modelmap import getModel
from utils.ALLdataFolder.testDatasetFolder import attackDatasetFolder
from functools import partial
import utils.method.method as method
import csv
import os
import sys
from PIL import Image

def getTransform(input_size):
    transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    return transform

#Gen
in_path='dev_data/'
out_path='output_images_untarget_exp/'
num_classes=110
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

inv_transform=torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2]),
    torchvision.transforms.ToPILImage(mode='RGB'),
    torchvision.transforms.Resize(299)
])

#genearting data
def getAdv(fn,models,weight=[1,1,1,1]):
    try:
        os.mkdir(out_path)
    except:
        pass
    testset=attackDatasetFolder(in_path,'dev.csv',transform=getTransform(299))
    loader=torch.utils.data.DataLoader(testset,shuffle=False,batch_size=1)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        image_name=testset.samples[target[0,2]][3]
        #mask_file_path=testset.samples[target[0,2]][0].replace('/input_images','./dev_mask').replace('png','npz')
        img=fn(data=data, target=target[:,0],white_models=models)
        img=img.cpu().detach()[0]
        img=inv_transform(img)
        img.save(os.path.join(out_path, image_name))

if __name__=='__main__':
    in_path=sys.argv[1]
    out_path=sys.argv[2]
    num_classes=110
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    inv_transform=torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2]),
        torchvision.transforms.ToPILImage(mode='RGB'),
        torchvision.transforms.Resize(299)
    ])
    white_model_names=['resnet50']#,'xception','InceptionV4','inceptionresnetv2'] #,'InceptionV3'
    white_models=[getModel(x,num_classes,device).eval() for x in white_model_names]
    # partial(method.PGD_l2_sum_logit_momentum_mask_v3,device=device,epsilon=200,step=5,momentum=0.5,mask_ratio=0.9,mode='bilinear',size=7,stride=7) 35
    fn=partial(method.PGD_l2_sum_logit_momentum_mask_v3,device=device,epsilon=300,step=5,momentum=0.5,mask_ratio=0.9,mode='bilinear',size=3,stride=7)
    getAdv(fn,white_models)
