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
import pickle
from PIL import Image

def getTransform(input_size):
    transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x:torch.cat([x,x,x]) if x.shape[0]==1 else x),
    torchvision.transforms.Lambda(lambda x:x[:-1,:,:] if x.shape[0]==4 else x),
    ])
    return transform

class getImageInfos(object):
    def __init__(self,path,transform=None):
        self.path=path[:path.rfind('/')+1]
        self.transform=transform
        
        with open(path,'rb') as f:
            self.raw=pickle.load(f)        
    def _getImage(self,l):
        x=[]
        for i in l:
            path=os.path.join(self.path,i)
            img=Image.open(path)
            x.append(img)
        if self.transform is not None:
            x=[self.transform(img) for img in x]
            x=torch.cat([img.view(1,3,img.shape[1],img.shape[2]) for img in x])
        return x
    def __call__(self,i):
        return self._getImage(self.raw[i])

def getAdv(ratio=0.5,mask_ratio=0.7):
    mask=torch.zeros([3,299,299])
    mask[:,int((1-mask_ratio)*299/2):int((1+mask_ratio)*299/2),int((1-mask_ratio)*299/2):int((1+mask_ratio)*299/2)]=1
    transform=getTransform(299)
    targetFolder=getImageInfos('./sample_images/infos.pkl',transform=transform)
    testset=attackDatasetFolder(in_path,'dev.csv',transform=transform)
    for img,target in testset:
        targetImages=targetFolder(testset.samples[target[2]][2])
        targetImages=targetImages[((targetImages-img)**2).view(targetImages.shape[0],-1).sum(1).argmin()]
        image_name=testset.samples[target[2]][3]
        try:
            image=(img*ratio+(1-ratio)*targetImages)*mask+(1-mask)*img
        except:
            image=img
        image=inv_transform(image.cpu())
        image=np.uint8(image)
        image=Image.fromarray(image)
        image.save(os.path.join(out_path, image_name))


inv_transform=torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2]),
    torchvision.transforms.ToPILImage(mode='RGB'),
    torchvision.transforms.Resize(299)
])

if __name__=='__main__':
    in_path=sys.argv[1]
    out_path=sys.argv[2]
    num_classes=110
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    getAdv(ratio=0,mask_ratio=0.8)
