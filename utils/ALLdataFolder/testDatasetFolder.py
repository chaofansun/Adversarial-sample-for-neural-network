from torchvision.datasets.folder import DatasetFolder
from PIL import Image
import csv
import torch
import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_for_defense(dir):
    images=[]
    counter=0
    files=list(os.walk(dir))[0][2]
    files.sort()
    for sample in files:
        try:
            extention=sample.split('.')[1]
            if extention=='png':
                path=os.path.join(dir,sample)
                item=(path,counter,sample)
                counter+=1
                images.append(item)
        except:
            continue
    return images

class defenseDatasetFolder(object):
    def __init__(self, root,loader=pil_loader,  transform=None):
        samples = make_dataset_for_defense(root)
        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path,idx,_= self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,torch.tensor([idx])

    def __len__(self): 
        return len(self.samples)


def make_dataset(dir,csv_name):
    counter=0
    images = []
    with open(os.path.join(dir,csv_name)) as csvfile:
        f=csv.reader(csvfile,delimiter=',')
        for row in f:
            try:
                idx=counter
                file_name=row[0]
                label=int(row[1])
                target=int(row[2])
                path=os.path.join(dir,file_name)
                images.append([path,label,target,file_name,idx])
                counter+=1
            except:
                continue
    return images

class attackDatasetFolder(object):
    def __init__(self, root,csv_name,loader=pil_loader,  transform=None):
        samples = make_dataset(root,csv_name)
        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path,label,target,_,idx= self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample,torch.tensor([label,target,idx])

    def __len__(self): 
        return len(self.samples)
