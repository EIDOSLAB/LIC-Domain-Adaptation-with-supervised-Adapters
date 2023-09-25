# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

class SquarePad:
    def __init__(self, patch_size) -> None:
        self.max_patch = max(patch_size[0], patch_size[1])
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        max_wh = max(max_wh, self.max_patch)
        hp = int((max_wh - w) / 2)+1
        vp = int((max_wh - h) / 2)+1
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding,  0,  'constant')




class DomainNet(Dataset):

    def __init__(self, root, transform = None, type = "sketch", split = "train"): 
        
        if split == "train":
            name =  "nuovo_train.txt"
            file_splitdir = Path(root) /  "splitting" / type / name
        else:
            name =   "nuovo_test.txt"
            file_splitdir = Path(root) /  "splitting" / type / name


        #data_dir =  Path(root) / "data" 




        file1 = open(file_splitdir, 'r')
        Lines = file1.readlines()




        self.samples =[]
        
        for i,line in enumerate(Lines):
            if i%10000==0:
                print(i)
            f_temp = line.strip().split(" ")[0]
            #path = os.path.join(line)

            img_t = Image.open(f_temp)

            l,w = img_t.size
            if l < 284 or w < 284:
                continue
                #print("porca miseria: ",i," ",img_t.size)


                    
            else:     


            #if split == "train":
            #    fls = Path(root) /  "splitting" / type /  "nuovo_train.txt"
            #else: 
            #    fls = Path(root) /  "splitting" / type /  "nuovo_test.txt"
            #f=open(fls , "a+")




                if split == "train":
                    self.samples.append(f_temp)
                    #f.write( path + "\n")
                    #f.close()  
                elif split == "valid":
                    self.samples.append(f_temp)
                    #f.write( path + "\n")
                    #f.close()
                    if i % 300 == 0:
                        self.samples.append(f_temp)
                    
                elif split == "test":
                    if i % 401 == 0:
                        self.samples.append(f_temp)



        if split == "valid":
            self.samples = random.sample(self.samples, 2048)     
        print("lunghezza: ",len(self.samples))
        self.transform = transform       

    def __getitem__(self, index):

        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    

class ImageFolder(Dataset):


    def __init__(self, root, num_images = 24000, transform=None, split="train"):
        splitdir = Path(root) / split / "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]

        num_images = num_images
            

        for i,f in enumerate(splitdir.iterdir()):
            if i%10000==0:
                print(i)
            if i <= num_images: 
                if f.is_file() and i < num_images:
                    self.samples.append(f)
            else:
                break
        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    
import glob 
import os
import pickle

from sklearn.model_selection import train_test_split

def elenca_file(directory):
    elenco_file = []

    # /sketch/
    different_classes = os.listdir(directory)
    for cl in different_classes:
        path = os.path.join(directory,cl)
        for files in os.listdir(path):
            percorso_completo = os.path.join(path,files)
            elenco_file.append(percorso_completo)

    #for path in glob.iglob(f'{directory}/**/*.png', recursive=True):
    #    print(path)

    return elenco_file

def extract_train_valid_test(root, vald_perc = 0.10, test_perc = 0.10, seed = 42):


    files_root = os.path.join("/scratch/dataset/PACS/splitting",root.split("/")[-1], str(seed)) # /scratch/dataset/PACS/splitting/art_painting/ ---
    #complete_root = os.path.join(root, str(seed))
    if os.path.isdir(files_root):
        print("la path è stata fatta, scarico i dati")

        with open(os.path.join(files_root,'file.pkl'), 'rb') as file:
            dati_caricati = pickle.load(file)
            return dati_caricati["train"], dati_caricati["validation"], dati_caricati["test"]
    
    else:
        lista_immagini = elenca_file(root)

        os.makedirs(files_root)
        print("il totale delle immagini sono: ",len(lista_immagini))
        train, temp = train_test_split(lista_immagini, test_size= vald_perc, random_state=seed)
        validation, test = train_test_split(temp, test_size=test_perc, random_state=seed)


        output = open(os.path.join(files_root,'file.pkl'), 'wb')

        pickle.dump({'train': train, 'validation': validation, 'test': test}, output)
        
        return train, validation, test


class PACS(Dataset):


    def __init__(self, samples, transform=None):
        


        self.samples = samples# [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    











class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [ transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def  handle_dataset(args,device): 

    if args.dataset_type == "openimages":


        train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images_train)
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)

        valid_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
        valid_dataset = ImageFolder(args.dataset, split="test", transform=valid_transforms, num_images=args.num_images_val)
        valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
        #test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)sss
        test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")
        test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
        
        filelist = [os.path.join("/scratch/dataset/kodak",f) for f in os.listdir("/scratch/dataset/kodak")]

        return train_dataloader, valid_dataloader, test_dataloader, filelist


    elif args.dataset_type == "domain":
        print("DOMAIN DATASET")





        overall_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])


    
        train_dataset = DomainNet(root = "/scratch/dataset/DomainNet", split="train", transform=overall_transforms, type = args.keyfold_dataset )
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)


        valid_dataset = DomainNet(root = "/scratch/dataset/DomainNet", split="valid", transform=overall_transforms, type = args.keyfold_dataset)
        valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)


        test_transforms = transforms.Compose([  transforms.CenterCrop(size = args.patch_size) ,transforms.ToTensor()])

        test_dataset =  DomainNet(root = "/scratch/dataset/DomainNet", split="test", transform= test_transforms, type = args.keyfold_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

        filelist = test_dataset.samples

        return train_dataloader, valid_dataloader, test_dataloader, filelist


    else:
        
        path_images = os.path.join("/scratch/dataset/PACS/kfold",args.keyfold_dataset)
        
        train, valid, test = extract_train_valid_test(path_images, seed = args.seed)




        print("Lunghezza testii: ",len(test))
        print("Lunghezza train: ",len(train) )
        print("lunghezza valid: ",len(valid))



        overall_transforms = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

        train_dataset = PACS(train, transform= overall_transforms)
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)
        
        valid_dataset = PACS(valid, transform= overall_transforms)
        valid_dataloader = DataLoader(valid_dataset,batch_size=1,num_workers=args.num_workers,shuffle=False, pin_memory=(device == "cuda"),)
        
        test_dataset = PACS(test, transform= overall_transforms)
        test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=args.num_workers,shuffle=False, pin_memory=(device == "cuda"),)


        print("done the dataset")
        return train_dataloader, valid_dataloader, test_dataloader, test



