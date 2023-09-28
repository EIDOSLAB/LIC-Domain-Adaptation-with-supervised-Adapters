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
from random import sample, seed, shuffle
from os.path import exists
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
        
        
        
        if exists(Path(root) /  "splitting" / type / "train_images.txt") and exists(Path(root) /  "splitting" / type / "valid_images.txt") and exists(Path(root) /  "splitting" / type / "test_images.txt"):
            print("ho gia fatto la divisione!!!!!")
            first_time = False
        else: 
            first_time = True
        



        if split == "train":
            if  first_time:
                name =  "nuovo_train.txt"
            else: 
                name =  "train_images.txt"
            file_splitdir = Path(root) /  "splitting" / type / name
        elif split == "valid":
            if  first_time:
                name =  "nuovo_test.txt"
            else: 
                name =  "valid_images.txt"
            file_splitdir = Path(root) /  "splitting" / type / name
        else:
            if  first_time:
                name =  "nuovo_test.txt"
            else: 
                name =  "test_images.txt"
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

            if first_time:
                img_t = Image.open(f_temp)

                l,w = img_t.size
                if l < 284 or w < 284:
                    continue
            



                    
                else:     


                    if split == "train":
                        fls = Path(root) /  "splitting" / type /  "train_images.txt"
                    elif split == "valid": 
                        fls = Path(root) /  "splitting" / type /  "valid_images.txt"
                    else: 
                        fls = Path(root) /  "splitting" / type /  "test_images.txt"
                    
                    f=open(fls , "a+")




                    if split == "train":
                        self.samples.append(f_temp)
                        f.write( f_temp+ "\n")
                        f.close()  
                    elif split == "valid":
                        self.samples.append(f_temp)

                        f.write( f_temp + "\n")
                        f.close()
                        if i % 300 == 0:
                            self.samples.append(f_temp)
                            #f.write( f_temp + "\n")
                            #f.close()
                        
                    elif split == "test":
                        if i % 401 == 0:
                            self.samples.append(f_temp)
                            f.write( f_temp + "\n")
                            f.close()
            else: 
                f_temp = line.strip().split(" ")[0]
                self.samples.append(f_temp)




        print("lunghezza: ",len(self.samples)," ", split)
        if split == "valid":
            self.samples = sample(self.samples, 2048)     
        print("lunghezza: ",len(self.samples)," ", split)
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
    

import os






    











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





        train_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])

        valid_transforms = transforms.Compose([ transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

        test_transforms = transforms.Compose([  transforms.CenterCrop(size = args.patch_size) ,transforms.ToTensor()])


    
        train_dataset = DomainNet(root = "/scratch/dataset/DomainNet", split="train", transform=train_transforms, type = args.keyfold_dataset )
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)


        valid_dataset = DomainNet(root = "/scratch/dataset/DomainNet", split="valid", transform=valid_transforms, type = args.keyfold_dataset)
        valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)


        

        test_dataset =  DomainNet(root = "/scratch/dataset/DomainNet", split="test", transform= test_transforms, type = args.keyfold_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

        filelist = test_dataset.samples

        return train_dataloader, valid_dataloader, test_dataloader, filelist




class AdapterDataset(Dataset):



    def __init__(self, root, path=["train.txt"], transform = None):
        for p in path:
            splitdir = Path(root) / p

            file_d = open(splitdir,"r") 
            Lines = file_d.readlines()


            self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]          

            for i,lines in enumerate(Lines):
                if i%10000==0:
                    print(i)
                self.samples.append((lines.split(" ")[0], lines.split(" ")[1]))
        shuffle(self.samples)
        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.samples[index][0]).convert("RGB")
        classes = int(self.samples[index][1])
        if self.transform:
            return self.transform(img),classes
        return img, classes

    def __len__(self):
        return len(self.samples)
    




def create_data_file_for_multiple_adapters(classes = {"natural": 0},
                                            split = "test", 
                                           savepath = "/scratch/dataset/DomainNet/splitting/mixed",
                                           num_im_per_class = 916,
                                           random_seed = 42):
   
    if split == "train":
        fls = savepath + "/" + split + ".txt"
    else: 
        fls = savepath + "/" + "valid_openimages" + ".txt"
    f=open(fls , "a+")

    seed(random_seed)
    for i,cl in enumerate(list(classes.keys())):
        print("**********************  CLASSE: ",cl)
        if cl == "natural":
            if split == "train":
                pth = "/scratch/dataset/openimages/train/data"
            else:
                pth = "/scratch/dataset/openimages/test/data"
            lista_immagini = [os.path.join(pth,f) for f in os.listdir(pth)]
            immagini_train = sample(lista_immagini,num_im_per_class)

            for single_images in immagini_train:
                f=open(fls , "a+")
                f.write( single_images + " " + str(classes[cl]) + "\n")
                f.close()  
        elif cl in ("sketch","clipart"):


            
            pth = "/scratch/dataset/DomainNet/splitting/" + cl  + "/total_" + split + ".txt"
            file_d = open(pth,"r") 
            Lines = file_d.readlines()
            immagini_pox = []
            for i,line in enumerate(Lines):
                if i%10000==0:
                    print(i)
                f_temp = "/scratch/dataset/DomainNet/data/" + line.strip().split(" ")[0]
                #path = os.path.join(line)

                img_t = Image.open(f_temp)

                l,w = img_t.size
                if l > 284 and w > 284:
                    immagini_pox.append(f_temp) # + " " + str(classes[cl]) + "\n")
                    #f=open(fls , "a+")
                    #f.write( f_temp + " " + str(classes[cl]) + "\n")
                    #f.close() 
            
            immagini_train = sample(immagini_pox,num_im_per_class + 1)

            for files in immagini_train:
                f=open(fls , "a+")
                f.write(files  + " " + str(classes[cl]) + "\n")
                f.close()                



def build_test_datafile(type = "clipart", savepath = "/scratch/dataset/DomainNet/splitting/mixed/"):
    if type in ("kodak","clic"):
        classe = "0"
        fls = savepath + "/" + "test_" + type + ".txt"
        f=open(fls , "a+")  
        path = os.path.join("/scratch","dataset",type)
        lista_kodak_immagini =  [os.path.join(path, f) for f in os.listdir(path)]

        for files in lista_kodak_immagini:
            f=open(fls , "a+")
            f.write(files  + " " + classe + "\n")
            f.close()        

    elif type in ("clipart","sketch"):
        path = savepath + "valid.txt"
        file_d = open(path,"r") 
        Lines = file_d.readlines()
        lista_immagini = []

        for line in Lines:
            if type in line:
                lista_immagini.append(line)
        seed(42)
        sub_test_list = sample(lista_immagini,100) # 100 immagini di test

        fls = savepath + "/" + "test_" + type + ".txt"
        f=open(fls , "a+")    

        for s in sub_test_list:
            f=open(fls , "a+")
            f.write(s)
            f.close()  

        fls = savepath + "/" + "valid_" + type + ".txt"
        f=open(fls , "a+")    

        for s in lista_immagini:
            if s not in sub_test_list:
                f=open(fls , "a+")
                f.write(s)
                f.close()  
        print("gol")

               












