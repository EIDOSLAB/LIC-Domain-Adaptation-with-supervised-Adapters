import matplotlib.pyplot as plt 

import os 
import math 
import numpy as np
from matplotlib import rc

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
rc('font', family='Times New Roman')



Colors = {
    "Ballé18": ["b",'-'],
    "Ballé18-ours": ["r","-"],
    "Cheng20": [palette[0],"-"],
    "Cheng20-ours-3A":[palette[3],"-"],
    "Cheng20-ours-2A": ["r","-"],
    "Cheng20-ours": ["r","-"],

    "Xie21": [palette[0],'-.'],
    "Xie21-ours": [palette[3],"-"],
    "Xie21-ours": [palette[3],"-"],
    "jpeg": ["y",'-'],
    "Zou22-ours": ["r","-"],
    "Reference":[palette[0],'-.'],
    "Zou22-ours-4A":[palette[3],"-"],
    "Zou22-ours-6A":[palette[3],"-"],
    "Zou22-ours-3A":[palette[3],"-"],
    "Zou22-ours-1A":[palette[3],"-"],
    "Zou22-ours-2A":[palette[3],"-"],
    "Proposed":[palette[3],"-"],
    "Zou22-ours-5A":[palette[3],"-"]
}


def extract_results(f):
    last_f = f.split("/")[-1]
    print(f,"   ",last_f)
    nature = last_f.split("-")[0]

    file1 = open(f, 'r')
    Lines = file1.readlines()
    for line in Lines:
        if "AVG" in line:
            bpp = float(line.strip().split(" ")[3])
            psnr = float(line.strip().split(" ")[5])
            #mssim = float(line.strip().split(" ")[7])
            print(bpp,"  ",psnr)
    return bpp, psnr, nature 






def plot_rate_distorsion(risultati,  savepath):




    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(risultati.keys())

    for i,type_name in enumerate(list_names): 

        
        risultati[type_name]["bpp"].sort()
        risultati[type_name]["psnr"].sort()

        bpp = risultati[type_name]["bpp"]
        print("vppp is what: ",bpp)
        psnr = risultati[type_name]["psnr"]
        colore = risultati[type_name]["colors"][0]
        tratteggio =  risultati[type_name]["colors"][1]


        plt.plot(bpp,psnr,tratteggio,color = colore, label = type_name ,markersize=7)
        plt.plot(bpp, psnr,'o',color =  colore, markersize=11)


    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks([35, 36])

    plt.xticks([0.6,0.7,0.8,0.9])
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    #plt.title('MS-SSIM comparison')
    plt.grid()
    plt.legend(loc='lower right', fontsize = 25)


    nome = "prova.png"

    cp =  os.path.join(savepath,nome)

    plt.grid(True)
    #plt.savefig(cp)
    plt.savefig(cp, dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close()  





    

def main():

    files_path = "../../results/files/kodak"#args.path # path con i risultati su txt
    lista_files = [files_path + "/" + f  for f in os.listdir(files_path)]

    risultati = {}
    savepath =  "../../results/images/kodak"

    ii = 0
    for i,f in enumerate(lista_files): 
        
        if  "ad2" in f:
            continue
        bpp, psnr, nature = extract_results(f)

        print(nature)
        if nature in list(risultati.keys()):
            risultati[nature]["bpp"].append(bpp)
            risultati[nature]["psnr"].append(psnr)
        else: 
            risultati[nature] = {"bpp": [],"psnr": [], "colors": []}
            risultati[nature]["bpp"] = [bpp]
            risultati[nature]["psnr"] = [psnr]
            if nature == "ref":
                risultati[nature]["colors"] = [palette[ii],'-.']
            else:
                risultati[nature]["colors"] = [palette[ii],'-']
            ii = ii + 1




    plot_rate_distorsion(risultati, savepath)




if __name__ == "__main__":
    #Enhanced-imagecompression-adapter
    main()




    
    # poi plottare il tutto 