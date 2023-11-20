import matplotlib.pyplot as plt 

import os 
import math 
import numpy as np
from matplotlib import rc
from compress.utils.bjontegaard_metric import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
rc('font', family='Times New Roman')





def extract_results(f):
    last_f = f.split("/")[-1]
    #print(f,"   ",last_f)
    nature = last_f.split("-")[0]

    file1 = open(f, 'r')
    Lines = file1.readlines()
    prob_0, prob_1, prob_2 = [],[],[] 
    predictions = []
    for line in Lines:
        if "AVG" in line:
            bpp = float(line.strip().split(" ")[3])
            psnr = float(line.strip().split(" ")[5])
            #mssim = float(line.strip().split(" ")[7])
            #print(bpp,"  ",psnr)
        
        if "gate" in f and "AVG" not in line:
            vector = np.array([float(line.strip().split(" ")[9]),float(line.strip().split(" ")[10]),float(line.strip().split(" ")[11])])
            predictions.append(np.argmax(vector))
            prob_0.append(float(line.strip().split(" ")[9]))
            prob_1.append(float(line.strip().split(" ")[10]))
            prob_2.append(float(line.strip().split(" ")[11]))

    if "gate" in f:

        r =  [sum(prob_0) / len(prob_0),sum(prob_1) / len(prob_1),sum(prob_2) / len(prob_2) ]

    else:
        r = []
        
    


    return bpp, psnr, nature, r




from matplotlib.lines import Line2D

def plot_rate_distorsion(risultati,  savepath, domain):




    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(risultati.keys())

    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for i,type_name in enumerate(list_names): 

        #print("type_name------> ",type_name)
        risultati[type_name]["bpp"].sort()
        risultati[type_name]["psnr"].sort()

        bpp = risultati[type_name]["bpp"]

        psnr = risultati[type_name]["psnr"]
        colore = risultati[type_name]["colors"][0]
        

        alpha = 0.7 if domain in ("kodak","clic") and type_name == "gate" else 1
        #print("alphaaaaaaa ",alpha)

        plt.plot(bpp,psnr,"-" if "gate" in type_name else ":",color = colore, label =  type_name ,markersize=7, alpha=alpha) #type
        plt.plot(bpp, psnr,'o' if "gate" in type_name else "*",color =  colore, markersize=5, alpha=alpha )



        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]
    print("----------------> ",domain,": ",psnr)
    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10 + 1), int(massimo_bpp*10 + 2))]
    #print(bpp_tick)
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    #plt.title('MS-SSIM comparison')
    plt.grid()
    
    legend_elements = [Line2D([0], [0], label= r'Zou et al. [10]',color=palette[0]),
                       Line2D([0], [0], label= r'Cheng et al. [9]',color=palette[1]),
                        Line2D([0], [0], marker = "o", label='Proposed', color='k'),
                     Line2D([0], [0], marker='*',linestyle= ":" , label='Reference', color='k')]

    plt.legend(handles=legend_elements, loc = "lower right",labelcolor='k',fontsize=25)
    #plt.legend(loc='lower right', fontsize = 25)


    nome = "mixture_result_with_CHENG_0611" + domain + "_.pdf"

    cp =  os.path.join(savepath,nome)

    plt.grid(True)
    #plt.savefig(cp)
    plt.savefig(cp, dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close()  



def plot_trend(df):
    pass

    

def main():
    
    
    nature = ["base","gate"]

    domain = "bam_comic"
    ii = 0
    
    risultati = {}
    for mod in ["devil2022","cheng2020"]:
        total_distribution = []
        files_path = "../../results/files/" + mod + "/writings/mixture/" + domain #args.path # path con i risultati su txt

        #exclusions = ["q2","q3"]
        
        savepath =  "../../results/images/" + domain 
        for n in ["base","gate"]:

            all_models = []
            path_nature = [os.path.join(files_path,n,f) for f in os.listdir(os.path.join(files_path,n))] #sketch/base/q1

            for q in path_nature:
                all_models.extend([os.path.join(q,qq) for qq in os.listdir(q)])
            
            # ora abbiamo una lista, fissata la natura, con tutti i risultati
            for i,f in enumerate(all_models):
                bpp, psnr, nature, distributions = extract_results(f)

                if n== "gate":
                    total_distribution.append(distributions)


                
                chiave = n + "_" + mod
                print("chiave: ",chiave)
                if chiave in list(risultati.keys()):
                    risultati[chiave]["bpp"].append(bpp)
                    risultati[chiave]["psnr"].append(psnr)
                else: 
                    risultati[chiave] = {"bpp": [],"psnr": [], "colors": []}
                    risultati[chiave]["bpp"] = [bpp]
                    risultati[chiave]["psnr"] = [psnr]
                    if nature == "ref":
                        risultati[chiave]["colors"] = [palette[ii],':']
                        risultati[chiave]["legend"] = ["Reference"]
                    else:
                        risultati[chiave]["colors"] = [palette[ii],'-']
                        risultati[chiave]["legend"] = ["Proposed"]
                    
                    
        ii = ii + 1


        #print("ISULTATI: ",risultati)
        
        if "cheng" in mod:
            bpp_ref = risultati["base_cheng2020"]["bpp"]
            psnr_ref = risultati["base_cheng2020"]["psnr"]

            bpp_prop = risultati["gate_cheng2020"]["bpp"]
            psnr_prop = risultati["gate_cheng2020"]["psnr"]

            print("******************** ",mod)
            print('BD-PSNR: ', BD_PSNR(bpp_ref, psnr_ref, bpp_prop, psnr_prop, 1))
            print('BD-RATE: ', BD_RATE(bpp_ref, psnr_ref, bpp_prop, psnr_prop, 1))


    plot_rate_distorsion(risultati, savepath, domain)

    
    """
    path = "../../results/validation_trend.csv"

    # Replace 'your_file.csv' with the actual path to your CSV file
    file_path = "../../results/validation_trend.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)


    steps = []
    top1 = []
    oracle = []
    weighted = []
    # Print the first 5 rows
    print(df.head())

    for index, row in df.iterrows():
        if index > 199:
            steps.append(row['valid'])
            top1.append(row[ 'Proposed-top1 - valid/psnr__MAX'])
            oracle.append(row['Proposed-Oracle - valid/psnr'])
            weighted.append(row['Proposed-weighted - valid/psnr__MAX'])
            # Ora puoi accedere ai dati di ciascuna riga tramite il dizionario 'row'
            # Ad esempio, per accedere alla colonna 'Nome' della riga corrente:

    print(top1)

    plt.plot(steps, top1,  linestyle='-', label = "top1")
    plt.plot(steps, weighted,  linestyle='-', label = "weighted")
    plt.plot(steps, oracle,  linestyle='-', label = "oracle")


    # Customize the plot
    plt.xlabel('epochs', fontsize = 20)
    plt.ylabel('PSNR', fontsize = 20)
    plt.title('')

    # Display the plot
    plt.legend(loc = 'best')
    plt.grid()
    plt.savefig("../../valid_psnr_trend.pdf", dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close()  

    """



if __name__ == "__main__":
    #Enhanced-imagecompression-adapter
    main()




    
    # poi plottare il tutto 
