
from compress.datasets.utils import create_data_file_for_multiple_adapters, build_test_datafile





def main():

    #create_data_file_for_multiple_adapters(split = "train", num_im_per_class = 40000 )
    #print("inizio test")
    #create_data_file_for_multiple_adapters(split = "test", num_im_per_class = 200 )

    base_path = "../../results/files/writings/bam_comic/base/q5/q5_bam_comic_.txt"
    gate_path = "../../results/files/writings/bam_comic/gate/q5/__q5_bam_comic__epoch_-1_.txt"


    file_d = open(base_path,"r") 
    Lines = file_d.readlines()
    for i,lines in enumerate(Lines):
        images = lines.split(" ")[1]
        file_gate = open(gate_path,"r")
        for ll in file_gate:
            gate_im = ll.split(" ")[1]

            if images == gate_im:
                print(images,":",lines.split(" ")[5],"  ",ll.split(" ")[5], float(ll.split(" ")[5]) - float(lines.split(" ")[5])) 
            




if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main()
    #create_data_file_for_multiple_adapters()
