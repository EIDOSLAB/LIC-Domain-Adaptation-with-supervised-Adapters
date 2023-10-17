
from compress.datasets.utils import create_data_file_for_multiple_adapters, build_test_datafile






def main():

    #create_data_file_for_multiple_adapters(split = "train", num_im_per_class = 40000 )
    #print("inizio test")
    create_data_file_for_multiple_adapters(split = "test", num_im_per_class = 200 )

if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main()
    #create_data_file_for_multiple_adapters()
