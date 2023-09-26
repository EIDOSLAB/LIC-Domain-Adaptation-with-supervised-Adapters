
import argparse
from compress.zoo import models
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="split",choices=models.keys(),help="Model architecture (default: %(default)s)",)

    parser.add_argument("-dt", "--dataset_type", type=str, default = "domain", choices = ["domain","openimages","PACS"], help="Training dataset")

    parser.add_argument("-kfd", "--keyfold_dataset", type=str, default = "sketch", choices = ["sketch","cartoon"], help="Training dataset")

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=800,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument('--sgd', type = str,default = "adam", help='use sgd as optimizer')


    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lmbda",type=float,default=0.0483,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--gamma",type=float,default=0.0,help="Bit-rate distortion parameter (default: %(default)s)",)


    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=64,help="Test batch size (default: %(default)s)",)
    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="7ckpt/model.pth.tar", help="Where to Save model")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")




    parser.add_argument("--dims_n",default=192,type=int,help="Number of epochs (default: %(default)s)",) 
    parser.add_argument("--dims_m",default=320,type=int,help="Number of epochs (default: %(default)s)",)








    parser.add_argument( "--momentum", default=0.0, type=float, help="momnetum for the optimizer",)
    parser.add_argument( "--weight_decay", default=0.0, type=float, help="weigth dacay for the optimizer (L2)",)



    parser.add_argument( "--num_images_train", default=16024, type=int, help="images for training",)
    parser.add_argument( "--num_images_val", default=1024, type=int, help="images for validation",)  


    #parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/derivation/adam/00670-q6-devil2022-adam.pth.tar") #ssssssssssszzz
    parser.add_argument("--pret_checkpoint",default = "/scratch/universal-dic/weights/q6/model.pth") 



    parser.add_argument('--unfreeze_hsa_loop',  action='store_true', help='unfreeze hyperprior analysis')
    parser.add_argument('--unfreeze_hma',  action='store_true', help='unfreeze hyperprior mean')
    parser.add_argument('--unfreeze_hsa',  action='store_true', help='unfreeze hyperprior scale')

    parser.add_argument("--scheduler","-sch", type = str, default ="plateau")
    parser.add_argument("--patience",default=20,type=int,help="patience",)


    parser.add_argument('--training_policy', '-tp',type = str, default = "mse", choices= ["entire", "quantization", "adapter","mse","rate"] , help='adapter loss')

    

    parser.add_argument( "--mean", default=0.0, type=float, help="initialization mean",)
    parser.add_argument( "--std", default=0.01, type=float, help="initialization std",)
    parser.add_argument("--bias","-bs",action='store_true',)
    parser.add_argument("--unfreeze_decoder","-ud",action='store_true',)
    parser.add_argument("--level_dec_unfreeze", nargs='+', type = int, default = [6,7,8])



    parser.add_argument("--depth", default=1, type = int)

    parser.add_argument("--type_adapter_attn_1",type = str, choices=["singular","transformer","attention","attention_singular","multiple"],default="singular",help = "typology of adapters")
    parser.add_argument("--dim_adapter_attn_1",default = 320, type = int)
    parser.add_argument("--stride_attn_1",default = 1, type = int)
    parser.add_argument("--kernel_size_attn_1", default = 3, type = int)
    parser.add_argument("--padding_attn_1", default = 1, type = int)
    parser.add_argument("--position_attn_1", default = "res_last", type = str)

    parser.add_argument("--type_adapter_attn_2",type = str, choices=["singular","transformer","attention","attention_singular","multiple"],default="singular",help = "typology of adapters")
    parser.add_argument("--dim_adapter_attn_2", default =192, type = int)
    parser.add_argument("--stride_attn_2", default = 1, type = int)
    parser.add_argument("--kernel_size_attn_2", default = 3, type = int)
    parser.add_argument("--padding_attn_2", default = 1, type = int)
    parser.add_argument("--position_attn_2", default = "res_last", type = str)




    parser.add_argument("--type_adapter_deconv_1",type = str, choices=["singular","transformer","attention","attention_singular"],default="singular",help = "typology of adapters")
    parser.add_argument("--dim_adapter_deconv_1",default = 0, type = int)
    parser.add_argument("--stride_deconv_1",default = 1, type = int)
    parser.add_argument("--kernel_size_deconv_1", default = 1, type = int)
    parser.add_argument("--padding_deconv_1", default = 1, type = int)
    parser.add_argument("--position_deconv_1", default = "res_last", type = str)

    parser.add_argument("--type_adapter_deconv_2",type = str, choices=["singular","transformer","attention","attention_singular"],default="singular",help = "typology of adapters")
    parser.add_argument("--dim_adapter_deconv_2", default =0, type = int)   
    parser.add_argument("--stride_deconv_2", default = 1, type = int)   
    parser.add_argument("--kernel_size_deconv_2", default = 1, type = int)   
    parser.add_argument("--padding_deconv_2", default = 0, type = int)   
    parser.add_argument("--position_deconv_2", default = "res_last", type = str)





    args = parser.parse_args(argv)
    return args



def parse_args_evaluation(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="both",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-tq","--targ_q",default="q6",help="quality level to check",)
    parser.add_argument("-pm","--path_models",default="/scratch/KD/devil2022/derivation/adam",help="Model architecture (default: %(default)s)",)
    
    parser.add_argument("--lrp", action="store_true", help="Use cuda")
    parser.add_argument("--lrp_path",default="nolrp",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-gm","--gamma",default="gm0",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/quantization_error/",help="Model architecture (default: %(default)s)",)

    parser.add_argument("-td","--test_dataset", default = "kodak", type = str)


    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument('--entropy_estimation', '-ep', action='store_true', help='entropy estimation')
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")


    args = parser.parse_args(argv)
    return args