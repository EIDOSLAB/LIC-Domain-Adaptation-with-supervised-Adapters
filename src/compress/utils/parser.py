
import argparse




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

    parser.add_argument("-e","--epochs",default=800,type=int,help="Number of epochs (default: %(default)s)",)

    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument('--entropy_estimation', '-ep', action='store_true', help='entropy estimation')
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")


    args = parser.parse_args(argv)
    return args


def parse_args_gate(argv):
    parser = argparse.ArgumentParser(description="Example training script.")


    parser.add_argument("-e","--epochs",default=500,type=int,help="Number of epochs (default: %(default)s)",)

    parser.add_argument("--name_model", type=str,default = "WACNN", choices = ["WACNN","cheng"], help="possible models")

    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--root", type=str,default = "/scratch/dataset/DomainNet/splitting/mixed", help="base root for dataset") #"/scratch/dataset/domain_adapter/MixedImageSets"


    parser.add_argument("--considered_classes", nargs='+', type = str, default = ["openimages","sketch","infograph"]) # ["openimages","sketch","clipart",,"painting","infograph"] #["natural","sketch","infographics"]
    parser.add_argument("--train_datasets", nargs='+', type = str, default = ["train.txt"]) #["_natural_.txt","_sketch_.txt","_infographics_.txt"]
    parser.add_argument("--valid_datasets", nargs='+', type = str, default =["_natural_.txt","_sketch_.txt","_infographics_.txt"]) # ["valid_openimages.txt","valid_sketch.txt","valid_clipart.txt","valid_painting.txt","valid_infograph.txt"] 
    parser.add_argument("--test_datasets", nargs='+', type = str, default = ["_kodak_.txt","_clic_.txt","_sketch_.txt","_infographics_.txt"]) # ["test_kodak.txt","test_clic.txt","test_sketch.txt","test_clipart.txt","test_painting.txt","test_infograph.txt"] #["_kodak_.txt","_clic_.txt","_sketch_.txt","_comic_.txt","_infographics_.txt","_watercolor_.txt"]
    parser.add_argument("--lmbda",type=float,default=2.5,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)") 

    parser.add_argument("--N",default=192,type=int,help="Number of epochs (default: %(default)s)",) 
    parser.add_argument("--M",default=320,type=int,help="Number of epochs (default: %(default)s)",) 

    parser.add_argument("--origin_model",type = str,default="base")
    parser.add_argument("--pret_checkpoint",default = "/scratch/universal-dic/weights/q5/model.pth") #/scratch/KD/devil2022/derivation/q1/model.pth.tar
    parser.add_argument("--pret_checkpoint_gate",default ="none") #/scratch/KD/devil2022/gate/q6_gate.pth.tar #dddddd
    parser.add_argument("--patience",default=15,type=int,help="patience",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--starting_epoch",default=10000,type=int,help="starting_epoch",)

    parser.add_argument( "--mean", default=0.0, type=float, help="initialization mean",)
    parser.add_argument( "--std", default=0.00, type=float, help="initialization std",) #0.01
    parser.add_argument("--bias","-bs",action='store_true',)


    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--train_baseline", action="store_true", help="Use cuda")


    parser.add_argument("--oracle",action= "store_true",help = "use oracle during training")
    parser.add_argument("--ms_ssim",action= "store_true",help = "finetune")

    parser.add_argument("--type_adapter_attn", nargs='+', type = str, default=["singular","singular","singular","singular","singular"],help = "typology of adapters") # one for each domain 
    parser.add_argument("--dim_adapter_attn",nargs = '+',type = int ,default = [-1,-1,-1,-1,-1] ) # [192,192,192,192] #dddd
    parser.add_argument("--stride_attn", nargs = '+', type = int, default = [1,1,1,1,1])
    parser.add_argument("--kernel_size_attn",nargs = '+', type = int,  default = [3,3,3,3,3])
    parser.add_argument("--padding_attn", nargs = '+', type = int, default = [1,1,1,1,1])
    parser.add_argument("--position_attn", nargs = '+', type = str, default = ["res","res","res","res","res"])

    parser.add_argument("--aggregation", default = "weighted", type = str)


    parser.add_argument("--restart_training",action="store_true")
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument('--sgd', type = str,default = "adam", help='use sgd as optimizer')

    parser.add_argument("--skipped", action="store_true", help="Use cuda")


    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)
    parser.add_argument('--training_policy', '-tp',type = str, default = "e2e",choices = ["gate","e2e","adapter","fgta","e2e_mse","mse"], help='adapter loss')
    parser.add_argument('--writing',type = str, default =  "/scratch/KD/devil2022/results/starting/q5/DomainNet", help='adapter loss')


    # cheng stuff 
    parser.add_argument("--quality",default=6,type=int,help="quality, from 1 to 6",)


    args = parser.parse_args(argv)
    return args