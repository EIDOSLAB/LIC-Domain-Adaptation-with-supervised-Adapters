
import argparse
from compress.zoo import models
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="latent",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=600,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument('--sgd', type = str,default = "adam", help='use sgd as optimizer')


    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lmbda",type=float,default=0.9,help="Bit-rate distortion parameter (default: %(default)s)",)
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



    parser.add_argument("--dim_adapter_1_attention",default = -1, type = int)
    parser.add_argument("--dim_adapter_2_attention", default = -1, type = int)

    parser.add_argument("--stride_1",default = 1, type = int)
    parser.add_argument("--stride_2", default = 1, type = int)



    parser.add_argument("--fact_extrema",default=30,type=int,help="factorized_extrema",)
    parser.add_argument('--fact_tr', '-ft', action='store_true', help='factorized trainable')
    parser.add_argument('--entropy_bot', '-eb', action='store_true', help='entropy bottleneck trainable')




    parser.add_argument("--gauss_extrema",default=120,type=int,help="gauss_extrema",)
    parser.add_argument('--gauss_tr', '-gt', action='store_true', help='gaussian trainable')







    parser.add_argument( "--alpha", default=0.95, type=float, help="target mean squared error",)

    parser.add_argument( "--momentum", default=0.0, type=float, help="momnetum for the optimizer",)
    parser.add_argument( "--weight_decay", default=0.0, type=float, help="weigth dacay for the optimizer (L2)",)

    parser.add_argument( "--starting_epoch", default=-1, type=int, help="first epoch for training (see difference)",)

    parser.add_argument( "--num_images_train", default=24016, type=int, help="images for training",)
    parser.add_argument( "--num_images_val", default=2048, type=int, help="images for validation",)  


    parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/derivation/sgd/00360-q6-devil2022.pth.tar") 
    parser.add_argument("--pret_checkpoint_teacher",default = "/scratch/universal-dic/weights/q6/model.pth")


    parser.add_argument('--unfreeze_ha',  action='store_true', help='unfreeze hyperprior analysis')
    parser.add_argument('--unfreeze_hma',  action='store_true', help='unfreeze hyperprior scale')
    parser.add_argument('--unfreeze_hsa',  action='store_true', help='unfreeze hyperprior mean')

    parser.add_argument("--scheduler","-sch", type = str, default ="plateau")
    parser.add_argument("--patience",default=10,type=int,help="patience",)

    parser.add_argument("--trainable_lrp","-tlrp",action='store_true',)

    parser.add_argument('--training_policy', '-tp',type = str, default = "kd", choices= ["entire_qe","quantization_lrp","residual","kd","entire", "quantization", "adapter","mse","controlled","only_lrp"] , help='adapter loss')

    parser.add_argument("--dim_adapter",default=1,type=int,help="dimension of the adapter",)
    parser.add_argument( "--mean", default=0.0, type=float, help="initialization mean",)
    parser.add_argument( "--std", default=0.01, type=float, help="initialization std",)
    parser.add_argument( "--kernel_size", default=1, type=int, help="initialization std",)
    parser.add_argument("--adapter_stride", default=-1, type = int)
    parser.add_argument("--padding", default=1, type = int)
    parser.add_argument("--bias","-bs",action='store_true',)
    parser.add_argument("--unfreeze_decoder","-ud",action='store_true',)
    parser.add_argument("--level_dec_unfreeze", default=-1, type = int)



    args = parser.parse_args(argv)
    return args



def parse_args_evaluation(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="no_mean",help="Model architecture (default: %(default)s)",)
    
    parser.add_argument("--lrp", action="store_true", help="Use cuda")
    parser.add_argument("--lrp_path",default="nolrp",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-gm","--gamma",default="gm0",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/quantization_error/",help="Model architecture (default: %(default)s)",)


    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument('--entropy_estimation', '-ep', action='store_true', help='entropy estimation')
    parser.add_argument("--cuda", action="store_true", help="Use cuda")


    args = parser.parse_args(argv)
    return args