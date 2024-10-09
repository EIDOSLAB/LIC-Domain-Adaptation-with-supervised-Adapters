# Domain Adaptation for learned image compression with supervised Adapters


Pytorch implementation of the paper "**Domain Adaptation for learned image compression with supervised Adapters**", published at DCC 2024. This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [STF](https://github.com/Googolxx/STF)

[Paper link](https://arxiv.org/abs/2404.15591)



## Abstract
In Learned Image Compression (LIC), a model is trained at encoding and decoding images sampled from a source domain, often outperforming traditional codecs on natural images; yet its performance may be far from optimal on images sampled from different domains. In this work, we tackle the problem of adapting a pre-trained model to multiple target domains by plugging into the decoder an adapter module for each of them, including the source one. Each adapter improves the decoder performance on a specific domain, without the model forgetting about the images seen at training time. A gate network computes the weights to optimally blend the contributions from the adapters when the bitstream is decoded. We experimentally validate our method over two state-of-the-art pre-trained models, observing improved rate-distortion efficiency on the target domains without penalties on the source domain. Furthermore, the gate's ability to find similarities with the learned target domains enables better encoding efficiency also for images outside them.




### Dataset 

Datasets used for training/evaluation as sored at [this link](https://drive.google.com/drive/u/0/folders/15cwEsnMxuBaEAH_h8_CnpcpRtOnXiuUQ)
In the unzipped folder we have:


domain_adaptation_dataset
│  
└──MixedImageSets
   └───|test.xtx
       |train.txt
       | valid.txt    
└───clipart
    |____ img1.jpg ...


Once you downloaded the datasets you put the final folder in path_to_dataset

base_pretrained model: put pretrained models in path_to_pretrained_model

## Usage

### Environment
```bash
conda create -n $YOUR_PY38_ENV_NAME python=3.8
conda activate $YOUR_PY38_ENV_NAME
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
### train 
```bash
cd src 

python train_final.py 
 --root path_to_dataset
 --pret_checkpoint path_to_base_checkpoint 
 --classes natural,sketch,clipart
 --test_classes kodak,sketch,clipart
 --quality q6
 --savepath path_where_to_save_model
 --seed 42
```


