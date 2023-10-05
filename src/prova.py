import torch 
import torch.nn as nn
#from compress.datasets.utils import create_data_file_for_multiple_adapters, build_test_datafile
import wandb
import neptune

def main(run):

    neptune.create_experiment('pytorch-quickstart', params = params)
    params = {"learning_rate": 0.001, "optimizer": "Adam"}
    run["parameters"] = params

    for epoch in range(10):

        run["train/loss"].append(value = 1.01 ** epoch,step = epoch)

    run["eval/f1_score"] = 0.66



    for epoch in range(10):
        run["test/loss"].append(value = 1.94** epoch, step = epoch)
    
    run.stop()


if __name__ == "__main__":
    #wandb.init(project="Gate-training", entity="albertopresta")  
    neptune.init(api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNWQwM2FmOC1mMjM2LTQ1YWUtOGI5NC0xMjBhYmRkMGM2NjAifQ==",
                  project_qualified_name="albertopresta/trial",
                  custom_run_id="Your custom run ID"
                  )

    #run = neptune.init_run(
    #    custom_run_id="Your custom run ID",
    #    project="albertopresta/trial",
    #    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNWQwM2FmOC1mMjM2LTQ1YWUtOGI5NC0xMjBhYmRkMGM2NjAifQ==",
    #    name="First PyTorch ever"
    #    )  # your credentials


    main()
    #Enhanced-imagecompression-adapter-sketch
    #create_data_file_for_multiple_adapters()
    #build_test_datafile()
