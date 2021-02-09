import argparse


def config():

    parser = argparse.ArgumentParser(description='Set paths for %(prog)s')
    parser.add_argument('-t','--train_dataset',type = str,default = "../datasetpaths/pf14_train_combined_metadata.txt",help='Path to training dataset')
    parser.add_argument('-v,','--validation_dataset',type = str,default = "../datasetpaths/pf14_val_combined_metadata.txt",help='Path to training dataset')
    parser.add_argument('-sd,','--save_dump',type = bool,default = True,help='Whether to dump numpy arrays')
    parser.add_argument('-lr,','--learning_rate',type = int,default = 0.002,help='set up learning rate')
    parser.add_argument('-bs,','--batch-size',type = int,default = 32,help='set batch size')
    parser.add_argument('-is,','--image_size',type = int,default = 150,help='Set image size for the model')
    parser.add_argument('-sc,','--save_checkpoint',type = bool,default = True,help='Set image size for the model')
    return parser

def add_to_config(d, config):
    """Add attributes to config class."""
    for k, v in d.items():
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        setattr(config, k, v)
    return config