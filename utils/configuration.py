import argparse


def config():

    parser = argparse.ArgumentParser(description='Set paths for %(prog)s')
    parser.add_argument('-t','--train_dataset',type = str,default = "../datasetpaths/pf14_train_combined_metadata.txt",help='Path to training dataset')
    parser.add_argument('-v,','--validation_dataset',type = str,default = "../datasetpaths/pf14_val_combined_metadata.txt",help='Path to training dataset')
    parser.add_argument('-sd,','--save_dump',type = bool,default = False,help='Whether to dump numpy arrays')
    parser.add_argument('-lr,','--learning_rate',type = float,default = 0.001,help='set up learning rate')
    parser.add_argument('-e,','--epochs',type = int,default = 2,help='set up number of epochs')
    parser.add_argument('-bs,','--batch_size',type = int,default = 32,help='set batch size')
    parser.add_argument('-is,','--image_size',type = int,default = 150,help='Set image size for the model')
    parser.add_argument('-sc,','--save_checkpoint',type = bool,default = False,help='Set image size for the model')
    parser.add_argument('-pr', '--precision_recall', type = bool, default=False, help='Print precision and recall')
    parser.add_argument('-load_checkpoint', type = bool, default=False, help='Whether to load checlpoint')
    parser.add_argument('-save_summary', type=bool, default = True, help='save summary')
    parser.add_argument('-print_frequency', type = int, default = 2, help='Print frequency for iterations')
    parser.add_argument('-weight_path', type = str, default = 'weights', help='Path to save trained weights')
    parser.add_argument('-checkpoint_path', type = str, default = 'checkpoints', help = 'Path to save checkpoints')
    parser.add_argument('-dump_path', type = str, default = 'dump', help = 'Path to save dump')
    parser.add_argument('-padding', type = str, default = 'SAME', help = 'padding for layers')
    return parser