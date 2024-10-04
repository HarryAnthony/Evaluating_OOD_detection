import os
from types import SimpleNamespace
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import wandb
from source.util.common_args import create_parser
from sklearn.model_selection import train_test_split
from source.util.general_utils import select_cuda_device
from source.util.processing_data_utils import get_dataset_config, get_weighted_dataloader, get_dataloader, get_dataset_selections
from source.util.training_utils import select_experiment_seed, get_class_weights, record_model, get_network_architecture, initialise_network, get_criterion, get_optimiser_scheduler, set_activation_function
from source.util.evaluate_network_utils import load_net
from source.util.Select_dataset import Dataset_selection_methods
from source.util.Train_DNN import Train_DNN


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()


#Select a GPU to use
use_cuda = select_cuda_device(args.cuda_device)
pin_memory = use_cuda
device_count = torch.cuda.device_count()


#Get configurations for a given dataset
dataset_name = args.dataset
cf, load_dataset = get_dataset_config(args.dataset)
savepath = args.save_path
num_epochs = cf.num_epochs
batch_size = args.batch_size
resize = cf.image_size
requires_split = 0 


# setup checkpoint and experiment tracking
if not os.path.isdir(savepath):
    os.mkdir(savepath)
save_point = savepath + '/'+str(args.dataset)+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)


#Select a seed for the experiment
args.allow_repeats = True if args.resume_training == True else args.allow_repeats
seed = select_experiment_seed(args.seed,savepath+'/'+str(args.dataset),allow_repeats=args.allow_repeats)
if args.dataset_seed == 'same':
    dataset_seed = int(seed)
else:
    if isinstance(args.dataset_seed,int) == False and args.dataset_seed.isdigit() == False:
        raise ValueError('Seed must be an integer')
    dataset_seed = int(args.dataset_seed)


#Load the dataset
if load_dataset == 1: #Process the dataset which does not need splitting

    classes_ID = cf.classes
    classes_OOD = []
    num_classes = len(classes_ID)

    #Split data into test, validation and training sets
    df_train_full = cf.train_ID
    train_idx, val_idx = train_test_split(list(range(len(df_train_full))), test_size=args.valSize,stratify=df_train_full.targets,random_state=int(dataset_seed))

    #Get dataloader for test and validation sets
    df_train = torch.utils.data.Subset(df_train_full, train_idx)
    df_validation = torch.utils.data.Subset(df_train_full, val_idx)
    df_validation.transform = cf.transform_test
    args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'pin_memory': use_cuda, 'device_count': device_count}
    trainloader = get_dataloader(args=SimpleNamespace(**args_train), dataset=df_train)
    class_selections,demographic_selections,dataset_selections,train_val_test_split_criteria = {},{},{},{}

else: #Process the dataset which needs splitting
    
    requires_split = 1

    # load dataframe 
    path_train = os.path.join(cf.root, 'train.csv')
    data_temp = pd.read_csv(path_train)
    path_train_valid = os.path.join(cf.root, 'valid.csv')
    data_valid = pd.read_csv(path_train_valid)
    if os.path.exists(cf.root+'/test.csv'):
        path_train_test = os.path.join(cf.root, 'test.csv')
        data_test = pd.read_csv(path_train_test)
        data = pd.concat([data_temp,data_valid,data_test])
    else:
        data = pd.concat([data_temp,data_valid])

    #Applies selections on the data to get desired training and validation datasets
    Dataset_selection = Dataset_selection_methods(data,cf,mode='train',verbose=args.verbose)
    class_selections, demographic_selections, dataset_selections, train_val_test_split_criteria = get_dataset_selections(cf,args,dataset_seed)
    dataset = Dataset_selection.apply_selections(class_selections=class_selections,
                                                 demographic_selections=demographic_selections,
                                                 dataset_selections=dataset_selections,
                                                 train_val_test_split_criteria=train_val_test_split_criteria)


    assert 'train_df' in dataset.keys(), 'train_df not in dataset, check mode of Dataset_selection_methods is set to "train"'
    assert 'validation_df' in dataset.keys(), 'validation_df not in dataset, check mode of Dataset_selection_methods is set to "train"'

    
    #Get dataloader for test and validation sets, which are class balanced
    df_train = cf.Database_class(cf.loader_root, dataset['train_df'], cf.transform_train[args.setting])
    df_validation = cf.Database_class(cf.loader_root, dataset['validation_df'], cf.transform_test[args.setting])
    # trainloader uses a weighted sampler as training data is class imbalanced
    train_weights = get_class_weights(dataset['train_df']) #Get weights for each class 
    args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'root': cf.loader_root, 'pin_memory': use_cuda, 'device_count': device_count}

    trainloader = get_weighted_dataloader(args=SimpleNamespace(**args_train), dataset=df_train, weights=train_weights)
    classes_ID = dataset['classes_ID']
    classes_OOD = class_selections['classes_OOD']

    num_classes = int(dataset['train_df']['class'].max()) + 1
    assert num_classes > 1, 'There must be more than one class to train the neural network'


args_validation = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'pin_memory': use_cuda, 'device_count': device_count}
validationloader = get_dataloader(args=SimpleNamespace(**args_validation), dataset=df_validation)

 
# Displays information about the experiment
if args.verbose == True:
    print('\nExperiment details:')
    print('| Running experiment seed: {}'.format(seed))
    print(f'| Classes ID: {classes_ID}')
    print(f'| Length of train dataset: {len(df_train)}')
    print(f'| Length of validation dataset: {len(df_validation)}')


if args.resume_training != True: #Get the network architecture and initialise the network
    net, file_name = get_network_architecture(args,num_classes,cf.df_name)
    net = set_activation_function(net,activation_function=args.act_func)
    net = initialise_network(net,initialisation_method=args.initialisation)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

else: #Load a saved model
    if args.seed == 'random':
        raise ValueError('Cannot resume training with a random seed')
    net, net_dict, cf_load_net = load_net(args.seed,use_cuda=use_cuda)
    file_name = net_dict['pathname']+'_resume_training'
    assert cf_load_net.df_name == cf.df_name, 'Dataset in the saved model does not match the current experiment'
    assert net_dict['num_classes'] == num_classes, 'Number of classes in the saved model does not match the number of classes in the current experiment'
    
criterion = get_criterion(criterion_name=cf.criterion,label_smoothing=args.label_smoothing)
optimiser, scheduler = get_optimiser_scheduler(net,args,cf,trainloader,num_epochs)

wandb_dict = {'bool':args.wandb_args}
if args.wandb_args:
    # Set up logging and pass params to wandb
    wandb.init(project="OOD_detection",
            name=f'{file_name}_{seed}', config=args)
    wandb.config.batch_size = batch_size
    wandb.config.file_name = file_name
    wandb.config.image_size = resize
    wandb.watch(net)
    wandb_dict['wandb'] = wandb


#Displays information about model training
if args.verbose == True:
    print('\nTraining model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimiser = ' + str(args.optimiser))
    print('| Scheduler = ' + str(args.scheduler))
    print('| Batch size = ' + str(batch_size))


#Record the model details
Model_details = {
    "model_index": len(pd.read_csv('outputs/saved_models/model_list.csv'))+1,
    "model_name": f"{file_name}-{str(seed)}",
    "seed": seed,
    "dataset_name": dataset_name,
    "setting": args.setting[-1],
    "model_type": args.net_type,
    "depth": args.depth,
    "widen_factor": args.widen_factor,
    "dropout": args.dropout,
    "DUQ": 0,
    "validation_accuracy": 0,
    "requires_split": requires_split,
    "dataset_seed": dataset_seed,
    "activation_function": args.act_func,
    "class_selections": class_selections,
    "demographic_selections": demographic_selections,
    "dataset_selections": dataset_selections,
    "train_val_test_split_criteria": train_val_test_split_criteria,
    "num_classes": num_classes,
    "activation_function_dropout": args.act_func_dropout,
    "learning_rate": args.lr,
    "optimiser": args.optimiser,
    "scheduler": args.scheduler,
    "max_lr": args.max_lr,
    "label_smoothing": args.label_smoothing,
    "batch_size": args.batch_size,
    "initialisation_method": cf.initialisation_method,
    "save_path": args.save_path,
    "criterion": cf.criterion,
    "save_model": args.save_model
    }
record_model('outputs/saved_models/model_list.csv',list(Model_details.values()))


#Train the model
training_dict = {'net': net,
                 'trainloader': trainloader,
                 'validationloader': validationloader,
                 'optimiser': optimiser,
                 'scheduler': scheduler,
                 'criterion': criterion,
                 'scheduler_name': args.scheduler,
                 'num_epochs': num_epochs,
                 'use_cuda': use_cuda,
                 'verbose': args.verbose,
                 'save_model_mode': args.save_model,
                 'save_point': save_point,
                 'filename': file_name,
                 'seed': seed,
                 'wandb_dict': wandb_dict,
                 }
train_model = Train_DNN(training_dict)
train_model(gradient_clipping=args.gradient_clipping)