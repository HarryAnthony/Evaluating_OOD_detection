import torchvision.transforms as T
import torch
from source.dataloaders.Dataset_class import Dataset_class
from source.dataloaders.BreastMNIST_dataloader import select_no_annotations_images, select_annotations_images
from source.util.general_utils import DefaultDict

Database_class = Dataset_class

#Training parameters
num_epochs = 400
momentum = 0.9 
weight_decay = 1e-4
lr_milestones = [int(num_epochs*0.75),int(num_epochs*0.9)]
lr_gamma = 0.2
criterion = 'CrossEntropyLoss'
initialisation_method = 'he'

# network architecture
dropout = 0.3 
depth = 28
widen_factor = 10

# data parameters
image_size = 224

# location of data
root = '/media/scat9348/disk2/datasets/breastmnist'
loader_root = '/media/scat9348/disk2/datasets/breastmnist/'
df_name = 'BreastMNIST'


def database_specific_selections(dataset,selections={},**kwargs):
    """
    Make selections on the dataset which are specific to the BreastMNIST dataset.

    Parameters
    ----------
    dict
        A dictionary containing the training, validation and test sets.
    selections : dict, optional
        A dictionary containing the selections to be made on the dataset. The default is {}.
        The keys of the dictionary are the names of the selections and the values are the criteria for the selection.
        The possible selections are:
            'annotations_selection': selects images with annnotations using the manual annotations.
            'no_annotations_selection': selects images with no annotations using the manual annotations.

    Returns
    -------
    dict
        A dictionary containing the training, validation and test sets after the selections have been made.
    """
    if 'annotations_selection' in selections.keys():
        dataset = select_annotations_images(dataset,criteria=selections['annotations_selection'])
    if 'no_annotations_selection' in selections.keys():
        dataset = select_no_annotations_images(dataset,criteria=selections['no_annotations_selection'])

    return dataset
    

#If setting is not known, then will use the default transform with mean and std of the dataset
transform_train = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=15), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur((3,3), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=0.32283312,std=0.2032362),
            lambda x: x.expand(3,-1,-1)

            ]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=35), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur((3,3), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=0.32283312,std=0.2032362),
            lambda x: x.expand(3,-1,-1)
            ]),
            
            })

transform_test = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=0.32283312,std=0.2032362),
            lambda x: x.expand(3,-1,-1)
            ]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=0.32283312,std=0.2032362),
            lambda x: x.expand(3,-1,-1)
            ]),
            
            })

#Pre-made dataset selection settings for the BreastMNIST dataset
dataset_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['All'], 'classes_OOD': []},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'no_annotations_selection':['remove all images with annotations']},
                                           'train_val_test_split_criteria': {'valSize': 0.1, 'testSize': 0}},
                                        }


OOD_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['All'], 'classes_OOD': []},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'annotations_selection':['remove all images without annotations']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                                           }


#The classes of the BreastMNIST dataset
classes = ('Normal', 'Benign', 'Malignant')