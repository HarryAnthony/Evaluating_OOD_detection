import torchvision.transforms as T
import torch
from source.dataloaders.Dataset_class import Dataset_class
from source.dataloaders.skin_lesion_dataloader import select_ruler_images, select_no_ruler_images
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
root = '../../data/D7P'
loader_root = '../../data/D7P/'
df_name = 'D7P'

def database_specific_selections(dataset,selections={},**kwargs):
    """
    Make selections on the dataset which are specific to the CheXpert dataset.

    Parameters
    ----------
    dict
        A dictionary containing the training, validation and test sets.
    selections : dict, optional
        A dictionary containing the selections to be made on the dataset. The default is {}.
        The keys of the dictionary are the names of the selections and the values are the criteria for the selection.
        The possible selections are:
            'ruler_selection': selects images with a ruler using the manual annotations.
            'no_ruler_selection': selects images with no rulers using the manual annotations.

    Returns
    -------
    dict
        A dictionary containing the training, validation and test sets after the selections have been made.
    """
    if 'no_ruler_selection' in selections.keys():
        dataset = select_no_ruler_images(dataset,criteria=selections['no_ruler_selection'],dataset_name='D7P')
    if 'ruler_selection' in selections.keys():
        dataset = select_ruler_images(dataset,criteria=selections['ruler_selection'],dataset_name='D7P')
    return dataset


#If setting is not known, then will use the default transform with mean and std of the CheXpert dataset
transform_train = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=15), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2),
            T.GaussianBlur((3,3), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.72662437,0.6243302,0.5687489],
                        std=[0.22084126,0.22352666,0.22693515]),

            ]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=15), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur((3,3), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.72662437,0.6243302,0.5687489],
                        std=[0.22084126,0.22352666,0.22693515]),
            ]),
            })

transform_test = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.72662437,0.6243302,0.5687489],
                        std=[0.22084126,0.22352666,0.22693515])
            ]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.72662437,0.6243302,0.5687489],
                        std=[0.22084126,0.22352666,0.22693515]),
            ]),
            })

#Pre-made dataset selection settings for the D7P dataset
dataset_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['nevus'], 'classes_OOD': [],'atleast_one_positive_class': False},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'no_ruler_selection':['remove all images with ruler']},
                                           'train_val_test_split_criteria': {'valSize': 0.1, 'testSize': 0}},
                                            }


OOD_selection_settings = {'setting1': {'class_selections' :  {'classes_ID': ['nevus'], 'classes_OOD': [],'atleast_one_positive_class': False},
                                           'demographic_selections' : {},
                                           'dataset_selections': {'ruler_selection':['remove all images without ruler','type_1_only']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                                           }


#The classes of the D7P dataset
classes = ('pigmented_benign_keratosis','nevus','vascular_lesion','basal_cell_carcinoma','melanoma','dermatofibroma')