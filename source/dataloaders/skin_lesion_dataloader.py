"""
Dataset specific selection functions for skin_lesion datasets
"""
import numpy as np
import pandas as pd


def select_no_ruler_images(dataset,criteria=['remove all images with ruler'],dataset_name='D7P'):
    """
    A function for selecting images with no ruler and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a no ruler. These include:
        'remove all images with ruler': Remove all images with ruler
        'set_to_train': Put all images with no ruler into the training set
        'set_to_val': Put all images with no ruler into the validation set
        'set_to_test': Put all images with no ruler into the test set
        'make_no_ruler_binary_classifier': Make a binary classifier with all images with no ruler as class 1 and the remaining as class 0
        'make_no_ruler_class': Make a new class for no ruler images without changing the other classes
    dataset_name: str
        The name of the dataset. Default: 'D7P'.

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    ruler_list_type_1 = [] #Type 1 rulers are black grid rulers
    ruler_list_type_2 = [] #Type 2 rulers are white grid rulers
    ruler_list_type_3 = [] #Type 3 rulers are remaining rulers not in type 1 or 2
    if 'allow_type_1' not in criteria:
        ruler_list_type_1 = np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_1.txt", dtype=str)
    if 'allow_type_2' not in criteria:
        ruler_list_type_2 = np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_2.txt", dtype=str)
    if 'allow_type_3' not in criteria:
        ruler_list_type_3 = np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_3.txt", dtype=str)
    ruler_list_uncertain = np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/uncertain.txt", dtype=str)
    ruler_list = np.concatenate([ruler_list_type_1,ruler_list_type_2,ruler_list_type_3,ruler_list_uncertain])

    #Get all images with no ruler
    nr_data =  dataset['total_df'][~dataset['total_df']['Path'].isin(ruler_list)]

    #Remove all images that contain a ruler (useful for making an ID or OOD dataset)
    if 'remove all images with ruler' in criteria:
        dataset['total_df'] = nr_data 
    
    #Put all images with no ruler into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], nr_data]) if 'train_df' in dataset else nr_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], nr_data]) if 'validation_df' in dataset else nr_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], nr_data]) if 'test_df' in dataset else nr_data
        dataset['total_df'] = dataset['total_df'].drop(nr_data.index)

    #Set all images with no ruler into class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_no_ruler_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in nr_data else 0)
    elif 'make_no_ruler_class' in criteria: #Make a new class for no ruler
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in nr_data})

    return dataset


def select_ruler_images(dataset,criteria=['remove all images without ruler'],dataset_name='D7P'):
    """
    A function for selecting images with a ruler and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a ruler. These include:
        'remove all images without ruler': Remove all images without a ruler
        'set_to_train': Put all images with a ruler into the training set
        'set_to_val': Put all images with a ruler into the validation set
        'set_to_test': Put all images with a ruler into the test set
        'make_ruler_binary_classifier': Make a binary classifier with all images with a ruler as class 1 and the remaining as class 0
        'make_ruler_class': Make a new class for ruler without changing the other classes
    dataset_name: str
        The name of the dataset. Default: 'D7P'.

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    #Decide which ruler types to allow. 
    # - Type 1 rulers are black grid rulers
    # - Type 2 rulers are white opaque rulers
    # - Type 3 rulers are remaining rulers not in type 1 or 2
    ruler_lists = {
    'type_1': np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_1.txt", dtype=str),
    'type_2': np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_2.txt", dtype=str),
    'type_3': np.loadtxt("data/"+str(dataset_name)+"/manual_annotations/ruler_type_3.txt", dtype=str),
    }

    #criteria = ['type_1_only']  # Example criteria, you can replace it with the actual criteria
    ruler_list = np.concatenate([
        ruler_lists[type] for type in ruler_lists if (f'type_{type[-1]}_only' in criteria)
    ])
    ruler_data =  dataset['total_df'][dataset['total_df']['Path'].isin(ruler_list)]

    #Remove all images that do not contain a ruler image (useful for making an ID or OOD dataset)
    if 'remove all images without ruler' in criteria:
        dataset['total_df'] = ruler_data 
    
    #Put all images with a ruler into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], ruler_data]) if 'train_df' in dataset else ruler_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], ruler_data]) if 'validation_df' in dataset else ruler_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], ruler_data]) if 'test_df' in dataset else ruler_data
        dataset['total_df'] = dataset['total_df'].drop(ruler_data.index)

    #Set all images with a ruler to class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_ruler_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in ruler_list else 0)
    elif 'make_ruler_class' in criteria: #Make a new class for ruler
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in ruler_list})

    return dataset

