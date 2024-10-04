"""
Dataset specific selection functions for BreastMNIST
"""
import numpy as np
import pandas as pd


def select_no_annotations_images(dataset,criteria=['remove all images with annotations']):
    """
    A function for selecting images with no annotations and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a no annotations. These include:
        'remove all images with annotations': Remove all images with annotations
        'set_to_train': Put all images with no annotations into the training set
        'set_to_val': Put all images with no annotations into the validation set
        'set_to_test': Put all images with no annotations into the test set
        'make_no_annotations_binary_classifier': Make a binary classifier with all images with no annotations as class 1 and the remaining as class 0
        'make_no_annotations_class': Make a new class for no annotations images without changing the other classes
    dataset_name: str
        The name of the dataset. Default: 'D7P'.

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    annotations_list = np.loadtxt("data/BreastMNIST/manual_annotations/annotations.txt",dtype=str)
    annotations_list = [str(element)+'.jpg' for element in annotations_list]
    no_annotations_data =  dataset['total_df'][~dataset['total_df']['Path'].isin(annotations_list)]

    #Remove all images that contain an annotation (useful for making an ID or OOD dataset)
    if 'remove all images with annotations' in criteria:
        dataset['total_df'] = no_annotations_data 
    
    #Put all images with an annotation into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], no_annotations_data]) if 'train_df' in dataset else no_annotations_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], no_annotations_data]) if 'validation_df' in dataset else no_annotations_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], no_annotations_data]) if 'test_df' in dataset else no_annotations_data
        dataset['total_df'] = dataset['total_df'].drop(no_annotations_data.index)

    return dataset


def select_annotations_images(dataset,criteria=['remove all images without annotations']):
    """
    A function for selecting images with a annotations and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a annotations. These include:
        'remove all images without annotations': Remove all images without a annotations
        'set_to_train': Put all images with a annotations into the training set
        'set_to_val': Put all images with a annotations into the validation set
        'set_to_test': Put all images with a annotations into the test set
        'make_annotations_binary_classifier': Make a binary classifier with all images with a annotations as class 1 and the remaining as class 0
        'make_annotations_class': Make a new class for images with annotations without changing the other classes
    dataset_name: str
        The name of the dataset. Default: 'D7P'.

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    annotations_list = np.loadtxt("data/BreastMNIST/manual_annotations/annotations.txt",dtype=str)
    annotations_list = [str(element)+'.jpg' for element in annotations_list ]
    annotations_data =  dataset['total_df'][dataset['total_df']['Path'].isin(annotations_list)]

    #Remove all images that do not contain a annotations image (useful for making an ID or OOD dataset)
    if 'remove all images without annotations' in criteria:
        dataset['total_df'] = annotations_data 
    
    #Put all images with a annotations into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], annotations_data]) if 'train_df' in dataset else annotations_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], annotations_data]) if 'validation_df' in dataset else annotations_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], annotations_data]) if 'test_df' in dataset else annotations_data
        dataset['total_df'] = dataset['total_df'].drop(annotations_data.index)

    return dataset