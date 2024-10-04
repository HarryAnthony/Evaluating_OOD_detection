import argparse
import numpy as np
import torch


def create_parser():
    """
    Initialise and configure an argument parser.

    Returns:
        parser: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='Evaluate OOD detection method')
    #General arguments
    parser.add_argument('--cuda_device', default='all', type=str,
                        help='Select device to run code, could be device number or "all"')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size (typically order of 2), default: 32')
    parser.add_argument('--verbose',default=True,type=bool, help='verbose')
    parser.add_argument('--seed', default='random', type=str,
                        help='Select experiment seed')
    parser.add_argument('--setting', default='setting1', type=str,
                    help='dataset setting for CheXpert, either setting1, setting2 or setting3')
    #Arguments for training DNNs (for training.py)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('--net_type', default='ResNet18',
                        type=str, help='Type of DNN to train')
    parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
    parser.add_argument('--act_func_dropout', default=0, type=float,
                        help='2D dropout to be applied in a CNN.')
    parser.add_argument('--dataset_seed', default='same', type=str,
                        help='Select seed for seperating the dataset into train, validation and test sets (Default is the same as the experiment).')
    parser.add_argument('--dataset', default='D7P', type=str,
                        help='dataset [D7P,BreastMNIST] (default: D7P).')
    parser.add_argument('--depth', default=28,
                         type=int, help='depth of WideResNet model')
    parser.add_argument('--widen_factor', default=10,
                        type=int, help='width of WideResNet model')
    parser.add_argument('--allow_repeats',default=False,type=bool,
                        help='Allow experiments of the same seed to be repeated (Default: False).')
    parser.add_argument('--optimiser', '-O', default='SGD',
                        help='optimiser method for the model (default: SGD).')
    parser.add_argument('--scheduler', '-Sc', default='MultiStepLR',
                        help='scheduler for the model (default: MultiStepLR).')
    parser.add_argument('--save_model', default='last_epoch', type=str,
                        help='When to save model parameters [best_acc,last_epoch] (default: last_epoch).')
    parser.add_argument('--max_lr', default=1e-2,
                        help='Maxmimum lr which can be reached when using OneCycleLR.')
    parser.add_argument('--act_func', '-Af', default='ReLU',
                        help='The activation function used for the DNN. (default: ReLU)')
    parser.add_argument('--class_selections', '-c_sel', default="{'classes_ID': ['Cardiomegaly'], 'classes_OOD': [], 'atleast_one_positive_class': False,'replace_values_dict':{}}",
                        help='The class selections to be used if the args.setting is not known.')
    parser.add_argument('--demographic_selections', '-d_sel', default="{}",
                        help='The demographic selections to be used if the args.setting is not known.')
    parser.add_argument('--dataset_selections', '-dataset_s', default="{}", 
                        help='The dataset specific selections to be used if the args.setting is not known (default is for CheXpert).')
    parser.add_argument('--train_val_test_split_criteria', '-split_sel', default="{'valSize': 0.2, 'testSize': 0}", 
                        help='The dataset splitting criteria to be used if the args.setting is not known.')
    parser.add_argument('--fold', default=0, type=int,
                        help='The fold to train with when using k-fold cross validation (default: 0).')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='Float for label smoothing (default: 0.0).')
    parser.add_argument('--save_path', default='outputs/saved_models', type=str,
                        help='Path to save the model (Default: outputs/saved_models).')
    parser.add_argument('--resume_training', default=False, type=bool,
                        help='Resume training from a saved model (Default: False).')
    parser.add_argument('--valSize', default=0.1, type=float,
                        help='Validation set size (Default: 0.1).')
    parser.add_argument('--testSize', default=0, type=float,
                        help='Test set size (Default: 0).')
    parser.add_argument('--gradient_clipping', default=False, type=bool,
                        help='Whether to use gradient clipping (Default: False)')
    parser.add_argument('--wandb_args', default=False, type=bool,
                        help='Whether to use wandb to log results (Default: False).')
    parser.add_argument('--initialisation', default='he', type=str,
                        help='Initialisation method of the DNN (Default: He).')
    parser.add_argument('--weight_decay', default=1e-10, type=float,
                        help='weight decay to use for the training (Default: 1e-10).')
    #For selecting OOD dataset (for evaluate_OOD_detection_methods.py)
    parser.add_argument('--ood_class_selections', '-ood_c_sel', default="{'classes_ID': ['All'], 'classes_OOD': []}", 
                        help='The class selections to be used if the args.ood_setting is not known.')
    parser.add_argument('--ood_demographic_selections', '-ood_d_sel', default="{}",
                        help='The demographic selections to be used if the args.ood_setting is not known.')
    parser.add_argument('--ood_dataset_selections', '-ood_dataset_sel', default="{}",
                        help='The dataset specific selections to be used if the args.ood_setting is not known (default is for CheXpert).')
    parser.add_argument('--ood_train_val_test_split_criteria', '-ood_split_sel', default="{'valSize': 0.0, 'testSize': 1.0}",
                        help='The dataset splitting criteria to be used if the args.ood_setting is not known.')
    #Used for selecting the OOD type (for evaluate_OOD_detection_methods.py)
    parser.add_argument('--ood_type', default='different_class',
                        help='Select ood task')
    #Arguments for synthetic OOD (for evaluate_OOD_detection_methods.py)
    parser.add_argument('--synth_artefact', default='square', type=str,
                        help='Form of synthetic artefact.')
    parser.add_argument('--synth_scale', default=(0.1,0.1), type=tuple,
                        help='Percentage of image size to scale synthetic artefact. Default: (0.1,0.1)')
    parser.add_argument('--synth_ratio', default=(1,1), type=tuple,
                        help='Ratio of synthetic artefact. Default: (1,1)')
    parser.add_argument('--synth_value', default=0, type=str,
                        help='Value of synthetic artefact (float) or can be [random_gaussian_noise, random_uniform_noise, foreign_texture, image_replace, image_replace_no_overlap] Default: 0')
    parser.add_argument('--synth_setting', default='random', type=str,
                        help='Location of synthetic artefact in image, can be [random, centred, near_centre, periphery, corners, near_corners, near_periphery]. Default: random')
    parser.add_argument('--synth_noise_mean', default='img_mean', type=str,
                        help='Mean of the Gaussian noise used if synth_value is random_gaussian_noise. Default: img_mean')
    parser.add_argument('--synth_noise_std', default='img_std', type=str,
                        help='Standard deviation of the Gaussian noise used if synth_value is random_gaussian_noise. Default: img_std')
    parser.add_argument('--synth_coarseness', default=1, type=float,
                        help='Coarseness of the noise or foriegn texture of the synthetic artefact (>=1). Default: 1')
    parser.add_argument('--synth_rotation_angle', default=0, type=str,
                        help='Rotation angle of synthetic artefact (float), can be "random". Default: 0')
    parser.add_argument('--synth_foreign_texture', default=torch.tensor(np.kron([[1,0]*5,[0,1]*5]*5, np.ones((10, 10))),dtype=torch.float32),
                        help='Foreign texture of synthetic artefact, should be 2D or 3D')
    parser.add_argument('--synth_gaussian_filter_sigma', default=0, type=float,
                        help='Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.')
    parser.add_argument('--synth_make_transparent', default=False, type=bool,
                        help='Whether to make the synthetic artefact transparent. Default is False.')
    parser.add_argument('--synth_transparency_power', default=5, type=float,
                        help='Power to raise the transparency mask to. Default is 1.')
    parser.add_argument('--synth_triangle_type', default='equilateral', type=str,
                        help='Type of triangle to use for synthetic artefact, can be [equilateral, isosceles, right]. Default is equilateral.')
    parser.add_argument('--synth_polygon_coordinates', default=np.array([(112, 50), (118, 80), (142, 80), (124, 98), (140, 122),(112, 110), (84, 122), (100, 98), (82, 80), (106, 80)]),
                        help='Coordinates of the polygon to be used for synthetic artefact. Default is random.')
    parser.add_argument('--synth_ellipse_parameter', default=1, type=float,
                        help='Ellipcticity of the ring (<=1). Default is 1.')
    parser.add_argument('--synth_ring_width', default=20, type=float,
                        help='Width of the ring. Default is 20.')
    parser.add_argument('--synth_text', default='OOD', type=str,
                        help='Text to be used for synthetic artefact. Default is OOD.')
    parser.add_argument('--synth_font_family', default='sans-serif', type=str,
                        help='Font family to be used for synthetic artefact. Default is sans-serif.')
    #Arguments for selecting the OOD dataset (for evaluate_OOD_detection_methods.py)
    parser.add_argument('--ood_dataset', default='SHVM', type=str,
                        help='Name of dataset to be used for ood.')
    #Arguments for OOD detection methods (for evaluate_OOD_detection_methods.py)
    parser.add_argument('--method', '-m', type=str, default='MCP',
                        help='Method for OOD Detection, one of [MCP (default), ODIN, MCDP, mahalanobis, deepensemble]')
    parser.add_argument('--save_results', default=False, type=bool,
                    help='Boolean whether to save results for OOD detection')
    parser.add_argument('--save_results_micro', default=False,
                    help='Save the micro results (confidence score for each image) for the OOD detection methods')
    parser.add_argument('--filename', default='practise', type=str,
                        help='Name of dataset to be used for ood.')
    parser.add_argument('--evaluate_ID_accuracy', default=False, type=bool,
                        help='Whether to measure the accuracy of the ID test dataset')
    parser.add_argument('--evaluate_OOD_accuracy', default=False, type=bool,
                        help='Whether to measure the accuracy of the OOD test dataset')
    parser.add_argument('--temperature', '-T', default='1', type=float,
                        help='Temperature parameter (used for methods like ODIN, energy)')
    parser.add_argument('--noiseMagnitude', '-eps', default='0', type=float,
                        help='Perturbation epislon (used for ODIN)')
    parser.add_argument('--MCDP_samples', default='10', type=int,
                        help='Samples used for method MCDP')
    parser.add_argument('--deep_ensemble_seed_list','-DESL', default='[]', type=str,
                        help='List of seeds to be used for deep ensemble')
    parser.add_argument('--return_metrics', default=False, type=bool,
                        help='Whether to return AUROC and AUCPR')
    parser.add_argument('--mahalanobis_module', default=-1,
                        help='Module of DNN to apply Mahalanobis distance')
    parser.add_argument('--mahalanobis_feature_combination', default=False, type=bool,
                        help='Combine the distances from different modules into one distance')
    parser.add_argument('--mahalanobis_alpha', default=None, 
                        help='List of weights to be used for combining the distances from different modules. Should be same length as mahalanobis modules')
    parser.add_argument('--mahalanobis_RMD', default=False, type=bool,
                        help='Whether to use relative Mahalanobis (True) or Mahalanobis (False). Default is False.')
    parser.add_argument('--mahalanobis_preprocess', default=False, type=bool,
                        help='Whether to preprocess an image with FGSM before calculating Mahalanobis distance (Default: False).')
    parser.add_argument('--MBM_type', default='MBM', 
                        help='The type of Multi-Branch Mahalanobis to use, should be MBM or MBM_act_func_only (Default: MBM).')
    parser.add_argument('--ReAct_percentile', default=0.7, 
                        help='Percentile of training activations at which to truncate (used for ReAct).')
    parser.add_argument('--DICE_sparsification_param', default=0.9, 
                        help='Sparsification parameter to use for Directed Sparsification (DICE).')
    parser.add_argument('--GRAM_power', default=[1,3,5,7], 
                help='A list of integer powers to apply to the GRAM matrices.')
    parser.add_argument('--GRAM_layer_weights', default=None,
                help='Array of weights for each layer when applying GRAM.')
    parser.add_argument('--grad_layer', default=-1,
                help='Layer to calculate the GradNorm for.')
    parser.add_argument('--gradnorm_summation_method', default='l1',
                help='The method to sum the gradients for GradNorm. Either l1 or l2.')
    
    return parser
