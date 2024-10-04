"""
Helper functions for evaluating the network and its OOD detection performance.
"""
import pandas as pd
import torch
import numpy as np 
from torch.autograd import Variable
import sklearn.metrics as skm
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from itertools import combinations
from torchvision.models import resnet18, vgg11, vgg16_bn, resnet34, resnet50, vgg16
from source.models.wide_resnet import Wide_ResNet
from source.util.training_utils import set_activation_function, add_dropout_network_architechture
from source.util.general_utils import print_progress, try_literal_eval
from torch.utils.data import DataLoader,Dataset


def check_net_exists(seed, verbose=True, get_output=False):
    """
    Check if the net exists in the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to check.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    get_output : bool, optional
        Whether to return the net information. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    None if get_output is False, otherwise returns a dictionary containing the net information.
    """
    model_list = pd.read_csv('outputs/saved_models/model_list.csv')  # Get list of nets

    matching_models = model_list[model_list['Seed'] == int(seed)]
    if len(matching_models) == 0:
        raise Exception('Experiment seed is not in the list of known experiments')
    model = matching_models.iloc[0]

    if verbose:
        print('Model database: {}\nModel setting: {}\nModel type: {}\nModel widen factor x depth: {} x {}\nDropout: {}\n'.format(
            model['Database'], model['Setting'], model['Model_type'], model['Widen_factor'],
            model['Depth'], model['Dropout']))

    if get_output:
        net_info = {
            'Model pathname': model['Model_name'],
            'Model database': model['Database'],
            'Model setting': 'setting'+str(int(model['Setting'])),
            'Model architecture': model['Model_type'],
            'Model widen factor': model['Widen_factor'],
            'Model depth': model['Depth'],
            'Model activation': model['Activation_function'],
            'Dropout': model['Dropout'],
            'Act_func_Dropout': model['act_func_dropout'],
            'DUQ': model['DUQ'],
            'Requires split': model['requires_split'],
            'Dataset seed': model['Dataset_seed'],
            'class_selections': model['class_selections'],
            'demographic_selections': model['demographic_selections'],
            'dataset_selections': model['dataset_selections'],
            'train_val_test_split_criteria': model['train_val_test_split_criteria'],
            'num_classes': int(model['num_classes'])
        }

        return net_info
    return None


def load_net(seed,verbose=True,use_cuda=True):
    """
    Load the net from the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to load.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    use_cuda : bool, optional
        Whether to use cuda. The default is True.
    ensemble : bool, optional
        Whether the net is being used for an ensemble. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    net : torch.nn.Module
        The loaded net.
    net_dict : dict
        A dictionary containing the net information.
    cf : config file
        The configuration file for the dataset.

    """
    net_info = check_net_exists(seed,verbose=verbose,get_output=True)

    net_dict = {'Requires split': net_info['Requires split'],
                'setting': net_info['Model setting']}

    #Get configuration for given datasets
    if net_info['Model database'] == 'cifar10':
        from source.config import cifar10 as cf_cifar10
        cf = cf_cifar10
    elif net_info['Model database'] == 'D7P':
        from source.config import D7P as cf_D7P
        cf = cf_D7P
    elif net_info['Model database'] == 'BreastMNIST':
        from source.config import BreastMNIST as cf_BreastMNIST
        cf = cf_BreastMNIST

    #Get the classes in and out
    if net_info['Requires split']:
        net_dict['class_selections'] = turn_str_into_dict(net_info['class_selections'])
        net_dict['demographic_selections'] = turn_str_into_dict(net_info['demographic_selections'])
        net_dict['dataset_selections'] = turn_str_into_dict(net_info['dataset_selections'])
        net_dict['train_val_test_split_criteria'] = turn_str_into_dict(net_info['train_val_test_split_criteria'])
        net_dict['classes_ID'] = net_dict['class_selections']['classes_ID']
        net_dict['classes_OOD'] = net_dict['class_selections']['classes_OOD']
    else:
        net_dict['classes_ID'] = cf.classes
        net_dict['classes_OOD'] = []

    net_dict['num_classes'] = net_info['num_classes']

    if verbose:
        print('| Preparing '+net_info['Model database']+' test with the following classes: ')
        print(f"| Classes ID: {net_dict['classes_ID']}")
        print(f"| Classes OOD: {net_dict['classes_OOD']}\n")

    #Select the network architecture
    model_architecture_dict = {
    'wide-resnet': (Wide_ResNet, ['Model depth', 'Model widen factor', 'Dropout']),
    'ResNet18': (resnet18, []),
    'ResNet34': (resnet34, []),
    'ResNet50': (resnet50, []),
    'vgg11': (vgg11, []),
    'vgg16_bn': (vgg16_bn, []),
    'vgg16': (vgg16, []),
    }
    model_architecture = net_info['Model architecture']
    model_func, model_args = model_architecture_dict.get(model_architecture, (None, None))
    if model_func is None:
        raise ValueError('Invalid model architecture')
    
    #Load the network architecture
    kwargs = {'num_classes': int(net_dict['num_classes'])}
    for arg in model_args:
        kwargs[arg] = int(net_info[arg])
    net = model_func(**kwargs)
    net = add_dropout_network_architechture(net,net_info)
    net_dict['file_name'] = f"{str(net_info['Model architecture'])}-{int(net_info['Model depth'])}x{net_info['Model widen factor']}_{str(net_info['Model database'])}-{int(seed)}"
    net_dict['save_dir'] = os.path.join("outputs", f"{net_info['Model database']}_{net_info['Model setting']}")
    net_dict['pathname'] = net_info['Model pathname']

    # Model setup
    assert os.path.isdir('outputs/saved_models'), 'Error: No saved_models directory found!'
    if use_cuda:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database']+'/'+net_info['Model pathname']+'.pth')
    else:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database']+'/'+net_info['Model pathname']+'.pth',map_location='cpu')
    #Apply parameters and activation function to the network
    params = {}
    for k_old in checkpoint.keys():
        k_new = k_old.replace('module.', '')
        params[k_new] = checkpoint[k_old]
    net.load_state_dict(params)
    net = set_activation_function(net,net_info['Model activation'])

    net_dict['act_func_dropout_rate'] = net_info['Act_func_Dropout']
    net_dict['net_type'] = net_info['Model architecture']

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    return net, net_dict, cf


def turn_str_into_dict(string):
    """
    Turn a string into a dictionary.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    convert_dict : dict
        The converted dictionary.
    """
    string = string.replace('np.nan', '\"null\"') #Replace np.nan with null to prevent errors
    string = string.replace('nan', '\"null\"')

    convert_dict = try_literal_eval(string)

    if 'replace_values_dict' in convert_dict.keys():
        if 'null' in convert_dict['replace_values_dict'].keys():
            convert_dict['replace_values_dict'][np.nan] = convert_dict['replace_values_dict']['null']
            convert_dict['replace_values_dict'].pop('null')
 
    return convert_dict
                

def evaluate_ood_detection_method(method,net,idloader,oodloader,return_metrics=False,**kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    method : str
        The name of the OOD detection method.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.

    Raises
    ------
    ValueError
        If the method is invalid.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """
    from source.methods import mcp, odin, mcdp, deepensemble, mahalanobis, energy, ReAct, MaxLogit, logit_KL_div, GRAM, gradnorm, entropy, jointEnergy, DICE, mcdp_mi

    OOD_detection_dict = {'MCP': {'function': mcp.evaluate, 'name': ['MCP']},
                           'ODIN': {'function': odin.evaluate, 'name': ['ODIN']},
                           'MCDP': {'function': mcdp.evaluate, 'name': ['MCDP']},
                           'deepensemble': {'function': deepensemble.evaluate, 'name': ['Deep_ensemble']},
                            'Mahalanobis' : {'function': mahalanobis.evaluate, 'name': ['Mahalanobis']},
                            'MBM':  {'function': mahalanobis.evaluate_MBM, 'name': ['MBM']},
                            'energy': {'function': energy.evaluate, 'name': ['Energy']},
                            'ReAct': {'function': ReAct.evaluate, 'name': ['ReAct']},
                            'MaxLogit': {'function': MaxLogit.evaluate, 'name': ['MaxLogit']},
                            'KL_div': {'function': logit_KL_div.evaluate, 'name': ['KL_div']},
                            'GRAM': {'function': GRAM.evaluate, 'name': ['GRAM']},
                            'gradnorm': {'function': gradnorm.evaluate, 'name': ['GradNorm']},
                            'entropy': {'function': entropy.evaluate, 'name': ['Entropy']},
                            'jointEnergy': {'function': jointEnergy.evaluate, 'name': ['JointEnergy']},
                            'DICE': {'function': DICE.evaluate, 'name': ['DICE']},
                            'MCDP_MI': {'function': mcdp_mi.evaluate, 'name': ['MCDP_Mutual_Information']},
    }
    
    if method not in OOD_detection_dict.keys():
        raise ValueError(f'Invalid OOD detection method: {method}')
    kwargs['OOD_dict'] = OOD_detection_dict[method]
    
    if return_metrics == True:
        AUROC, AUCPR = ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, return_metrics=True,**kwargs)
        return AUROC, AUCPR
    ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, **kwargs)
    

def ood_evaluation(ood_detection_method, net, idloader, oodloader, verbose=True, save_results=False, save_results_micro=False, save_dir=None, return_metrics=False, use_cuda=True, filename='', **kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    ood_detection_method : dict
        The dictionary of the OOD detection method and the methods name.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    verbose : bool, optional
        Whether to print the AUROC and AUCPR. The default is True.
    save_results : bool, optional
        Whether to save the results in textfiles. The default is False.
    save_dir : str, optional
        The directory to save the results, requires save_results to be True. The default is None.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.
    filename : str, optional
        The filename to save the results as, requires save_results to be True. The default is ''.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """
    confidence_id_ood = ood_detection_method['function'](net, idloader, oodloader, **kwargs)

    ood_detection_method['name'] = kwargs['OOD_dict']['name'] if ood_detection_method['name'] != kwargs['OOD_dict']['name'] else ood_detection_method['name']
    confidence_id_ood = [confidence_id_ood] if isinstance(confidence_id_ood[0][0],(float,int,np.float32)) == True else confidence_id_ood
    OOD_detection_method_scores = []

    if save_results_micro:
        idloader_f, oodloader_f = loader_with_paths(idloader,oodloader)
        net.eval()    
        [id_names,id_logits_list,id_labels_list,id_correct_list,id_predicted_list],[ood_names,ood_logits_list,ood_labels_list,ood_correct_list,ood_predicted_list] = get_image_micro_results(idloader_f,oodloader_f,net,verbose=False,use_cuda=use_cuda)
        
        metrics_filename_ID = "Micro_metrics_ID%s.txt" % (filename) if (len(ood_detection_method['name'])!=1) else "Micro_metrics_ID_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f1_path = os.path.join(str(save_dir), str(metrics_filename_ID))
        with open(f1_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['OOD detection method', 'Image', 'Metric', 'Correct', 'Target', 'Predicted'])

        metrics_filename_OOD = "Micro_metrics_OOD%s.txt" % (filename) if (len(ood_detection_method['name'])!=1) else "Micro_metrics_OOD_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f2_path = os.path.join(str(save_dir), str(metrics_filename_OOD))
        with open(f2_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['OOD detection method', 'Image', 'Metric', 'Correct','Target', 'Predicted'])

    for idx, (id, ood) in enumerate(confidence_id_ood):
        AUROC, AUCPR = get_AUROC_AUCPR(id, ood)
        OOD_detection_name = ood_detection_method['name'][idx]
        OOD_detection_method_scores.append([OOD_detection_name,AUROC, AUCPR])

        if verbose:
            print(OOD_detection_name, 'AUROC:', AUROC, 'AUCPR:', AUCPR)

        if save_results_micro:
            with open(f1_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                for image_name,metric,correct_bool,target,predicted in zip(id_names,id,id_correct_list,id_labels_list,id_predicted_list):
                    writer.writerow([OOD_detection_name,image_name,metric,correct_bool,target,predicted])
            with open(f2_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                for image_name,metric,correct_bool,target,predicted in zip(ood_names,ood,ood_correct_list,ood_labels_list,ood_predicted_list):
                    writer.writerow([OOD_detection_name,image_name,metric,correct_bool,target,predicted])

    if save_results:  
        metrics_filename = "Macro_metrics%s.txt" % (filename) if (len(ood_detection_method['name'])!=1) else "Macro_metrics_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f4_path = os.path.join(save_dir, metrics_filename)

        with open(f4_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['OOD detection method', 'AUROC', 'AUCPR'])
            for (OOD_method_name, AUROC, AUCPR) in OOD_detection_method_scores:
                writer.writerow([OOD_method_name, AUROC, AUCPR])

    if return_metrics:
        return OOD_detection_method_scores[0][1], OOD_detection_method_scores[0][2]
    
    
def get_softmax_score(inputs,net,use_cuda=True,required_grad=False,softmax_only=False,temper=1,**kwargs):
    """
    Classify inputs using a given neural network and output the softmax scores.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    required_grad : bool, optional
        Whether to require gradients for the inputs. The default is False.
    softmax_only : bool, optional
        Whether to only output the softmax scores. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    softmax_score : torch.Tensor
        The softmax outputs of the neural network.
    """
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=required_grad)
    outputs = net(inputs)
    softmax_score = softmax(outputs,temper=temper)  #Convert outputs into softmax
    if softmax_only == True:
        return softmax_score
    return outputs, inputs, softmax_score


def get_softmax_score_report_accuracy(inputs,targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_correct_list=False,**kwargs):
    """
    Classify inputs with a given neural network, output the softmax scores and report the accuracy of the classifier.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    targets : torch.Tensor
        The targets of the inputs.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct_list : list
        The list of correct predictions.
    predicted_list : list
        The list of predicted labels.
    required_grad : bool, optional
        Whether requireS gradients for the inputs. The default is False.
    required_correct_list : bool, optional
        Whether requireS the correct list. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    hidden : torch.Tensor
        The hidden layer outputs of the neural network.
    total : int
        The total number of predictions.
    """
    outputs, inputs, softmax_score = get_softmax_score(inputs,net,use_cuda=use_cuda,**kwargs)

    if use_cuda:
        targets = targets.cuda()
    targets = Variable(targets)

    with torch.no_grad():
            logits_list.append(outputs.data)
            labels_list.append(targets.data)

    if required_correct_list:
        #Compare classifier outputs to targets to get accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_list.extend(predicted.eq(targets.data).cpu().tolist())
        predicted_list.extend(predicted.cpu().tolist())
        return outputs, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list

    return outputs, inputs, softmax_score, logits_list, labels_list


def calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion,verbose=True):
    """
    Calculate the accuracy and AUC of an ood_method.

    Parameters
    ----------
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    correct_list : list
        The list of correct predictions.
    confidence_list : list
        The list of confidence scores.
    ece_criterion : torch.nn.Module
        The ECE criterion.
    verbose : bool, optional
        Whether to print the accuracy. The default is False.

    Returns
    -------
    acc : float
        The accuracy of the classifier.
    acc_list : float
        The accuracy of the classifier.
    auroc_classification : float
        The AUROC of the classifier.
    """
    #Calculate the accuracy
    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece = ece_criterion(logits, labels)
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))

    pred_probs_total = combine_arrays([confidence_list,correct_list])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    confidence_list = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    correct_list = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])


    from sklearn.metrics import balanced_accuracy_score
    bal_acc = balanced_accuracy_score(torch.argmax(logits,dim=1).cpu(), labels.cpu())


    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)


    if verbose:
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
        print(f'| ECE: {ece.item()}')
        print(f'| Acc list: {acc_list}')
        print(f'| AUROC classification: {auroc_classification}')
        print(f'| Balanced accuracy : {bal_acc}')


    return acc, auroc_classification, ece.item(), bal_acc


def softmax(outputs, temper=1):
    """
    Calculate the softmax using the outputs of a neural network.

    Parameters
    ----------
    outputs : torch.Tensor
        The outputs of a neural network.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    """
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs/temper) / np.sum(np.exp(nnOutputs/temper), axis=1, keepdims=True)
    return nnOutputs


def evaluate_accuracy(net,loader,verbose=True,use_cuda=True,save_results=False,save_dir='',filename='ID',return_outputs=False):
    ece_criterion = ECELoss()
    if use_cuda:
        ece_criterion.cuda()
    net.eval()
    net.training = False
    correct, total = 0, 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = np.array([])
    correct_list = []
    predicted_list = []

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    l = len(loader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (inputs, targets) in enumerate(loader):
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        with torch.no_grad():
            _, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_grad=False,required_correct_list=True)
        confidence_list = np.concatenate([confidence_list,np.max(softmax_score,axis=1)])

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    labels_list_array = [value.item() for tensor in labels_list for value in tensor.view(-1)]

    set_style(fontsize=12)

    cm = confusion_matrix(labels_list_array, predicted_list) #normalize='false'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No nevus','nevus']) #['benign_keratosis','nevus','vascular_lesion','basal_cell_carcinoma','melanoma','dermatofibroma','actinic_keratosis'])
    disp.plot()
    disp.plot(xticks_rotation=90)
    plt.title('No Devices ID, Pacemaker OOD task (OOD data)')
    plt.tight_layout()
    plt.show()

    acc, auroc, ece, bal_acc = calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion)


    if save_results:
        f1_path = os.path.join(save_dir,'ID_task_accuracy'+str(filename)+'.txt')
        with open(f1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID task accuracy', 'AUROC', 'ECE'])
            for (acc_val, auroc_val, ece_val) in [(acc, auroc, ece)]:
                writer.writerow([acc_val, auroc_val, ece_val])

        true_confidences = [confidence for i, confidence in enumerate(confidence_list) if correct_list[i]]
        false_confidences = [confidence for i, confidence in enumerate(confidence_list) if not correct_list[i]]
        correct_bool_list = [True if i in correct_list else False for i in range(len(confidence_list))]

        f2_path = os.path.join(save_dir,'ID_task_confidence'+str(filename)+'.txt')
        with open(f2_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Confidence', 'Correct'])
            for (confidence_val, correct_val) in [(confidence_list, correct_bool_list)]:
                writer.writerow([confidence_val, correct_val])

    if return_outputs == True:
        return acc, bal_acc, labels_list


def get_metrics(path_id_confidence, path_ood_confidence, verbose=True, normalized=True):
    """ 
    Returns most common metrics (AUC, FPR, TPR) for comparing OOD vs ID inputs.
    Assumes that values are probabilities/confidences between 0 and 1 as default. 
    If not, please set normalized to False.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    verbose : bool, optional
        Whether to print the metrics. The default is True.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.

    Returns
    -------
    auroc : float
        The AUROC of the classifier.
    aucpr : float
        The AUCPR of the classifier.
    fpr : float
        The FPR of the classifier.
    tpr : float
        The TPR of the classifier.
    """
    id = np.loadtxt(path_id_confidence)
    ood = np.loadtxt(path_ood_confidence)
    if verbose:
        print('Mean confidence OOD: {}, Median: {}, Length: {}'.format(np.mean(ood), np.median(ood), len(ood)))
        print('Mean confidence ID: {}, Median: {}, Length: {}'.format(np.mean(id), np.median(id), len(id)))
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
    if not normalized:
        # use unity based normalization to also catch negative values
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    pred_probs_total = combine_arrays([pred_probs,true_labels])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    pred_probs = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    true_labels = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    if verbose:
        print('AUROC: {}'.format(auroc))
    return auroc, aucpr, fpr, tpr


def combine_arrays(input_arrays):
    """
    Combines elements of arrays into a single array

    Parameters
    ----------
    input_arrays : list
        A list of arrays to be combined.

    Returns
    -------
    output_array : list
        A list of arrays containing the combined elements of the input arrays.
    """
    if any(len(input_arrays[0])!= len(i) for i in input_arrays):
        raise Exception("Lists must all be the same length")

    output_array = []
    for i in range(0,len(input_arrays[0])):
        output_array.append([])
        for j in range(0,len(input_arrays)):
            output_array[i].append(input_arrays[j][i])
    
    return output_array
    

def get_AUROC_AUCPR(id,ood,return_fpr_tpr=False):
    """
    Calculates the AUROC, AUCPR, FPR and TPR of an ID and OOD dataset.

    Parameters
    ----------
    id : numpy array
        The confidence scores of the in-distribution data.
    ood : numpy array
        The confidence scores of the out-of-distribution data.

    Returns
    -------
    auroc : float
        The AUROC score.
    aucpr : float
        The AUCPR score.
    fpr : float
        The false positive rate.
    tpr : float
        The true positive rate.
    """
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
        # use unity based normalization to also catch negative values
    pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    
    if return_fpr_tpr:
        return auroc, aucpr, fpr, tpr
    return auroc, aucpr


def set_style(fontsize=12):
    """
    Sets the style of the plots.

    Parameters
    ----------
    fontsize : int, optional
        The fontsize of the plots. The default is 20.
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    #mpl.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams.update({'font.size': fontsize})


def ensure_class_overlap(OOD_dataset,classes_ID,classes_OOD):
    """
    Ensures that there is class overlap between the ID and OOD datasets.

    Parameters
    ----------
    OOD_dataset : dict
        The OOD dataset.
    classes_ID : list
        The list of classes in the ID dataset.
    classes_OOD : list
        The list of classes in the OOD dataset.

    Returns
    -------
    OOD_dataset : dict
        The OOD dataset with class overlap.
    """
    assert any(item in classes_OOD for item in classes_ID), 'There must be class overlap between ID and OOD'
    allowed_classes = [index for index, item_OOD in enumerate(classes_OOD) if item_OOD in classes_ID]
    OOD_dataset= OOD_dataset[OOD_dataset['class'].isin(allowed_classes)]
    class_mapping = {index_OOD: classes_ID.index(item_OOD) for index_OOD, item_OOD in enumerate(classes_OOD) if item_OOD in classes_ID}
    OOD_dataset['class'] = OOD_dataset['class'].map(class_mapping)
    
    print('Checking accuracy of DNN for classes which overlap with ID data:')
    for element in allowed_classes:
        print(classes_OOD[element], end=', ')
    print('\n')

    return OOD_dataset


def expand_classes(classes_ID,class_sel_dict):
    """
    Expands the classes to include all possible combinations of classes.

    Parameters
    ----------
    classes_ID : list
        The list of classes in the ID dataset.
    class_sel_dict : dict
        The dictionary of the class selection.

    Returns
    -------
    classes_ID : list
        The list of classes in the ID dataset.
    """
    if classes_ID == []:
        classes_ID = class_sel_dict['classes_ID']

    if ('atleast_one_positive_class' in class_sel_dict.keys() and class_sel_dict['atleast_one_positive_class'] == False
        ) or ('allow_multiple_positive_classes' in class_sel_dict.keys() and class_sel_dict['allow_multiple_positive_classes'] == True):
        result = ['Neg ' + ', '.join(classes_ID)] # Initialize the result list with 'no classes'
        # Generate all possible combinations of classes
        for r in range(1, len(classes_ID) + 1):
            for combo in combinations(classes_ID, r):
                # Create a string representation of the combination
                combo_str = ' '.join(['Pos ' + c if c in combo else 'Neg ' + c for c in classes_ID])
                result.append(combo_str)
        classes_ID = result

    return classes_ID


def normalise_image(img_tensor):
    # Normalize img tensor to [0, 1] range
    min_val = torch.min(img_tensor)
    max_val = torch.max(img_tensor)
    normalized_img = (img_tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_img


def loader_with_paths(idloader,oodloader):
    class ModifiedDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            image, target = self.base_dataset.__getitem__(index)
            image_path = self.base_dataset.image_paths[index]
            return image, target, image_path
    
    idloader_2 = ModifiedDataset(idloader.dataset)
    oodloader_2 = ModifiedDataset(oodloader.dataset)

    #return DataLoader(dataset, args.batch_size, shuffle=args.shuffle,pin_memory=True, num_workers=args.device_count*4, prefetch_factor = args.device_count, drop_last=drop_last_batch, persistent_workers=True, worker_init_fn=worker_init_fn)
    idloader_f = DataLoader(idloader_2, batch_size=idloader.batch_size, shuffle=False,pin_memory=idloader.pin_memory,num_workers=idloader.num_workers,
                            prefetch_factor=idloader.prefetch_factor,drop_last=idloader.drop_last,persistent_workers=idloader.persistent_workers,
                            worker_init_fn=idloader.worker_init_fn)
    
    oodloader_f = DataLoader(oodloader_2, batch_size=oodloader.batch_size, shuffle=False,pin_memory=oodloader.pin_memory,num_workers=oodloader.num_workers,
                            prefetch_factor=oodloader.prefetch_factor,drop_last=oodloader.drop_last,persistent_workers=oodloader.persistent_workers,
                            worker_init_fn=oodloader.worker_init_fn)
    
    return idloader_f, oodloader_f

def get_image_micro_results(idloader,oodloader,net,verbose=False,use_cuda=True):
    ood_logits_list = []
    ood_labels_list = []
    ood_correct_list = []
    ood_predicted_list = []
    id_logits_list = []
    id_labels_list = []
    id_correct_list = []
    id_predicted_list = []
    correct, total = 0, 0
    ood_total = 0
    id_total = 0

    id_names = []
    ood_names = []


    l = len(idloader)
    for batch_idx, (inputs, targets,names) in enumerate(idloader):
        id_names.extend(names)
        id_labels_list.extend([int(tensor.item()) for tensor in targets])
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        with torch.no_grad():
            _, _, _, id_total, _, id_logits_list, _,id_correct_list,id_predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,id_total,id_logits_list,id_logits_list,id_correct_list,id_predicted_list,required_grad=False,required_correct_list=True)
            
    l = len(oodloader)
    for batch_idx, (inputs, targets,names) in enumerate(oodloader):
        ood_names.extend(names)
        ood_labels_list.extend([int(tensor.item()) for tensor in targets])
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        with torch.no_grad():
            _, _, _, ood_total, _, ood_logits_list, _,ood_correct_list,ood_predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,ood_total,ood_logits_list,ood_logits_list,ood_correct_list,ood_predicted_list,required_grad=False,required_correct_list=True)

            
    return [id_names,id_logits_list,id_labels_list,id_correct_list,id_predicted_list],[ood_names,ood_logits_list,ood_labels_list,ood_correct_list,ood_predicted_list]