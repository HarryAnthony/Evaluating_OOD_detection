import torch
import torch.nn.functional as F
import torch
import numpy as np 
from source.util.general_utils import print_progress, variable_use_cuda
import matplotlib.pyplot as plt


def softmax_entropy(logits, epsilon=1e-8):
    """
    Calculate the Shannon entropy of the softmax distribution of logits.

    Parameters
    ----------
    logits: torch.Tensor
        The logits output from a neural network.

    Returns
    -------
    float
        The Shannon entropy of the softmax distribution.
    """
    if isinstance(epsilon, float) != True:
        raise ValueError('epsilon must be a float')
    # Compute the softmax distribution
    probs = F.softmax(logits, dim=1)
    entropy = torch.sum(probs * torch.log(probs+epsilon), dim=1)
    return entropy


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,**kwargs):
    """
    Evaluate MCP on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    use_cuda: bool
        Whether to use cuda. Default: True
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()
    net.training = False
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50, verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50, verbose=verbose)
            
            with torch.no_grad():
                inputs = variable_use_cuda(inputs,use_cuda)
                out = net(inputs)
                entropy_list = softmax_entropy(out)

            confidence[OOD].extend(entropy_list.cpu().numpy().tolist())

    return confidence   