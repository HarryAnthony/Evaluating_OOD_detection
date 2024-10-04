import torch
import numpy as np 
from source.util.general_utils import print_progress, variable_use_cuda
from scipy.stats import entropy


def compute_mean_logits(trainloader, model, num_classes, use_cuda=True, **kwargs):
    """
    Compute the mean logit distribution for each class in the training data.

    Parameters
    ----------
    trainloader: torch.utils.data.DataLoader
        The DataLoader for the training data
    model: torch.nn.Module
        The neural network model
    num_classes: int
        The number of classes

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_classes, num_logits) representing the mean logits for each class
    """
    # Initialize a list to hold sum of logits for each class and a count for each class
    logits_sum = torch.zeros(num_classes, 0)
    class_counts = np.zeros(num_classes)

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in trainloader:
            # Forward pass to get logits
            inputs = variable_use_cuda(inputs, use_cuda)
            logits = model(inputs)
            
            # Convert logits to numpy for processing
            logits_np = logits.cpu().numpy()

            # Update sums and counts for each class
            for i in range(num_classes):
                class_mask = (labels == i)
                if class_mask.any():
                    if logits_sum.shape[1] == 0:
                        logits_sum = np.zeros((num_classes, logits_np.shape[1]))
                    logits_sum[i] += logits_np[class_mask].sum(axis=0)
                    class_counts[i] += class_mask.sum()

    # Calculate mean logits for each class
    mean_logits = torch.tensor(logits_sum / class_counts[:, None], dtype=torch.float32)

    return mean_logits


def ood_scoring_function(testloader, model, mean_logits, use_cuda=True, verbose=True, **kwargs):
    """
    Use the smallest KL-divergence to a mean class distribution as the OOD scoring function.

    Parameters
    ----------
    testloader: torch.utils.data.DataLoader
        The DataLoader for the test data
    model: torch.nn.Module
        The neural network model
    mean_logits: torch.Tensor
        The mean logits for each class

    Returns
    -------
    list
        A list of OOD scores for each test sample
    """
    ood_scores = []

    # Ensure the model is in evaluation mode
    model.eval()

    l = len(testloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

    with torch.no_grad():
        for batch_idx, (inputs, _) in testloader:
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            # Forward pass to get logits
            inputs = variable_use_cuda(inputs, use_cuda)
            logits = model(inputs)

            # Convert logits to softmax probabilities
            softmax_probs = torch.softmax(logits, dim=1).cpu().numpy()

            # Compute KL-divergence with each class mean and take the minimum
            for prob in softmax_probs:
                kl_divs = [entropy(prob, torch.softmax(mean_logit, dim=0).numpy()) for mean_logit in mean_logits]
                min_kl_div = -1*min(kl_divs)
                ood_scores.append(min_kl_div)

    return ood_scores


def evaluate(net, idloader, oodloader, use_cuda=True, trainloader=None, num_classes=2,verbose=True,**kwargs):
    """
    Evaluate Logit KL-Matching on the ID and OOD datasets.

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
    trainloader: torch.utils.data.DataLoader
        The dataloader for the training dataset
    num_classes: int
        The number of classes. Default
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
    
    mean_logits = compute_mean_logits(trainloader, net, num_classes, use_cuda=use_cuda, **kwargs)

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
            
        confidence[OOD] = ood_scoring_function(loader, net, mean_logits, use_cuda=use_cuda, **kwargs)

    return confidence   