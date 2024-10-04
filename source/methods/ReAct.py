import torch
import numpy as np 
from source.util.general_utils import print_progress, variable_use_cuda
from source.methods.DICE import get_layer_activations, get_weight_data


def apply_truncation_hook(net, threshold=np.inf, net_type='ResNet18'):
    """
    Apply a hook to the model to truncate the output.

    Parameters
    ----------
    net: torch.nn.Module
        The model to apply the hook to.
    threshold: float
        The threshold to truncate the output. Default: np.inf
    net_type: str
        The type of the model.

    Returns
    -------
    torch.nn.Module
        The model with the hook applied.
    torch.utils.hook.RemovableHandle
        The handle for the hook.
    """
    def truncation_hook(module, input, output):
        return torch.clamp(output, max=threshold)

    if net_type == 'ResNet18':
        handle = net.avgpool.register_forward_hook(truncation_hook)
    elif net_type == 'VGG16':
        handle = net.classifier[5].register_forward_hook(truncation_hook)

    return net, handle


def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, trainloader=None, temper=1, net_type='ResNet18', ReAct_percentile=0.9, **kwargs):
    """
    Evaluate energy score on the ID and OOD datasets.

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
    trainloader: torch.utils.data.DataLoader
        The dataloader for the training dataset. Default: None
    temper: float
        The temperature parameter for the softmax function. Default: 1
    net_type: str
        The type of the model. Default: 'ResNet18'
    ReAct_percentile: float
        The percentile to use for the threshold. Default: 0.9

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

    _, layer = get_weight_data(net, net_type=net_type)
    activations = get_layer_activations(net, layer, trainloader, use_cuda=use_cuda)
    # Calculate the percentile from the training data activations
    threshold = torch.quantile(activations.flatten(), ReAct_percentile).item()
    net, hook_handle = apply_truncation_hook(net, threshold=threshold, net_type=net_type)

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            with torch.no_grad():
                inputs = variable_use_cuda(inputs, use_cuda)
                out = net(inputs)

            energy = temper*torch.logsumexp(out / temper, dim=1)
            energy = [t.item() for t in energy]
            confidence[OOD].extend(energy)

    hook_handle.remove()

    return confidence
