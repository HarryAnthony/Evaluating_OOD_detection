import torch
from source.util.general_utils import print_progress, variable_use_cuda


def get_layer_activations(model, layer, trainloader, use_cuda=True):
    """
    Get the activations of a given layer for a given dataset.

    Parameters
    ----------
    model: torch.nn.Module
        The model containing the layer to extract activations from.
    layer: torch.nn.Module
        The layer to extract activations from.
    trainloader: torch.utils.data.DataLoader
        The dataloader for the dataset.
    use_cuda: bool
        Whether to use cuda. Default: True

    Returns
    -------
    torch.Tensor
        The activations of the layer for the training dataset.
    """

    activations = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    hook_handle = layer.register_forward_hook(hook)

    # Pass data through the model
    model.eval()
    with torch.no_grad():
        for batch in trainloader:
            inputs = batch[0]
            inputs = variable_use_cuda(inputs, use_cuda)
            model(inputs)

    # Remove the hook
    hook_handle.remove()

    # Concatenate all collected activations
    activations = torch.cat(activations)
    return activations.squeeze(-1).squeeze(-1)


def calculate_contribution_matrix(weights, activations):
    """
    Calculate the contribution matrix V based on weights and activations.

    Parameters
    ----------
    weights: torch.Tensor
         The weight matrix of the final layer. Shape: [C, m]
    activations: torch.Tensor
        The activations from the preceding layer. Shape: [N, m], where N is the number of samples in the dataset.

    Returns
    -------
    torch.Tensor
        The contribution matrix V. Shape: [m, C]
    """
    weights = weights.to(activations.device)

    # Element-wise multiplication and averaging
    # Broadcasting weights to [N, C, m] and activations to [N, m, 1] for element-wise multiplication
    V = torch.einsum('cm,nm->nmc', weights, activations)
    V_mean = V.mean(dim=0).transpose(0, 1)
    return V_mean


def create_masking_matrix(V, p):
    """
    Create a masking matrix M based on the top-k largest elements in V.

    Parameters
    ----------
    V: torch.Tensor
        The contribution matrix of shape [m, C].
    p: float
        The sparsity parameter indicating the fraction of weights to be dropped.

    Returns
    -------
    torch.Tensor
        The masking matrix M of the same shape as V.
    """
    m, C = V.shape
    k = int((1 - p) * m * C)  # Calculate the number of elements to keep based on sparsity parameter p

    # Flatten V and find the k-th largest value
    V_flattened = V.flatten()
    top_k_values, _ = torch.topk(V_flattened, k)
    kth_largest_value = top_k_values[-1]  # The k-th largest value serves as a threshold

    # Create the mask: 1 for top-k elements, 0 otherwise
    M = (V >= kth_largest_value).float()

    return M


def get_weight_data(net, net_type='ResNet'):
    """
    Get the weight data and the layer of the model.

    Parameters
    ----------
    net: torch.nn.Module
        The model to extract the weight data from.
    net_type: str
        The type of the model. Default: 'ResNet'

    Returns
    -------
    torch.Tensor
        The weight data for layer of the model.
    torch.nn.Module
        The layer of the model.
    """
    if net_type == 'ResNet18':
        weights = net.fc.weight.data
        layer = net.avgpool
    elif net_type == 'VGG16':
        weights = net.classifier[6].weight.data
        layer = net.classifier[5]
    else:
        raise TypeError("Model type not supported. Please use a ResNet or VGG model.")
    
    return weights, layer


def apply_mask_hook(net, original_weight, M, net_type):
    """
    Apply a hook to the model to mask the weights.

    Parameters
    ----------
    net: torch.nn.Module
        The model to apply the hook to.
    original_weight: torch.Tensor
        The original weight data.
    M: torch.Tensor
        The masking matrix.
    net_type: str
        The type of the model.

    Returns
    -------
    torch.nn.Module
        The model with the hook applied.
    torch.utils.hook.RemovableHandle
        The handle for the hook.
    """
    def hook(module, input, output):
        module.weight.data = original_weight.cuda() * M.cuda()

    if net_type == 'ResNet18':
        handle = net.fc.register_forward_hook(hook)
    elif net_type == 'VGG16':
        handle = net.classifier[6].register_forward_hook(hook)

    return net, handle


def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, trainloader=None, temper=1, DICE_sparsification_param = 0.9, net_type='VGG16', **kwargs):
    """
    Evaluate DICE score on the ID and OOD datasets.

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
    DICE_sparsification_param: float
        The sparsity parameter for the DICE score. Default: 0.9

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

    weights, layer = get_weight_data(net, net_type=net_type)
    activations = get_layer_activations(net, layer, trainloader, use_cuda=use_cuda)
    V = calculate_contribution_matrix(weights, activations)
    M = create_masking_matrix(V, DICE_sparsification_param)
    net, handle = apply_mask_hook(net, weights, M, net_type)

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

    handle.remove() #Remove the hook

    return confidence


