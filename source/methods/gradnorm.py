
import torch.nn.functional as F
import torch
from source.util.general_utils import print_progress


def calculate_kl_divergence(logits):
    """
    Calculate the KL divergence between the model's output and a uniform distribution.

    Parameters
    ----------
    logits: torch.Tensor
        The model's output logits

    Returns
    -------
    torch.Tensor
        The KL divergence between the model's output and a uniform distribution.
    """
    softmax_probs = F.softmax(logits, dim=1)
    uniform_probs = torch.ones_like(softmax_probs) / softmax_probs.size(1)
    kl_div = F.kl_div(softmax_probs.log(), uniform_probs, reduction='batchmean')
    softmax_probs.retain_grad()
    return kl_div


def gradnorm(model,data,grad_layer=-1,gradnorm_summation_method='l1'):
    """
    Implement GradNorm for OOD detection.

    Parameters
    ----------
    model: torch.nn.Module
        The model to evaluate
    data: torch.Tensor
        A batch of images
    grad_layer: int
        The layer to calculate the gradient norm from. Default: -1
    summation_method: str
        The method to sum the gradients. Either 'l1' or 'l2'. Default: 'l1'

    Returns
    -------
    float
        The L1-norm of the gradients from the model's output layer.
    """
    grad_norms = []

    for i in range(data.size(0)):  # Iterate over each image in the batch
        # Select the ith image
        image = data[i].unsqueeze(0)  # Add batch dimension
        image.requires_grad = True

        # Forward pass
        logits = model(image)

        # Calculate KL divergence loss
        loss = calculate_kl_divergence(logits)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Compute gradient norm for the current image
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        # Calculate the gradient norm
        output_layer_gradients = gradients[grad_layer]
        if gradnorm_summation_method == 'l1':
            grad_norm = torch.sum(torch.abs(output_layer_gradients)).item()
        elif gradnorm_summation_method == 'l2':
            grad_norm = torch.norm(output_layer_gradients, p=2).item()
        else:
            raise ValueError('summation_method must be either "l1" or "l2"')
        grad_norms.append(grad_norm)

    return grad_norms


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,**kwargs):
    """
    Evaluate GradNorm on the ID and OOD datasets.

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

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
            softmax_score = gradnorm(net, inputs.cuda() if use_cuda else inputs)
            confidence[OOD].extend(softmax_score)

    return confidence








