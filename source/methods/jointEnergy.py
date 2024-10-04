import torch
import numpy as np 
from source.util.general_utils import print_progress, variable_use_cuda


def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, **kwargs):
    """
    Evaluate JointEnergy on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
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
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            with torch.no_grad():
               inputs = variable_use_cuda(inputs, use_cuda)
               logits = net(inputs.cuda())  # Forward pass to get logits for each class
            
            # Calculate the label-wise free energy
            label_wise_free_energy = -torch.log(1 + torch.exp(-logits))

            # Calculate the JointEnergy score for each image in the batch
            joint_energy = -torch.sum(label_wise_free_energy, dim=1)

            confidence[OOD].extend(joint_energy.cpu().numpy().tolist())

    return confidence    