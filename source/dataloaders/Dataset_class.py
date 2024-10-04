from PIL import Image
from torch.utils.data import  Dataset

class Dataset_class(Dataset):
    def __init__(self, folder_dir, dataframe, transform):
        """
        Class a dataset with images (# adapted from https://github.com/kamenbliznashki/chexpert/blob/2bf52b1b70c3212a4c2fd4feacad0fd198fe8952/dataset.py#L17)
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        transform: torchvision.transforms
            transform to apply to images
        """
        self.image_paths = [] # List of image paths
        self.targets = [] # List of image labels
        self.patient_ids = []
        
        self.transform = transform

        for row in dataframe.to_dict('records'):
            self.patient_ids.append(row['Path']) 
            image_path = str(folder_dir) + str(row['Path'])
            self.image_paths.append(image_path)
            self.targets.append(row['class'])

        assert len(self.targets) == len(self.image_paths), 'Label count does not match the image count.'

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        image_path = self.image_paths[index]
        image_data = Image.open(image_path) 

        if self.transform is not None:
            image = self.transform(image_data)

        return image, self.targets[index] 