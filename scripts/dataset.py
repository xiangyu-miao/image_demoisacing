import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MosaicDataset(Dataset):
    """ `Dataset` stores the samples and their corresponding labels.
    """
    def __init__(self, mosaic_dir, original_dir, transform=None):
        ''' The __init__ function is run once when instantiating the Dataset object.
        '''
        self.mosaic_dir = os.path.expanduser(mosaic_dir)
        self.original_dir = os.path.expanduser(original_dir)
        self.transform = transform
        self.image_names = sorted(os.listdir(self.mosaic_dir))  # [mosaic_000001.jpg, mosaic_000002.jpg, ...]

    def __len__(self):
        ''' The __len__ function returns the number of samples in our dataset.
        '''
        return len(self.image_names)

    def __getitem__(self, idx):
        ''' The __getitem__ function loads and returns a sample from the dataset at the given index idx.
        '''
        mosaic_path = os.path.join(self.mosaic_dir, self.image_names[idx])
        original_path = os.path.join(self.original_dir, self.image_names[idx].replace("mosaic_", "original_"))
        
        # 打开图像
        mosaic_image = Image.open(mosaic_path).convert("RGB")
        original_image = Image.open(original_path).convert("RGB")
        
        # 数据变换
        if self.transform:
            mosaic_image = self.transform(mosaic_image)
            original_image = self.transform(original_image)
        else:
            mosaic_image = transforms.ToTensor()(mosaic_image)
            original_image = transforms.ToTensor()(original_image)
        
        return mosaic_image, original_image