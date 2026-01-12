""" Custom dataset for the projects
"""

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class NIHChestXrayDataset(Dataset):
    """Custom Dataset for NIH Chest X-ray images
    """
    def __init__(self, df, transform=None):
        """Initialize the dataset with a dataframe and optional transformations.
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (albumentations.Compose, optional): Transformations to apply to images.
        """
        self.img_paths = df["image_path"].tolist()
        
        self.labels = df['multi_labels'].tolist()
        
        self.transform = transform if transform else A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        """Return the total number of samples in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding label by index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the corresponding label.
        """
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise RuntimeWarning(f"Image at index {idx} could not be read from path: {self.img_paths[idx]}")
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            raise RuntimeError(f"Unsupported image format at path {self.img_paths[idx]} with shape: {img.shape}")
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label