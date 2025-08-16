from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class FERDataset(Dataset):
    """
    Custom dataset for facial expression recognition (FER) tasks.
    This dataset assumes that images are stored in a directory structure
    where each subdirectory corresponds to a class label.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.label_map = {
            'sad': 0, 
            'disgust': 1, 
            'angry': 2, 
            'neutral': 3, 
            'fear': 4, 
            'surprise': 5, 
            'happy': 6
        }

        self.inverted_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load images and labels from the directory structure
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def _load_raw(self, idx):
        """Load the PIL image without applying any transforms."""
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing {img_path}")
        img = Image.open(img_path).convert("RGB")
        return img
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the image and its label.
        """
        label = self.labels[idx]
        img = self._load_raw(idx)

        if self.transform:
            img = self.transform(img)

        return {"pixel_values": img, "label": self.label_map[label]}
    


class BinaryFERDataset(FERDataset):
    """
    Custom dataset for storing only two classes of facial expressions. 
    """
    
    def __init__(self, root_dir, class1, class2, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in subdirectories.
            class1 (str): First class label to include.
            class2 (str): Second class label to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root_dir, transform)
        self.class1 = class1
        self.class2 = class2
        self.image_paths = []
        self.labels = []

        # Filter images and labels for the specified classes
        for idx, label in enumerate(self.labels):
            if label == class1 or label == class2:
                self.image_paths.append(self.image_paths[idx])
                self.labels.append(label)