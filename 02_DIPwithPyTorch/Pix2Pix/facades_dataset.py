import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file1, list_file2):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file1, 'r') as file:
            self.image_filenames_1 = [line.strip() for line in file]
        with open(list_file2, 'r') as file:
            self.image_filenames_2 = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames_1)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames_1[idx]
        img_color_semantic = cv2.imread(img_name)
        label_name = self.image_filenames_2[idx]
        label_semantic = cv2.imread(label_name)
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        label = torch.from_numpy(label_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        # image_rgb = image[:, :, :256]
        # image_semantic = image[:, :, 256:]
        return image, label