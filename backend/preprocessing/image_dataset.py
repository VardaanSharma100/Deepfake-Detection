from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from logs.image_logging import image_logger
logger=image_logger().get_logger()
class DeepfakeImageDataset(Dataset):
    def __init__(self,split_file,image_size=384,augment=False):
        self.root_dir='data/image/processed'
        self.image_size=image_size
        self.augment=augment
        self.image_paths=[]
        self.labels=[]
        try:
            with open(split_file,'r') as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        print("line not found")
                    path,label=line.split()
                    self.image_paths.append(os.path.join(self.root_dir,path))
                    self.labels.append(int(label))

        except Exception as e:
            logger.warning(str(e))
        self.transform=transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        img_path=self.image_paths[index]
        label=self.labels[index]
        try:
            img=Image.open(img_path).convert("RGB")
            img=self.transform(img)
            return img,label
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {str(e)}") from e
