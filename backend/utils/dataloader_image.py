from torch.utils.data import DataLoader
from preprocessing.image_dataset import DeepfakeImageDataset

def get_dataloader(splits_path,image_size=384,batch_size=4,num_worker=1):
    train_dataset=DeepfakeImageDataset(splits_path['train'],image_size=image_size)
    val_dataset=DeepfakeImageDataset(splits_path['val'],image_size=image_size)
    test_dataset=DeepfakeImageDataset(splits_path['test'],image_size=image_size)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_worker)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_worker)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_worker)

    return train_loader,val_loader,test_loader