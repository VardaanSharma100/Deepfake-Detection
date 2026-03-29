import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
from tqdm import tqdm
from models.proposed_model import Proposed_model
from utils.dataloader_image import get_dataloader
import os

def train():
    start_epoch=0
    checkpoint_path='weights/proposed_model.pth'
    best_val_accuracy=0.0
    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        checkpoint=torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']+1
        best_val_accuracy=checkpoint['best_val_accuracy']
        print(f"Resuming from Epoch {start_epoch} with best accuracy{best_val_accuracy:.2f}%")
    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss=0.0
        correct,total=0,0
        loop=tqdm(train_loader,desc=f'Epoch {epoch+1}/{num_epochs}')

        for images,labels in loop:
            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _,preds=torch.max(outputs,1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

            loop.set_postfix(loss=loss.item(),acc=100.0*correct/total)
        model.eval()
        val_loss=0.0
        val_correct,val_total=0,0
        with torch.no_grad():
            for images,labels in tqdm(val_loader):
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels)
                val_loss+=loss.item()
                _,preds=torch.max(outputs,1)
                val_correct+=(preds==labels).sum().item()
                val_total+=labels.size(0)
        val_accuracy=100*val_correct/val_total
        print("Val Accuracy is --->",val_accuracy,"Best val_accuracy",best_val_accuracy)
        scheduler.step(val_accuracy)
        if val_accuracy>best_val_accuracy:
            best_val_accuracy=val_accuracy
            os.makedirs('weights',exist_ok=True)
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_val_accuracy':best_val_accuracy,
            },checkpoint_path)
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    batch_size=16
    # batch_size=16 or 8 or 4 or 2
    num_workers=2
    # num_workers=0
    learning_rate=1e-4
    num_epochs=20
    split={'train':'data/image/splits/train.txt','test':"data/image/splits/test.txt",'val':"data/image/splits/val.txt"}

    criterion=nn.CrossEntropyLoss()

    model=Proposed_model(num_classes=2,freeze_features=False).to(device)

    optimizer=Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3)

    train_loader, val_loader, test_loader=get_dataloader(splits_path=split,batch_size=batch_size,num_worker=num_workers)

    train()



    