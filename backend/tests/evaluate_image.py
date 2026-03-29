import torch
from utils.dataloader_image import get_dataloader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from models.proposed_model import Proposed_model


def evaluate(model,dataloader,device):
    model.eval()
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for images,labels in tqdm(dataloader,desc='Evaluating'):
            images=images.to(device)
            labels=labels.to(device)
            output=model(images)
            preds=torch.argmax(output,dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds,all_labels
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split={'train':'data/video/splits/train.txt','test':"data/video/splits/test.txt",'val':"data/video/splits/val.txt"}
    train_loader, val_loader,test_loader=get_dataloader(splits_path=split,batch_size=128,num_worker=6)
    checkpoint=torch.load('weights/efficientnet_b2_best.pth', map_location=device)
    model = Proposed_model(num_classes=2)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds, labels = evaluate(model, test_loader, device)

    print("Classification Report:\n", classification_report(labels, preds, target_names=["real", "fake"]))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))

    