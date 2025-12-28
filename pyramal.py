import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Subset, Dataset, DataLoader
from torchmetrics.classification import Accuracy, MulticlassRecall, MulticlassF1Score, MulticlassPrecision
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import random
import numpy as np

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

transform = torchvision.transforms.ToTensor()
def transform_image(image):
    try:
        return transform(Image.open(image))
    except:
        return transform(Image.open(image.replace('.png', '.exe.png')))

class DatasetMalimg(Dataset):
    def __init__(self, images_dir, labels_path):
        self.images_dir = images_dir
        self.df = pd.read_csv(labels_path).reset_index()
        families = ['Adialer.C', 'Allaple.L', 'C2LOP.gen!g', 'Dontovo.A', 'Lolyda.AA1', 'Lolyda.AT', 'Rbot!gen', 'Swizzor.gen!I', 'Yuner.A', 'Agent.FYI', 'Alueron.gen!J', 'C2LOP.P', 'Fakerean', 'Lolyda.AA2', 'Malex.gen!J' ,'Skintrim.N', 'VB.AT', 'Allaple.A', 'Autorun.K', 'Dialplatform.B', 'Instantaccess', 'Lolyda.AA3', 'Obfuscator.AD', 'Swizzor.gen!E', 'Wintrim.BX']
        self.df['family_index'] = self.df['family'].apply(lambda x : families.index(x))

    def __getitem__(self, index):
        image, family = self.df['file'].loc[index], self.df['family'].loc[index]
        tensor = transform_image(os.path.join(self.images_dir, family, image + '.png'))
        return tensor, self.df['family_index'].loc[index]

    def __len__(self):
        return self.df.shape[0]

class MalwareDC(torch.nn.Module):
    def __init__(self, num_classes, lr):
        super().__init__()

        resnet50 = models.resnet50()
        resnet50.load_state_dict(torch.load('path/to/resnet50-19c8e357.pth'))
        resnet50.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
        self.model = resnet50
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
    def forward(self, X, y):
        y_hat = self.model(X)
        l = self.loss(y_hat, y)
        return l, y_hat
    
def train_model_kfold(net, epochs, dataset, kfold, train_batch_size=64, train_dataloader_workers=4, eval_batch_size=64, eval_dataloader_workers=4, metrics_setting={'num_classes':25, 'average':'macro'} ):

    # By default, a single-card GPU is used.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_datasets, test_datasets = [], []
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(dataset.df['file'], dataset.df['family']):
        train_datasets.append(Subset(dataset, indices=train_index))
        test_datasets.append(Subset(dataset, indices=test_index))

    def data_loader(kf_index):
        train_dataset, test_dataset = train_datasets[kf_index], test_datasets[kf_index]
        return DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_dataloader_workers), DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=eval_dataloader_workers)
    

    accuracy_test = Accuracy(task="multiclass", num_classes=metrics_setting['num_classes']).to(device)
    recall_test = MulticlassRecall(num_classes=metrics_setting['num_classes'], average=metrics_setting['average']).to(device)
    precision_test = MulticlassPrecision(num_classes=metrics_setting['num_classes'], average=metrics_setting['average']).to(device)
    f1_score_test = MulticlassF1Score(num_classes=metrics_setting['num_classes'], average=metrics_setting['average']).to(device)

    for kf_index in range(kfold):       
        print(f'The {kf_index}-th fold training...')

        net_current = net[kf_index].to(device)  
        train_iter, test_iter = data_loader(kf_index)  
        # train model
        print('*' * 80)
        for epoch in range(epochs):
            net_current.train()
            for X, y in tqdm(train_iter, 
                             desc=f"train@epoch: {epoch}", leave=False, colour='green', ncols=128):
                X, y = X.to(device), y.to(device)
                net_current.optimizer.zero_grad()
                loss, y_hat = net_current(X, y)
                loss.backward()
                net_current.optimizer.step()

            if hasattr(net_current, 'scheduler'):
                net_current.scheduler.step()

            # Evaluation
            net_current.eval()
            accuracy_test.reset()
            recall_test.reset()
            precision_test.reset()
            f1_score_test.reset()

            with torch.no_grad():
                for X, y in tqdm(test_iter, desc="eval", leave=False, colour='green', ncols=128):
                    X, y = X.to(device), y.to(device)
                    loss, y_hat = net_current(X, y)
                    accuracy_test.update(y_hat, y)
                    recall_test.update(y_hat, y)
                    precision_test.update(y_hat, y)
                    f1_score_test.update(y_hat, y)
            print(f'epoch {epoch}:  Accuracy:{accuracy_test.compute() : f},  Recall:{recall_test.compute() : f},  Precision:{precision_test.compute() : f},  F1-score:{f1_score_test.compute() : f}\n')
        

if __name__ == '__main__':
    # Take the training and testing of PyraMal on the Malimg dataset as an example.
    num_classes, kfold, epochs, lr, batch_size = 25, 10, 20, 0.01, 64
    dataset = DatasetMalimg('path/to/dataset/dir', 'path/to/dataset/labels')
    net = []
    for index in range(kfold):
        net.append(MalwareDC(num_classes, lr))

    train_model_kfold(net, epochs, dataset, kfold, batch_size, metrics_setting={'num_classes':num_classes, 'average':'macro'})
    print('over.')