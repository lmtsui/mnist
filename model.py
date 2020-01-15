import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(11*11*64,300)
        self.fc2 = nn.Linear(300,10)
        self.dropout = nn.Dropout(p=0.3)
        self.cross_ent = nn.CrossEntropyLoss()
    def forward(self,x,labels):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.dropout(x)
        logit=self.fc2(x)
        # log_P = F.log_softmax(logit,dim=-1)
        # log_P_gathered = torch.gather(log_P,dim=1,index=labels.unsqueeze(-1)).squeeze(-1)
        # batch_loss = -log_P_gathered.sum()
        loss = self.cross_ent(logit,labels)
        return loss, logit