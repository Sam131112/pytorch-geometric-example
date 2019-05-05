import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CoraFull
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Planetoid
from collections import Counter
from collections import defaultdict
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Model Name")

parser.add_argument("-model",action="store",dest="model",type=int,default=1)
parser.add_argument("-net",action="store",dest="net",type=int,default=1)
pr = parser.parse_args()


label_ids = defaultdict(list)

if   pr.net == 1:
     print("Data Cora")
     _data = Planetoid(root="./pcora",name="Cora")
elif pr.net == 2:
     print("Data CiteSeer")
     _data = Planetoid(root="./pciteseer",name="Citeseer")
elif pr.net == 3:
     print("Data Pubmed")
     _data = Planetoid(root="./ppubmed",name="Pubmed")
elif pr.net == 4:
     print("Data CoraFull")
     _data = CoraFull("./Corafull")
elif pr.net == 5:
     print("Data Coauthor CS")
     _data = Coauthor("./CS","CS")
elif pr.net == 6:
     print("Data Coauthor Physics")
     _data = Coauthor("./Physics","Physics")
elif pr.net == 7:
     print("Data Amazon Computer")
     _data = Amazon("./Computer","Computers")
elif pr.net == 8:
     print("Data Amazon Photos")
     _data = Amazon("./Photo","Photo")


#_data = Coauthor("./Physics","Physics")
#_data = Coauthor("./CS","CS")

#_data = CoraFull("./Corafull")

#_data = Planetoid(root="./pcora",name="Cora")
#_data = Planetoid(root="./pciteseer",name="Citeseer")
#_data = Planetoid(root="./ppubmed",name="Pubmed")

#_data = Amazon("./Computer","Computers")
#_data = Amazon("./Photo","Photo")



print("Number of Features,Number of Classes ",_data.num_features,_data.num_classes)

print("Class Distribution ",Counter(_data.data.y.numpy()))

labels = _data.data.y.numpy()

for i in range(len(labels)):
    label_ids[labels[i]].append(i)

train_mask_ids = []
test_mask_ids = []

for _x in label_ids:
    temp = np.random.choice(label_ids[_x],int(0.3*len(label_ids[_x])),replace=False)
    #print("Inside ",_x,Counter(_data.data.y[temp].numpy()),len(temp))
    #id1 = int(len(temp)/2.0)
    temp1 = temp[0:20]
    temp2 = temp[20:]
    train_mask_ids.extend(temp1)
    test_mask_ids.extend(temp2)
    #print("Inside ",_x,Counter(_data.data.y[temp1].numpy()),len(temp1))
    #print("Inside ",_x,Counter(_data.data.y[temp2].numpy()),len(temp2))
#print("Trains ",len(train_mask_ids))


print("Train Test Size ",Counter(_data.data.y[train_mask_ids].size()),Counter(_data.data.y[test_mask_ids].size()),len(train_mask_ids),len(test_mask_ids))


#print("Sanity ",Counter(Train_mask))
#print("Sanity ",Counter(_data.data.y[Train_mask].numpy()))
#print(Counter(_data.data.y))


torch.manual_seed(0)


#dataset = Planetoid(root="/tmp/Cora/",name="Cora")
#dataset = _data.data
#dataset = dataset.data


dataset = _data


print(dataset[0].x.shape)
print(dataset.slices)
print(dataset.num_classes)
print(dataset.data.num_nodes)
print(dataset.data.num_edges)


class LinearLayer(torch.nn.Module):
    def __init__(self,in_feature,out_feature,in_hidden):
        super(LinearLayer,self).__init__()
        self.fc1 = nn.Linear(in_feature,in_hidden)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_hidden,out_feature)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)



class GCNmyConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,num_nodes):
        super(GCNmyConv,self).__init__()
        #self.lin = torch.nn.Linear(in_channels,out_channels)
        #self.num_nodes = num_nodes
    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        edge_index = add_self_loops(edge_index,num_nodes = x.size(0))
        x = self.lin(x)
        return self.propagate(aggr="mean",edge_index=edge_index,x=x)

    def message(self,x_j,edge_index):
        row,col = edge_index
        deg = degree(row,dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return norm.view(-1,1)*x_j

    def update(self,aggr_out):
        return aggr_out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1  = GCNConv(dataset.num_features,16)
        #self.conv2 = GCNConv(512,256)
        #self.conv3 = GCNConv(256,128)
        self.conv2 = GCNConv(16,dataset.num_classes)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x,edge_index)
        #x = F.relu(x)
        #x = F.dropout(x,training=self.training)
        #x = self.conv3(x,edge_index)
        #x = F.relu(x)
        #x = F.dropout(x,training=self.training)
        #x = self.conv4(x,edge_index)
        return F.log_softmax(x,dim=1)


#torch.cuda.manual_seed_all(1000)
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
#data  = dataset.to(device)
if pr.model:
    model = Net().to(device)
else:
    model = LinearLayer(dataset.num_features,dataset.num_classes,16).to(device)
data = dataset.data
data.__setitem__("Train_Mask",torch.Tensor.long(torch.Tensor(train_mask_ids)))
data.__setitem__("Test_Mask",torch.Tensor.long(torch.Tensor(test_mask_ids)))



data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.05,weight_decay=5e-4)
model.train()
print("Starting Epochs ")
print(data.x.shape)
print(data.edge_index.shape)

model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    #print(out.shape)
    #print(out[data.train_mask],data.y[data.train_mask])
    loss = F.nll_loss(out[data.Train_Mask],data.y[data.Train_Mask])
    #print(loss.item())
    loss.backward()
    optimizer.step()


print("Labels ",Counter(data.y[data.Test_Mask].cpu().numpy()))

model.eval()
_,pred = model(data).max(dim=1)
_correct1 = pred[data.Test_Mask].eq(data.y[data.Test_Mask]).sum().item()
#_correct2 = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
acc1 = _correct1 / data.Test_Mask.size()[0]
#acc2 = _correct2 / data.val_mask.sum().item()
print(_correct1,acc1)
print(data.Test_Mask.size())

