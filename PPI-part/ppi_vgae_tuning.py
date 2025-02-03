import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import VGAE
from torch_geometric.utils import negative_sampling
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


# Specify the directory path for ppi networks of different sp
directory_path = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/string_networks/'

# Get a list of all file names in the directory
file_list = os.listdir(directory_path)

# Filter out directories (if you only want files)
file_list = [f for f in file_list if os.path.isfile(os.path.join(directory_path, f))]

random.seed(42)
random.shuffle(file_list)

# Calculate the split indices
total_files = len(file_list)
train_split = int(0.8 * total_files)
val_split = train_split + int(0.1 * total_files)

# Split the filenames
train_files = file_list[:train_split]
val_files = file_list[train_split:val_split]
test_files = file_list[val_split:]

# Output the results
print("Train Files:", train_files)
print("Validation Files:", val_files)
print("Test Files:", test_files)

# node2vec embeddings for all plant proteins 
with open('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/node2vec_dict.pkl', 'rb') as f:
    node2vec_embeddings = pickle.load(f)
f.close()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def genDataObjFromTSV(tsv, node2vec_dict):
    df = pd.read_csv(directory_path+tsv, sep='\t')

    #passing index for each node
    col1 = df['protein1'].to_list()
    col2 = df['protein2'].to_list()
    full_list = col1 + col2
    unique_list = list(dict.fromkeys(full_list))    # unique proteins in ppi file

    ind2node = {index: item for index, item in enumerate(unique_list)}  # index to protien dict 
    node2ind = {v: k for k, v in ind2node.items()}  # protein to index dict 

    df['protein1'] = df['protein1'].map(node2ind)
    df['protein2'] = df['protein2'].map(node2ind)

    # prepare interaction to be used as COO format
    first_prot = df['protein1'].to_list()   
    second_prot = df['protein2'].to_list()
    edge_index = np.array([first_prot, second_prot])
    edge_index = torch.from_numpy(edge_index)
    node_features = np.array([node2vec_dict[prot] for prot in unique_list])  
    node_features = torch.tensor(node_features) 
    data = Data(x= node_features, edge_index=edge_index)
    return data


#####################################################################
class PPIEncoder1(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder1, self).__init__()
        self.conv_mu = SAGEConv(in_size, out_size)
        self.conv_logstd = SAGEConv(in_size, out_size)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        

class PPIEncoder2(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder2, self).__init__()
        self.conv1 = SAGEConv(in_size, mid_size)
        self.conv_mu = SAGEConv(mid_size, out_size)
        self.conv_logstd = SAGEConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    

class PPIEncoder3(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder3, self).__init__()
        self.conv1 = SAGEConv(in_size, mid_size)
        self.conv2 = SAGEConv(mid_size, mid_size)
        self.conv_mu = SAGEConv(mid_size, out_size)
        self.conv_logstd = SAGEConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# parameters
out_channels = 200 
num_features = 256
mid_channels = 220
epochs = len(train_files)

# Manually print out the model summary
def print_model_summary(model):
    print("Model Summary:")
    print(model)
    print("\nModel Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
        total_params += param.numel()
    print(f"\nTotal Parameters: {total_params}")

##########################################
def train(train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.edge_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss) 

def test(test_val_data):
    model.eval()
    neg_edge_index = negative_sampling(test_val_data.edge_index)
    with torch.no_grad():
        z = model.encode(test_val_data.x, test_val_data.edge_index)
    loss = model.recon_loss(z, test_val_data.edge_index)
    loss = loss + (1 / test_val_data.num_nodes) * model.kl_loss()
    auc, ap = model.test(z, test_val_data.edge_index, neg_edge_index)
    return auc, ap, float(loss) 
#############################################################

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/ppi_vgae.pt'):
        """
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            delta (float): Minimum change to qualify as improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_auc = 0 
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_auc, model):
        if val_auc > self.best_auc + self.delta:  # AUC should increase
            self.best_auc = val_auc
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), self.path)  # Save best model
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

#################################################################

main_auc_tuning_df = pd.DataFrame()
main_ap_tuning_df = pd.DataFrame()


for lr in [0.1, 0.01, 0.001]:
    for i,encoder in enumerate([PPIEncoder1, PPIEncoder2, PPIEncoder3]):
        early_stopping = EarlyStopping(patience=3, delta=0.001)
        model = VGAE(encoder(num_features, mid_channels,out_channels))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print_model_summary(model)
        print(lr, i)
        best_val_auc = 0
        
        for epoch in range(0,epochs):
            train_data = genDataObjFromTSV(train_files[epoch], node2vec_embeddings)
            trloss = train(train_data.to(device))

            test_aucs, test_aps = [], []
            for test_data in test_files:
                test_data = genDataObjFromTSV(test_data, node2vec_embeddings)
                test_data = test_data.to(device)
                test_auc, test_ap, _ = test(test_data)
                test_aucs.append(test_auc)
                test_aps.append(test_ap)
            mean_test_auc = np.mean(test_aucs)
            mean_test_ap = np.mean(test_aps)

            val_aucs, val_aps = [], []
            for val_data in val_files:
                val_data = genDataObjFromTSV(val_data, node2vec_embeddings)
                val_data = val_data.to(device)
                val_auc, val_ap, _ = test(val_data)
                val_aps.append(val_ap)
                val_aucs.append(val_auc)
            mean_val_auc = np.mean(val_aucs)
            mean_val_ap = np.mean(val_aps)
                
            print(f'epoch {epoch+1} train loss : {trloss:.4f} | mean test auc : {mean_test_auc:.4f} | mean val auc : {mean_val_auc:.4f} | mean test ap : {mean_test_ap:.4f} | mean val ap : {mean_val_ap:.4f}')
            
            early_stopping(mean_val_auc, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        #####
        model.load_state_dict(torch.load('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/ppi_vgae.pt', weights_only=True))

        test_aucs, test_aps = [], []
        for test_data in test_files:
            test_data = genDataObjFromTSV(test_data, node2vec_embeddings)
            # test_data = torch.load(test_data)
            test_data = test_data.to(device)
            test_auc, test_ap, _ = test(test_data)
            test_aucs.append(test_auc)
            test_aps.append(test_ap)

        val_aucs, val_aps = [], []
        for val_data in test_files:
            val_data = genDataObjFromTSV(val_data, node2vec_embeddings)
            # val_data = torch.load(val_data)
            val_data = val_data.to(device)
            val_auc, val_ap, _ = test(val_data)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)

        #################################################
        auc_df = pd.DataFrame({
            "AUROC": test_aucs + val_aucs,
            "Distribution": ["Test"] * len(test_aucs) + ["Validation"] * len(val_aucs),
            "Parameters": [f'GSAGE layers-{i+1}, lr-{lr}']*len(test_aucs + val_aucs)
        })

        main_auc_tuning_df = pd.concat([main_auc_tuning_df, auc_df], ignore_index= True)

        ap_df = pd.DataFrame({
            "AP": test_aps + val_aps,
            "Distribution": ["Test"] * len(test_aps) + ["Validation"] * len(val_aps),
            "Parameters": [f'GSAGE layers-{i+1}, lr-{lr}']*len(test_aps + val_aps)
        })

        main_ap_tuning_df = pd.concat([main_ap_tuning_df, ap_df], ignore_index=True)

        

plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df, kind='violin', hue='Parameters')
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/violinplot_roc.png', bbox_inches='tight')


plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df, kind='box', hue='Parameters')
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/boxplot_roc.png', bbox_inches='tight')


plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df, kind='violin', hue='Parameters')
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/violinplot_ap.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df, kind='box', hue='Parameters')
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/boxplot_ap.png', bbox_inches='tight')