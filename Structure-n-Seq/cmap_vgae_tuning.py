#libraries
import torch 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
from torch_geometric.utils import negative_sampling
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Specify the directory path for ppi networks of different sp
directory_path = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/maps/'

# Get a list of all file names in the directory
file_list = os.listdir(directory_path)

# Filter out directories (if you only want files)
file_list = [directory_path + f for f in file_list if os.path.isfile(os.path.join(directory_path, f))]

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
# print("Train Files:", train_files)
# print("Validation Files:", val_files)
# print("Test Files:", test_files)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


class CMAPEncoder1(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(CMAPEncoder1, self).__init__()
        self.conv_mu = GCNConv(in_size, out_size)
        self.conv_logstd = GCNConv(in_size, out_size)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        

class CMAPEncoder2(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(CMAPEncoder2, self).__init__()
        self.conv1 = GCNConv(in_size, mid_size)
        self.conv_mu = GCNConv(mid_size, out_size)
        self.conv_logstd = GCNConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    

class CMAPEncoder3(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(CMAPEncoder3, self).__init__()
        self.conv1 = GCNConv(in_size, mid_size)
        self.conv2 = GCNConv(mid_size, mid_size)
        self.conv_mu = GCNConv(mid_size, out_size)
        self.conv_logstd = GCNConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    


# parameters
out_channels = 200 
num_features = 1024
mid_channels = 256
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


#################
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
########################




main_auc_tuning_df = pd.DataFrame()
main_ap_tuning_df = pd.DataFrame()

for lr in [0.1, 0.01, 0.001]:
    for i,encoder in enumerate([CMAPEncoder1, CMAPEncoder2, CMAPEncoder3]):
        model = VGAE(encoder(num_features, mid_channels,out_channels))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print_model_summary(model)
        print(lr, encoder)
        best_val_auc = 0

        for epoch in range(epochs):
            train_data = torch.load(train_files[epoch])
            trloss = train(train_data.to(device))

            test_aucs, test_aps = [], []
            for test_data in test_files:
                test_data = torch.load(test_data)
                test_data = test_data.to(device)
                test_auc, test_ap, test_loss = test(test_data)
                test_aucs.append(test_auc)
                test_aps.append(test_ap)
            mean_test_auc = np.mean(test_aucs)
            mean_test_ap = np.mean(test_aps)


            val_aucs, val_aps = [], []
            for val_data in val_files:
                val_data = torch.load(val_data)
                val_data = val_data.to(device)
                val_auc, val_ap, val_loss = test(val_data)
                val_aucs.append(val_auc)
                val_aps.append(val_ap)
            mean_val_auc = np.mean(val_aucs)
            mean_val_ap = np.mean(val_aps)


            if mean_val_auc > best_val_auc:
                best_val_auc = mean_val_auc
                torch.save(model.state_dict(), f'/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/cmap_vgae.pt')
            
            print(f'epoch {epoch+1} train loss : {trloss} | test loss : {test_loss} | val loss : {val_loss} | mean test auc : {mean_test_auc} | mean val auc : {mean_val_auc} | mean test ap : {mean_test_ap} | mean val ap : {mean_val_ap}')
        #####
        model.load_state_dict(torch.load('/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/cmap_vgae.pt', weights_only=True))

        test_aucs, test_aps = [], []
        for test_data in test_files:
            test_data = torch.load(test_data)
            test_data = test_data.to(device)
            test_auc, test_ap, _ = test(test_data)
            test_aucs.append(test_auc)
            test_aps.append(test_ap)

        val_aucs, val_aps = [], []
        for val_data in test_files:
            val_data = torch.load(val_data)
            val_data = val_data.to(device)
            val_auc, val_ap, _ = test(val_data)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)

        #################################################
        auc_df = pd.DataFrame({
            "AUROC": test_aucs + val_aucs,
            "Distribution": ["Test"] * len(test_aucs) + ["Validation"] * len(val_aucs),
            "Parameters": [f'GCN layers-{i+1}, lr-{lr}']*len(test_aucs + val_aucs)
        })

        main_auc_tuning_df = pd.concat([main_auc_tuning_df, auc_df], ignore_index= True)

        ap_df = pd.DataFrame({
            "AP": test_aps + val_aps,
            "Distribution": ["Test"] * len(test_aps) + ["Validation"] * len(val_aps),
            "Parameters": [f'GCN layers-{i+1}, lr-{lr}']*len(test_aps + val_aps)
        })

        main_ap_tuning_df = pd.concat([main_ap_tuning_df, ap_df], ignore_index=True)

        

plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df, kind='violin', hue='Parameters')
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/violinplot_roc.png', bbox_inches='tight')


plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df, kind='box', hue='Parameters')
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/boxplot_roc.png', bbox_inches='tight')


plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df, kind='violin', hue='Parameters')
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/violinplot_ap.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df, kind='box', hue='Parameters')
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('/home/hpc_users/2020s17811@stu.cmb.ac.lk/structure_part/boxplot_ap.png', bbox_inches='tight')

