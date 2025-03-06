import torch 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE, global_add_pool
import pickle
import numpy as np
import csv
import os 


cmap_vgae_weight_file = r"D:\sachintha\structu seq\cmap_vgae.pt"
ppi_vgae_embedding_dict_path = r"D:\sachintha\ppi part\ppi_vgae_embedds_dict.pkl"
cmap_folder = 'D:\\sachintha\\structu seq\\cmaps\\'
sift_file_path = r"D:\sachintha\data preprocess\pdb_chain_go.csv"
annot_file_path = r"D:\sachintha\data preprocess\_annot.tsv"
saving_dir = 'D:\\sachintha\\prediction\\'
train_list_file = r"D:\sachintha\data preprocess\_train.txt"
val_list_file = r"D:\sachintha\data preprocess\_valid.txt"
test_list_file = r"D:\sachintha\data preprocess\_test.txt"
pdb_unip_mapper_file = r'D:\sachintha\data preprocess\pdb_chain_uniprot.csv'


class CmapEncdoer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CmapEncdoer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels) 
        self.conv_logstd = GCNConv(in_channels, out_channels) ##

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index) , self.conv_logstd(x, edge_index)


# parameters
num_features = 1024
out_channels = 200

device = 'cpu'
# model
model = VGAE(CmapEncdoer(num_features, out_channels))
model = model.to(device)
model.load_state_dict(torch.load(cmap_vgae_weight_file, weights_only=True))
model.eval()


def genPoolEmbedd(data):
    with torch.no_grad():
        graph_embedd = model.encode(data.x, data.edge_index)
        pooled = global_add_pool(graph_embedd, torch.zeros(graph_embedd.shape[0], dtype=torch.long)).flatten()
    return pooled


def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        for row in reader:
            print(row)
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
    return prot2annot, goterms, gonames


def readPDBUnipMap(mapper_file_path):
    pdb_unip_map = dict()
    with open(mapper_file_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb_id = row[0].strip().upper()
            chain_id = row[1].strip()
            unip_id = row[2].strip()
            pdb_chain = pdb_id + '-' + chain_id
            pdb_unip_map[pdb_chain] = unip_id
    print('total mapped pdb chains:', len(pdb_unip_map))
    print(list(pdb_unip_map.keys())[:5])
    print(list(pdb_unip_map.values())[:5])
    return pdb_unip_map # upper case pdb id 


def merge_n_label(chain_ids:list, cmap_folder, pdb_string_id_map_dict:dict, prot2annot:dict, ppi_vgae_emb_dict:dict, save_file_name,  aspect='biological_process', need_labeling=True):
    dataset = {'input': [], 'output': []}  
    
    for id in chain_ids:
        string_id = pdb_string_id_map_dict.get(id)
        if string_id is not None:  
            ppi = ppi_vgae_emb_dict.get(string_id)  
            if ppi is not None:
                try:    
                    data_obj = torch.load(cmap_folder + id + '.pt')
                    pooling = genPoolEmbedd(data_obj).tolist()            
                    merged = pooling + ppi
                    dataset['input'].append(merged)
                    if need_labeling and id in prot2annot:
                        dataset['output'].append(prot2annot[id][aspect].tolist())
                except FileNotFoundError:
                    print(id, '.pt not dound')
            else:
                print(f'{string_id} not available in vgae emb dict')
        else:
            print(f'{id} not availabel in id mapper')


    # Convert lists to tensors
    dataset['input'] = torch.tensor(dataset['input']).float()
    dataset['output'] = torch.tensor(dataset['output']).float()

    torch.save(dataset, f'{saving_dir}{save_file_name}.pt')
    print(f'no. of chains {dataset["input"].shape[0]}')



#######################################################################################################################################
with open(ppi_vgae_embedding_dict_path, 'rb') as f:
    ppi_vgae_embedding_dict = pickle.load(f)

pdb_string_id_map_dict = readPDBUnipMap(pdb_unip_mapper_file)

train_list, test_list, val_list = [], [], []
with open(train_list_file, 'r') as f:
    for line in f:
        train_list.append(line.strip())

with open(test_list_file, 'r') as f:
    for line in f:
        test_list.append(line.strip())

with open(val_list_file, 'r') as f:
    for line in f:
        val_list.append(line.strip())

prot2annot, _, _=load_GO_annot(annot_file_path)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'train_dataset',  'biological_process', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'test_dataset',  'biological_process', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'val_dataset', 'biological_process', True)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'train_dataset',  'molecular_function', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'test_dataset', 'molecular_function', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'val_dataset', 'molecular_function', True)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'train_dataset',  'cellular_component', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'test_dataset', 'cellular_component', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, ppi_vgae_embedding_dict,'val_dataset', 'cellular_component', True)