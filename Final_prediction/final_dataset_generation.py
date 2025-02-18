import torch 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE, global_add_pool
import pickle
import numpy as np
import csv
import os 

cmap_vgae_weight_file = "D:\\sachintha\\structu seq\\cmap_vgae.pt"
ppi_vgae_embedding_dict_path = "D:\\sachintha\\ppi part\\ppi_vgae_embedds_dict.pkl"
cmap_folder = 'D:\\sachintha\\structu seq\\cmaps\\'
pdb_unip_id_map_dict_path = "D:\\sachintha\\prediction\\plant_unip_mapped_dict.pkl" # simple letters for pdb id 
annot_file_path = "D:\\sachintha\\prediction\\_annot.tsv"
saving_dir = 'D:\\sachintha\\prediction\\'
train_list_file = "D:\\sachintha\\prediction\\_train.txt"
val_list_file = "D:\\sachintha\\prediction\\_valid.txt"
test_list_file = "D:\\sachintha\\prediction\\_test.txt"


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


def merge_n_label(chain_ids, cmap_folder, pdb_string_id_map_dict, prot2annot, save_file_name, input_len, label_len, aspect='biological_process', need_labeling=True):
    cmaps = os.listdir(cmap_folder)
    cmaps = [f.split('.')[0] for f in cmaps if os.path.isfile(os.path.join(cmap_folder, f))]
    cmaps = set(cmaps)
    chain_ids = set(chain_ids)
    intersec = cmaps.intersection(chain_ids)

    dataset = {'input': [], 'output': []}  
    
    for id in intersec:
        try:
            pdb_id = id.split('-')[0]   # capital letters 
            if pdb_id not in pdb_string_id_map_dict.keys():
                continue  # Skip if mapping does not exist
            
            string_id = pdb_string_id_map_dict[pdb_id]
            if string_id not in ppi_vgae_embedding_dict.keys():
                continue  # Skip if no embedding available
            
            data_obj = torch.load(cmap_folder + id + '.pt')
            pooling = genPoolEmbedd(data_obj).tolist()
            ppi = ppi_vgae_embedding_dict[string_id]
            merged = pooling + ppi
            
            dataset['input'].append(merged)
            
            if need_labeling and id in prot2annot:
                dataset['output'].append(prot2annot[id][aspect].tolist())

        except Exception as e:
            print(f"Error processing {id}: {e}")
            continue

    # Convert lists to tensors
    dataset['input'] = torch.tensor(dataset['input']).float()
    dataset['output'] = torch.tensor(dataset['output']).float()

    torch.save(dataset, f'{saving_dir}{save_file_name}.pt')
    print(f'no. of chains {dataset["input"].shape[0]}')



#######################################################################################################################################
with open(ppi_vgae_embedding_dict_path, 'rb') as f:
    ppi_vgae_embedding_dict = pickle.load(f)

with open(pdb_unip_id_map_dict_path, 'rb') as f:
    pdb_string_id_map_dict = pickle.load(f)
pdb_string_id_map_dict = {pdb.upper():string for pdb, string in pdb_string_id_map_dict.items()}


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

prot2annot, goterms, _=load_GO_annot(annot_file_path)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'train_dataset', 200, len(goterms['biological_process']), 'biological_process', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'test_dataset', 200, len(goterms['biological_process']), 'biological_process', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'val_dataset', 200, len(goterms['biological_process']), 'biological_process', True)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'train_dataset', 200, len(goterms['molecular_function']), 'molecular_function', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'test_dataset', 200, len(goterms['molecular_function']), 'molecular_function', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'val_dataset', 200, len(goterms['molecular_function']), 'molecular_function', True)

merge_n_label(train_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'train_dataset', 200, len(goterms['cellular_component']), 'cellular_component', True)
merge_n_label(test_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'test_dataset', 200, len(goterms['cellular_component']), 'cellular_component', True)
merge_n_label(val_list, cmap_folder, pdb_string_id_map_dict, prot2annot, 'val_dataset', 200, len(goterms['cellular_component']), 'cellular_component', True)