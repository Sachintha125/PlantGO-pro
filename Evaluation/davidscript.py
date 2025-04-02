import pickle
import os
import csv
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE, global_add_pool
from sklearn.preprocessing import StandardScaler
import numpy as np

ppi_vgae_embeddings_file = "/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/ppi_vgae_embedds_dict.pkl"
cmap_vgae_weight_file = "/home/hpc_users/2020s17811@stu.cmb.ac.lk/david/cmap_vgae.pt"
classifier_weight_file = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/david/bp_best.pt'
go_annots_file = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/david/_annot.tsv'
plant_unip_map_file = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/david/plant_pdb_string_map.pkl'
maize_maps = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/alphaFold/maize_maps/'
rice_maps = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/alphaFold/oryza_maps/'
soy_maps = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/alphaFold/soybean_maps/'
out_dir = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/david/'

with open(ppi_vgae_embeddings_file, 'rb') as f:
    ppi_embeddings = pickle.load(f) # capital uni ids # a dict 
print('PPI embeddings loaded')

with open(plant_unip_map_file, 'rb') as f:
    plant_unip_map_dict = pickle.load(f) # dict
plant_string_ids = set(plant_unip_map_dict.values())
print('mapper loaded')

def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
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
    return goterms

go_terms = load_GO_annot(go_annots_file)
print('GO annotations loaded')
bp_go_terms = go_terms['biological_process']
root_development_index = bp_go_terms.index('GO:0048364')
print('index of root development ', root_development_index)

##################################################################
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

# model
cmap_model = VGAE(CmapEncdoer(num_features, out_channels))
cmap_model = cmap_model
cmap_model.load_state_dict(torch.load(cmap_vgae_weight_file, weights_only=True, map_location='cpu'))
cmap_model.eval()
print('cmap vgae loaded')

def genPoolEmbedd(data):
    with torch.no_grad():
        graph_embedd = cmap_model.encode(data.x, data.edge_index)
        pooled = global_add_pool(graph_embedd, torch.zeros(graph_embedd.shape[0], dtype=torch.long)).flatten()
    return pooled


class PredictorModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout_prob):
        super(PredictorModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, input_size))  
        self.dropout = nn.Dropout(p=dropout_prob)
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_size, input_size))  
        
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

bp_model = PredictorModel(400, len(bp_go_terms), 0, 0.3)
bp_model.load_state_dict(torch.load(classifier_weight_file, weights_only=True))
bp_model.eval()
print('bp model loaded')

##########################################################################################

scaler = StandardScaler()
prediction_thresh = 0.5

for species in [maize_maps, rice_maps, soy_maps]:
    org = species.split('/')[-1]
    david_candidates = []
    maps_list = os.listdir(species)
    print(maps_list)
    for map in maps_list:
        id = map.split('-')[1]
        if id not in plant_string_ids:
            data_obj = torch.load(f'{species}{map}')
            pooled = genPoolEmbedd(data_obj).tolist()
            ppi = ppi_embeddings.get(id) 
            if ppi is not None:
                merged = pooled + ppi
                merged = np.array([merged])
                # merged = scaler.fit_transform(merged)
                merged = torch.from_numpy(merged).float()
                with torch.no_grad():
                    outputs = bp_model(merged).sigmoid().flatten().tolist()
                root_dev_score = outputs[root_development_index]
                print(id, " ", root_dev_score)
                if root_dev_score >= prediction_thresh:
                    david_candidates.append(id)

    with open(f'{out_dir}{org}_new_root_genes.txt', 'w') as f:
        for prot in david_candidates:
            f.write(f"{prot}\n")
    f.close()
