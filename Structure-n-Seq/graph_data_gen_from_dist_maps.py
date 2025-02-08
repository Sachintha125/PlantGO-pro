from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from torch_geometric.data import Data

dist_map_dir = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/model_compare/DeepFRI/preprocessing/cmaps/'
out_dir = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/structureSeq/cmap_graph_datas/'
distance_threshold = 10.0


tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()


def seq2protbert(seq):
    seq = ' '.join(seq)
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

    embeddings = embeddings 
    attention_mask = attention_mask 
    features = []
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        if seq_len > 2:
            seq_emd = embeddings[seq_num][1:seq_len-1]  # without [CLS] and [SEP]
            features.append(seq_emd)

    # Convert list of arrays to 2D array
    if features:
        features_2d = torch.vstack(features).to('cpu')  # Stack all sequences into a 2D array
        return features_2d


def contact_map_to_edge_index(contact_map):
    row, col = torch.nonzero(contact_map, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def extract_ContactMap_SeqEmbedds(npz_file_name):
    chain_id = npz_file_name.split('/')[-1].split('.')[0]
    npz = np.load(npz_file_name)
    dist_map = npz['C_alpha']
    seq = np.array2string(npz['seqres'])
    node_features = seq2protbert(seq)
    contact_map = (dist_map <= distance_threshold).astype(int)
    contact_map = torch.from_numpy(contact_map)
    
    if contact_map.shape[0] == len(seq):
        edge_index = contact_map_to_edge_index(contact_map)
        data = Data(x=node_features, edge_index=edge_index)
        torch.save(data, f'{out_dir}{chain_id}.pt')


#############################################################################################################
npz_file_list = [f for f in os.listdir(dist_map_dir) if os.path.isfile(os.path.join(dist_map_dir, f))]

for file in npz_file_list:
    file_path = dist_map_dir + file
    extract_ContactMap_SeqEmbedds(npz_file_name=file_path)
    print(file, ' done ************************************\n')

