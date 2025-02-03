from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from torch_geometric.data import Data

# #Inializing here
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

distance_threshold = 10.0

def contact_map_to_edge_index(contact_map):
    row, col = torch.nonzero(contact_map, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def extract_ContactMap_SeqEmbedds(file_name):
    pdb_id = file_name.split('/')[-1].split('.')[0]
    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, file_name)

    # Iterate through chains
    for model in structure:
        # model_id = model.get_id()
        for chain in model:
            chain_id = chain.get_id()
            ca_atoms = []
            chain_seq = ''
            for residue in chain:
                if is_aa(residue) and 'CA' in residue:
                    chain_seq += seq1(residue.resname)
                    ca_atoms.append(residue['CA'].get_coord())

            num_atoms = len(ca_atoms)
            print('ca atom count ', num_atoms)

            if len(chain_seq) > 60:
                node_features = seq2protbert(chain_seq)
                contact_map = torch.zeros((num_atoms, num_atoms))

                # Calculate distances between all pairs of C-alpha atoms
                for i in range(num_atoms):  # through each c-alpha atom
                    for j in range(i+1, num_atoms): # cal euclidean norm with other c-alphas = total c-alphas - 1
                        distance = np.linalg.norm(ca_atoms[i] - ca_atoms[j])
                        if distance <= 10.0 and distance > 0:
                            contact_map[i, j] = 1
                            contact_map[j, i] = 1

                map_name = f'{pdb_id}_{chain_id}'
                print(map_name)

                edge_index = contact_map_to_edge_index(contact_map)
                data = Data(x=node_features, edge_index=edge_index)
                torch.save(data, f'/home/hpc_users/2020s17811@stu.cmb.ac.lk/soybean_maps/{map_name}.pt')
        break

folder_path = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/alphaFold/'
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for file in file_list:
    file_path = folder_path + file
    try:
        extract_ContactMap_SeqEmbedds(file_name=file_path)
        print(file, ' done ************************************\n')
    except:
        continue

