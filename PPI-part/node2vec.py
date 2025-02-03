import os
import pandas as pd
import torch
from torch_geometric.data import Data
import sys
import pickle
from torch_geometric.nn.models import Node2Vec

directory_path = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/string_networks/'
file_list = os.listdir(directory_path)
file_list = [f for f in file_list if os.path.isfile(os.path.join(directory_path, f))]

merged_df = pd.DataFrame()

for file in file_list:
    tax = file.split('.')[0]
    file_path = directory_path + file   
    df = pd.read_csv(file_path, sep="\t")

    #passing index for each node
    col1 = df['protein1'].to_list()
    col2 = df['protein2'].to_list()
    full_list = col1 + col2
    unique_list = list(dict.fromkeys(full_list))    

    ind2node = {index: item for index, item in enumerate(unique_list)} 
    node2ind = {v: k for k, v in ind2node.items()}  

    df['protein1'] = df['protein1'].map(node2ind)
    df['protein2'] = df['protein2'].map(node2ind)

    # prepare interaction to be used as COO format
    first_prot = df['protein1'].to_list()   
    second_prot = df['protein2'].to_list()

    print("List of numbers for Column 1:", first_prot)
    print("List of numbers for Column 2:", second_prot)

    edge_index = torch.tensor([first_prot, second_prot], dtype=torch.long)
    data = Data(edge_index=edge_index)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize Node2Vec model
    embedding_dim = 256  
    node2vec = Node2Vec(
        edge_index=data.edge_index.to(device),
        embedding_dim=embedding_dim,
        walk_length=10, # rnadom walk length
        context_size=10,    # window size for skip-gram model
        walks_per_node=10,
        p=1,    # exploration parameter
        q=2,    # return paramter
        sparse=True).to(device)

    num_workers = 4 if sys.platform == 'linux' else 0
    loader = node2vec.loader(batch_size=64, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    # training loop function
    def train():
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    for epoch in range(1, 50):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    node_embeddings = node2vec().detach().cpu().tolist() # node2vec embeddings 

    node_embedding_df = pd.DataFrame(columns=['protein', 'index', 'embedding'])
    node_embedding_df['protein'] = ind2node.values()
    node_embedding_df['index'] = ind2node.keys()
    node_embedding_df['embedding'] = node_embedding_df['index'].map(lambda idx: node_embeddings[idx])
    node_embedding_df = node_embedding_df[['protein', 'embedding']]
    merged_df = pd.concat([merged_df, node_embedding_df], ignore_index=True)
    # node_embedding_df.to_pickle(f'/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/node2vec/{tax}_node2vec_embedd_dframe.pkl')

ppi_dict = dict(zip(merged_df['protein'].to_list(), merged_df['embedding'].to_list()))

with open('/home/hpc_users/2020s17811@stu.cmb.ac.lk/ppi/node2vec_dict.pkl', 'wb') as f:
    pickle.dump(ppi_dict, f)

