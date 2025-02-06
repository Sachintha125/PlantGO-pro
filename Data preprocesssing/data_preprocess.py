import os
import pickle
from collections import defaultdict

cif_directory_path = 'D:\\sachintha\\structures\\'
sift_file_path = 'd:\\year 4\\semester 1\\BT\\BT 4033\\data preprocess\\pdb_chain_go.csv'
string_prot_file = 'D:\\sachintha\\data preprocess\\protein.info.v12.0.txt'
output_dir = 'd:\\year 4\\semester 1\\BT\\BT 4033\\data preprocess\\'


cif_file_list = os.listdir(cif_directory_path)
cif_file_list = [f for f in cif_file_list if os.path.isfile(os.path.join(cif_directory_path, f))]


def getPDBUnipMap(sift_path):
    pdb_unip_map = dict()   # no chain id obly pdb id

    with open(sift_path, 'r') as f:
        for line in f: 
            line = line.strip()
            if not line.startswith('#') and line: 
                line = line.split(',')
                if len(line) < 4:  
                    continue
                pdb_id, _, unip_id, *_ = line  

                if pdb_id not in pdb_unip_map:
                    pdb_unip_map[pdb_id] = unip_id
    return pdb_unip_map


def mapCIFIds(cif_file_list, pdb_unip_map):
    cif_id_list = [f.split('.')[0] for f in cif_file_list]
    plant_unip_mapped_dict = {cif: pdb_unip_map[cif] for cif in cif_id_list if cif in pdb_unip_map.keys()}
    return plant_unip_mapped_dict


def getStringIds(string_id_file):
    string_ids = []
    with open(string_id_file, 'r') as f:
        for line in f: 
            line = line.strip()
            if not line.startswith('#') and line: 
                line = line.split('\t')
                id = line[0].split('.')[1]
                string_ids.append(id)
    return string_ids


def getPDBStringIntersection(my_uniprot_list, string_ids):
    my_uniprot_list = set(my_uniprot_list)
    string_ids = set(string_ids)
    intersec = my_uniprot_list.intersection(string_ids)
    return intersec


def getPDBChains(plant_unip_mapped_dict, sift_file):
    from itertools import chain
    inverse_dict = defaultdict(list)
    for k, v in plant_unip_mapped_dict.items():
        inverse_dict[v].append(k)
    all_values = list(chain(*inverse_dict.values()))
    chains = []
    with open(sift_file, 'r') as f:
        for line in f:  
                line = line.strip()
                if not line.startswith('#') and line:  
                    line = line.split(',')
                    if len(line) < 4:  # Ensure line has enough columns
                        continue
                    pdb_id, chain,  *_ = line 
                    if pdb_id in all_values:
                        chains.append(pdb_id.upper()+'-'+chain)
    chains = set(chains)
    return chains

###############################################################################

pdb_unip_map = getPDBUnipMap(sift_file_path)
plant_unip_mapped_dict = mapCIFIds(cif_file_list, pdb_unip_map)
all_string_prots = getStringIds(string_prot_file)
plant_struct_with_ppi = getPDBStringIntersection(list(plant_unip_mapped_dict.values()), all_string_prots)
pdb_chains = getPDBChains(plant_unip_mapped_dict, sift_file_path)

with open(f'{output_dir}initial_chainids.pkl', 'wb') as f:
    pickle.dump(pdb_chains, f)

with open(f'{output_dir}plant_unip_mapped_dict.pkl', 'wb') as f:
    pickle.dump(plant_unip_mapped_dict, f)

with open(f'{output_dir}plant_struct_with_ppi_unip_id_list.pkl', 'wb') as f:
    pickle.dump(plant_struct_with_ppi, f)

