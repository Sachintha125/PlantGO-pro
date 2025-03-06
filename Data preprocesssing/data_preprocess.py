import os
import pickle
from Bio import SeqIO
import csv
import networkx as nx
import obonet
import pickle
import numpy as np

cif_directory_path = r'D:\sachintha\unzipped'
sift_file_path = r"D:\sachintha\data preprocess\pdb_chain_go.csv"
output_dir = 'D:\\sachintha\\data preprocess\\'
seqres_file = r'D:\sachintha\data preprocess\pdb_seqres.txt'
obo_file = r"D:\sachintha\data preprocess\go-basic.obo"
ppi_vgae_embeddings_file = r"D:\sachintha\ppi part\ppi_vgae_embedds_dict.pkl"
pdb_unip_mapper_file = r'D:\sachintha\data preprocess\pdb_chain_uniprot.csv'


exp_evidence_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED'])
root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])

cif_file_list = os.listdir(cif_directory_path)
cif_file_list = [f.split('.')[0].upper() for f in cif_file_list] # cap pdb ids 
print('sample cif ids:', cif_file_list[:4])
print('all cif files:', len(cif_file_list))


def load_go_graph(fname):
    # read *.obo file
    go_graph = obonet.read_obo(open(fname, 'r'))
    return go_graph


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    
    handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1200:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq
    print('init prot2seq len :', len(prot2seq))
    return prot2seq # cap pdb chain id and chain with seq 


def read_sifts(fname, chains, go_graph):
    print ("### Loading SIFTS annotations...")
    pdb2go = {}
    go2info = {}
    with open(fname, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            evidence = row[4].strip()
            go_id = row[5].strip()
            pdb_chain = pdb + '-' + chain
            if (pdb_chain in chains) and (go_id in go_graph) and (go_id not in root_terms):
                if pdb_chain not in pdb2go:
                    pdb2go[pdb_chain] = {'goterms': [go_id], 'evidence': [evidence]}
                namespace = go_graph.nodes[go_id]['namespace']
                go_ids = nx.descendants(go_graph, go_id)
                go_ids.add(go_id)
                go_ids = go_ids.difference(root_terms)
                for go in go_ids:
                    pdb2go[pdb_chain]['goterms'].append(go)
                    pdb2go[pdb_chain]['evidence'].append(evidence)
                    name = go_graph.nodes[go]['name']
                    if go not in go2info:
                        go2info[go] = {'ont': namespace, 'goname': name, 'pdb_chains': set([pdb_chain])}
                    else:
                        go2info[go]['pdb_chains'].add(pdb_chain)
    print('annotated chains:', len(pdb2go))
    return pdb2go, go2info


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


def getPlantChains(sift_path, cif_files:list):
    plant_chains = set()

    with open(sift_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb_id = row[0].strip().upper()
            chain_id = row[1].strip()
            if pdb_id in cif_files:
                plant_chains.add(f'{pdb_id}-{chain_id}')
    print('total plant chain ids:', len(plant_chains))
    return plant_chains # upper case pdb id + chain


def getpPlantChainUnipIds(chain_unip_map:dict, plant_chains:set):
    plantChainUnipMap = dict()
    for chain in plant_chains:
        unip_id = chain_unip_map.get(chain)
        if unip_id is not None:
            plantChainUnipMap[chain] = unip_id
    print('total unip mapped plant chains:', len(plantChainUnipMap))
    return plantChainUnipMap


def getPDBStringIntersection(plantChainMap:dict, ppi_embedds:dict):
    filtered_chains = []
    for chain, unipId in plantChainMap.items():
        ppi_vector = ppi_embedds.get(unipId)
        if ppi_vector is not None:
            filtered_chains.append(chain)

    print('number of plant chains having PPI ', len(filtered_chains))
    return filtered_chains 


def write_prot_list(protein_list, filename):
    # write list of protein IDs
    fWrite = open(filename, 'w')
    for p in protein_list:
        fWrite.write("%s\n" % (p))
    fWrite.close()


def write_output_files(fname, pdb2go, go2info):
    # select goterms (> 49, < 5000)
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()
    for goterm in go2info:
        prots = go2info[goterm]['pdb_chains']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        if num > 49 and num <= 5000:
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    for ont in onts:
        print ("###", ont, ":", len(selected_goterms_list[ont]))

    protein_list = []
    with open(fname + '_annot.tsv', 'wt',newline="") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for ont in onts:
            tsv_writer.writerow(["### GO-terms (%s)" % (ont)])
            tsv_writer.writerow(selected_goterms_list[ont])
            tsv_writer.writerow(["### GO-names (%s)" % (ont)])
            tsv_writer.writerow(selected_gonames_list[ont])
        tsv_writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        for chain in selected_proteins:
            goterms = set(pdb2go[chain]['goterms'])
            if len(goterms) > 2:
                # selected goterms
                mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                    protein_list.append(chain)
                    tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms)])

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number of annot PDB=%d" % (len(protein_list)))

    test_split = int(0.1*len(protein_list))
    test_list = protein_list[:test_split]
    test_list = set(test_list)
    
    # i = 0
    # while len(test_list) < int(len(protein_list)*0.1) and i < len(protein_list):
    #     goterms = pdb2go[protein_list[i]]['goterms']
    #     evidence = pdb2go[protein_list[i]]['evidence']
    #     goterm2evidence = {goterms[i]: evidence[i] for i in range(len(goterms))}

    #     # selected goterms
    #     mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
    #     bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
    #     cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))

    #     mf_evidence = [goterm2evidence[goterm] for goterm in mf_goterms]
    #     mf_evidence = [1 if evid in exp_evidence_codes else 0 for evid in mf_evidence]

    #     bp_evidence = [goterm2evidence[goterm] for goterm in bp_goterms]
    #     bp_evidence = [1 if evid in exp_evidence_codes else 0 for evid in bp_evidence]

    #     cc_evidence = [goterm2evidence[goterm] for goterm in cc_goterms]
    #     cc_evidence = [1 if evid in exp_evidence_codes else 0 for evid in cc_evidence]

    #     if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
    #         if sum(mf_evidence) > 0 and sum(bp_evidence) > 0 and sum(cc_evidence) > 0:
    #             test_list.add(protein_list[i])
    #     i += 1

    print ("Total number of test PDB=%d" % (len(test_list)))

    protein_list = list(set(protein_list).difference(test_list))
    np.random.shuffle(protein_list)

    idx = len(protein_list) - len(test_list)
    write_prot_list(test_list, fname + '_test.txt')
    write_prot_list(protein_list[:idx], fname + '_train.txt')
    write_prot_list(protein_list[idx:], fname + '_valid.txt')

###############################################################################
if __name__ == "__main__":

    with open(ppi_vgae_embeddings_file, 'rb') as f:
        ppi_vgae_embeddings = pickle.load(f)

    pdb2seq = read_fasta(seqres_file)
    pdb2seq_chains = set(pdb2seq.keys())
    go_graph = load_go_graph(obo_file)
    pdb_unip_map = readPDBUnipMap(pdb_unip_mapper_file)
    plant_chains = getPlantChains(sift_file_path, cif_file_list)
    plant_chains = plant_chains.intersection(pdb2seq_chains)
    plantChainUnipMap = getpPlantChainUnipIds(pdb_unip_map, plant_chains)
    plantChainStringIntersection = getPDBStringIntersection(plantChainUnipMap, ppi_vgae_embeddings)
    pdb2go, go2info = read_sifts(sift_file_path, plantChainStringIntersection, go_graph)
    write_output_files(output_dir, pdb2go, go2info)

