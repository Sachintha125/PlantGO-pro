#!/usr/bin/env python

from create_nrPDB_GO_annot import read_fasta
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from functools import partial
import numpy as np
import csv
import os
import multiprocessing


annot_file = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/preprocess/_annot.tsv'   ## created tsv file 
seqres_file = './data/pdb_seqres.txt.gz'    ## from sift 
out_dir = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/model_compare/DeepFRI/preprocessing/cmaps/' # dist map saving loc
cif_dir = '/home/hpc_users/2020s17811@stu.cmb.ac.lk/structureSeq/unzipped'  # cif files loc
n_threads = 50

def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()
    return ca.chains, cb.chains


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
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames


def load_EC_annot(filename):
    """ Load EC annotations """
    prot2annot = {}
    ec_numbers = []
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = next(reader)
        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            prot2annot[prot] = [ec_num for ec_num in prot_ec_numbers.split(',')]
    return prot2annot, ec_numbers


def retrieve_pdb(pdb, chain, chain_seqres, pdir):
    ca, cb = make_distance_maps(pdir + '/' + pdb +'.cif', chain=chain, sequence=chain_seqres)
    return ca[chain]['contact-map'], cb[chain]['contact-map']


def load_list(fname):
    """
    Load PDB chains
    """
    pdb_chain_list = set()
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.add(line.strip())
    fRead.close()

    return pdb_chain_list


def write_annot_npz(prot, prot2seq=None, out_dir=None):
    """
    Write to *.npz file format.
    """
    pdb, chain = prot.split('-')
    print ('pdb=', pdb, 'chain=', chain)
    try:
        A_ca, A_cb = retrieve_pdb(pdb.lower(), chain, prot2seq[prot], pdir=cif_dir)
        np.savez_compressed(os.path.join(out_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print (e)


####################################################################################################
# load annotations
prot2goterms, _, _ = load_GO_annot(annot_file)

# load sequences
prot2seq = read_fasta(seqres_file)
print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))
to_be_processed = set(prot2goterms.keys())

# process on multiple cpus
nprocs = n_threads
nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
if nprocs > 4:
    pool = multiprocessing.Pool(processes=nprocs)
    pool.map(partial(write_annot_npz, prot2seq=prot2seq, out_dir=out_dir),
                 to_be_processed)
else:
    for prot in to_be_processed:
        write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)
