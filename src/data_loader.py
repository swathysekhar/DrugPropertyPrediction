from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from src.functional_groups import extract_functional_groups
MAX_SEQ_LEN = 100
def load_data_long(dataset, device):
    mole_dict = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: " Ne",
                11: "Na", 12:"Mg", 13: "Al", 14:"Si", 15:"P", 16: "S", 17: "Cl", 18:"Ar", 19:"K", 20:"Ca", 22:"Ti", 24:"Cr", 26:"Fe", 28:"Ni",
                29:"Cu", 31:"Ga", 32:"Ge", 34:"Se", 35:"Br", 40:"Zr", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 50:"Sn", 51:"Sb", 52:"Te", 53: "I", 65:"Tb", 75:"Re", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
                81:"Tl", 82:"Pb", 83:"Bi"}

    pair_list = ["Br", "Cl", "Si", "Na", "Ca", "Ge", "Cu", "Au", "Sn", "Tb", "Pt", "Re", "Ru", "Bi", "Li", "Fe", "Sb", "Hg","Pb", "Se", "Ag","Cr","Pd","Ga","Mg","Ni","Ir","Rh","Te","Ti","Al","Zr","Tl"]

    data_file = f"/content/drive/MyDrive/DrugPropertyProject/TraGT/datasets/{dataset}_train.txt"
    file = open(data_file, "r")
    node_types = set()
    label_types = set()
    tr_len = 0
    for line in file:
        tr_len += 1
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        i = 0
        s = []
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                s.append(smiles[i] + smiles[i+1])
                i += 2
            else:
                s.append(smiles[i].upper())
                i += 1
        node_types |= set(s)
        label_types.add(label)
    file.close()

    te_len = 0
    data_file = f"/content/drive/MyDrive/DrugPropertyProject/TraGT/datasets/{dataset}_test.txt"
    file = open(data_file, "r")
    for line in file:
        te_len += 1
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        i = 0
        s = []
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                s.append(smiles[i] + smiles[i+1])
                i += 2
            else:
                s.append(smiles[i].upper())
                i += 1
        node_types |= set(s)
        label_types.add(label)
    file.close()

    #print(tr_len)
    #print(te_len)

    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}

    #print(node2index)
    #print(label2index)

    data_file = f"/content/drive/MyDrive/DrugPropertyProject/TraGT/datasets/{dataset}_train.txt"
    file = open(data_file, "r")
    train_adjlists = []
    train_features = []
    train_sequence = []
    train_fg = []
    train_labels = torch.zeros(tr_len)
    for line in file:
        smiles = line.split("\t")[1]
        fg = extract_functional_groups(smiles)
        label = line.split("\t")[2][:-1]
        mol = Chem.MolFromSmiles(smiles)
        graph_nodes = []
        for atom in mol.GetAtoms():
            graph_nodes.append(mole_dict[atom.GetAtomicNum()])
        # print(graph_nodes)
        i = 0
        s = 0
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                i += 2
            else:
                i += 1
            s += 1

        feature = torch.zeros(s, len(node_types))

        atom_to_seq_map = {}
        se_num = 0
        gr_num = 0
        i = 0
        smiles_seq = []
        while i < len(smiles):
            this_str = smiles[i]
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                this_str = smiles[i] + smiles[i+1]
                i += 2
            else:
                this_str = this_str.upper()
                i += 1
            smiles_seq.append(node2index[this_str])
            if this_str in graph_nodes and this_str == mole_dict[mol.GetAtoms()[gr_num].GetAtomicNum()]:
                map[gr_num] = se_num
                gr_num += 1
            feature[se_num, node2index[this_str]] = 1
            se_num += 1


        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # print(i,j)
            typ = bond.GetBondType()
            adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
            adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])

        # train_labels[len(train_adjlists)]= int(label2index[label])
        train_labels[len(train_adjlists)]= int(label)
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
        train_fg.append(fg)
    file.close()

    data_file = f"/content/drive/MyDrive/DrugPropertyProject/TraGT/datasets/{dataset}_test.txt"
    file = open(data_file, "r")
    test_adjlists = []
    test_features = []
    test_sequence = []
    test_fg = []
    test_labels = np.zeros(te_len)
    for line in file:
        smiles = line.split("\t")[1]
        # print(smiles)
        fg = extract_functional_groups(smiles)
        label = line.split("\t")[2][:-1]
        mol = Chem.MolFromSmiles(smiles)
        graph_nodes = []
        for atom in mol.GetAtoms():
            graph_nodes.append(mole_dict[atom.GetAtomicNum()])
        # print(graph_nodes)
        i = 0
        s = 0
        while i < len(smiles):
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                i += 2
            else:
                i += 1
            s += 1

        feature = torch.zeros(s, len(node_types))

        atom_to_seq_map = {}
        se_num = 0
        gr_num = 0
        i = 0
        smiles_seq = []
        while i < len(smiles):
            this_str = smiles[i]
            if i < len(smiles)-1 and (smiles[i] + smiles[i+1]) in pair_list:
                this_str = smiles[i] + smiles[i+1]
                i += 2
            else:
                this_str = this_str.upper()
                i += 1
            smiles_seq.append(node2index[this_str])
            if this_str in graph_nodes and this_str == mole_dict[mol.GetAtoms()[gr_num].GetAtomicNum()]:
                atom_to_seq_map[gr_num] = se_num
                gr_num += 1
            feature[se_num, node2index[this_str]] = 1
            se_num += 1

        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # print(i,j)
            typ = bond.GetBondType()
            adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
            adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])
                adj_list[atom_to_seq_map[i]].append(atom_to_seq_map[j])
                adj_list[atom_to_seq_map[j]].append(atom_to_seq_map[i])

        # test_labels[len(test_adjlists)] = int(label2index[label])
        test_labels[len(test_adjlists)] = int(label)
        test_adjlists.append(adj_list)
        test_features.append(torch.FloatTensor(feature).to(device))
        test_sequence.append(torch.tensor(smiles_seq))
        test_fg.append(fg)
    file.close()

    train_data = {}
    train_data['adj_lists'] = train_adjlists
    train_data['features'] = train_features
     # Pad train_sequence to length 100
    padded_train_sequence = []
    for seq in train_sequence:
        seq = seq[:MAX_SEQ_LEN]
        padded_seq = torch.nn.functional.pad(seq,(0, MAX_SEQ_LEN - len(seq)),'constant',0)
        padded_train_sequence.append(padded_seq)
    train_data['sequence'] = padded_train_sequence
    train_data['fg'] = train_fg

    test_data = {}
    test_data['adj_lists'] = test_adjlists
    test_data['features'] = test_features
    padded_test_sequence = []
    for seq in test_sequence:
      seq = seq[:MAX_SEQ_LEN]
      padded_seq = torch.nn.functional.pad(seq, (0, MAX_SEQ_LEN - len(seq)), 'constant', 0)
      padded_test_sequence.append(padded_seq)
    test_data['sequence'] = padded_test_sequence
    test_data['fg'] = test_fg   

    return train_data, train_labels, test_data, test_labels

class CustomDataset(Dataset):
    def __init__(self, data_list, sequence_list, fg_list):
        self.data_list = data_list
        self.sequence_list = sequence_list
        self.fg_list = fg_list   

    def __getitem__(self, index):
        data = self.data_list[index]
        sequence = self.sequence_list[index]
        fg = self.fg_list[index]   

        return data, sequence, fg   

    def __len__(self):
        return len(self.data_list)

def adj_list_to_adj_matrix(adj_list):
    num_nodes = max(adj_list.keys()) + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix