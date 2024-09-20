import copy
import torch
import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa

alphabet = '-ACDEFGHIKLMNPQRSTVWY'

class Proteins():

    def __init__(self, pdb_dir, chains_info, radius = 5.0):
        self.radius = radius
        parser = PDBParser(QUIET=True)
        try: model = parser.get_structure('structure', pdb_dir)[0]
        except Exception as e:
            print('pdb structure cannot be parsed, please check the input file.')
            raise
        self.complex = self.make_protein(model, chains_info)

    def get_valided_res(self, chain):

        chain_name = str(chain.get_id())
        seq_index = []
        residues = []

        for residue in chain:

            if is_aa(residue.get_resname(), standard=True):
                index = int(residue.get_id()[1])
                if index not in seq_index: seq_index.append(index)      # ensure the uniqueness and continuity of the sequence
                residues.append(residue)
            else:
                seq_index.append(0)

        return chain_name, seq_index, residues

    def select_interface(self, ligand, ns):

        residue_selected = []
        for chain in ligand:
            for residue in chain:
                for atom in residue:
                    neighbors = ns.search(atom.coord, self.radius, level='R')
                    for neighbor in neighbors:
                        num = int(neighbor.get_id()[1])
                        residue_selected.append(num)
        
        residue_selected = sorted(set(residue_selected))

        return residue_selected
    
    def form_seq_coords(self, chain_dict):

        residues = chain_dict['residues']
        residues_icode = []
        res_dict = {}
        backbone_atom = ['N', 'CA', 'C', 'O']
        for residue in residues:
            index = int(residue.get_id()[1])
            icode = residue.get_id()[2]

            if icode != ' ':
                residues_icode.append(residue)
                continue

            res_dict[index] = {}
            res_dict[index]['aa_one'] = three_to_one(residue.get_resname())

            for atom in backbone_atom:
                coord = residue[atom].get_coord().tolist()
                res_dict[index][atom] = [round(num,3) for num in coord]
        
        
        seq_index = chain_dict['seq_index']
        chain_seq = ""
        coord_dict = {key: [] for key in backbone_atom}
        for index in seq_index:
            if index in res_dict: 
                chain_seq += res_dict[index]['aa_one']
                for key in coord_dict: coord_dict[key].append(res_dict[index][key])

            else: 
                chain_seq += '-'
                for key in coord_dict: coord_dict[key].append([0,0,0])
        
        return chain_seq, coord_dict, residues_icode
    
    def trim_proteins(self, structure):

        for chain in structure:
            if chain == 'flag': continue
            start_index = (structure[chain]['mask'] != 0).nonzero()[0].item()
            end_index = (structure[chain]['mask'] != 0).nonzero(as_tuple=False).max().item()
            if end_index - start_index + 1 == len(structure[chain]['mask']): continue
            for key in structure[chain]:
                if key == 'coords':
                    for atom in structure[chain][key]: 
                        structure[chain][key][atom] = structure[chain][key][atom][start_index:end_index+1]
                elif key == 'residues_icode': continue
                else: 
                    structure[chain][key] = structure[chain][key][start_index:end_index+1]
                    
        return structure

    def form_resicode_dict(self, res_icode):

        backbone_atom = ['N', 'CA', 'C', 'O']
        icode_dict = {}

        for residue in res_icode:
            coords = {}
            index = int(residue.get_id()[1])
            if index not in icode_dict: icode_dict[index] = {}
            icode = residue.get_id()[2]
            if icode not in icode_dict[index]: icode_dict[index][icode] = {}
            aa_name = three_to_one(residue.get_resname())
            for atom in backbone_atom:
                coord = residue[atom].get_coord().tolist()
                coords[atom] = [round(num,3) for num in coord]

            icode_dict[index][icode]['aa_name'] = aa_name
            icode_dict[index][icode]['coords'] = coords
            # icode_dict[index][icode]['icode'] = icode

        return icode_dict
    
    def make_protein(self, model, chains_info):
        # Filter the standard residues and generate mask
        protein = {}
        for chain in model:
            chain_name, seq_index, residues = self.get_valided_res(chain)
            if chain_name not in chains_info: continue    # excluding chains of non-antibody or non-antigen
            mask = torch.zeros(len(seq_index))
            # print('length:', len(seq_index))
            standard_aa_indices = torch.tensor(seq_index).nonzero().squeeze()
            mask[standard_aa_indices] = 1
            # print("mask:", mask)
            protein[chain_name] = {}
            protein[chain_name]['seq_index'] = seq_index
            protein[chain_name]['residues'] = residues
            protein[chain_name]['mask'] = mask

        # Identify interaction regions
        ab_chains = str(chains_info.split('_')[0])    # 'HL'
        ag_chains = str(chains_info.split('_')[1])    # 'AB'
        for chain in protein:
            ns = NeighborSearch([atom for residue in protein[chain]['residues'] for atom in residue.get_atoms()])
            if chain in ab_chains:
                ligands = [protein[x]['residues'] for x in ag_chains]
            else:
                ligands = [protein[x]['residues'] for x in ab_chains]
            interface = self.select_interface(ligands, ns)
            for site in interface:
                if site in protein[chain]['seq_index']:
                    index = protein[chain]['seq_index'].index(site)
                    protein[chain]['mask'][index] += 1
            # genertate sequences and coordinates
            protein[chain]['sequence'], protein[chain]['coords'], residues_icode = self.form_seq_coords(protein[chain])
            if residues_icode != []: protein[chain]['residues_icode'] = self.form_resicode_dict(residues_icode)

        for chain in protein: 
            if chain == 'flag': continue
            del protein[chain]['residues']
        
        protein = self.trim_proteins(protein)   # Trim non-standard residues at the head and tail
        for chain in protein:
            if isinstance(protein[chain]['mask'], torch.Tensor): protein[chain]['mask'] = protein[chain]['mask'].tolist()

        return protein


def get_mut_pos(mut_pos):

    mutation = {}

    for snp in mut_pos.split(','):

        ori_aa = snp[0]
        mut_chain = snp[1]
        mut_aa = snp[-1]
        try:
            mut_index = int(snp[2:-1])
            icode = ''
        except:
            mut_index = int(snp[2:-2])
            icode = snp[-2].upper()

        if mut_chain not in mutation: mutation[mut_chain] = [[], [], [], []]
        if mut_index in mutation[mut_chain][0]: continue
        mutation[mut_chain][0].append(mut_index)
        mutation[mut_chain][1].append(mut_aa)
        mutation[mut_chain][2].append(ori_aa)
        mutation[mut_chain][3].append(icode)

    return mutation
    

def form_mut_chain(chain,  mutation_chain):

    mut_index = []
    for index in mutation_chain[0]:
        if index not in mut_index: mut_index.append(index)
            
    mut_aa = mutation_chain[1]
    ori_aa = mutation_chain[2]
    mut_icode = mutation_chain[3]

    sequence = chain['sequence']
    seq_index = chain['seq_index']
    mask = copy.deepcopy(chain['mask'])
    coords = chain['coords']

    if 'residues_icode' in chain:
        residues_icode = {}
        for pos in chain['residues_icode'] :
            residues_icode[int(pos)] = chain['residues_icode'][pos]

        for i, mut in enumerate(mut_index):

            if mut not in residues_icode: continue
            icode = mut_icode[i]

            if icode not in residues_icode[mut]: continue
            index = seq_index.index(mut)
            aa_dict = residues_icode[mut][icode]

            sequence = sequence[:index] + ori_aa[i] + sequence[index + 1 :]

            coords_new = aa_dict['coords']
            for atom in coords:
                coords[atom][index] = coords_new[atom]
    ori_sequence = sequence

    for i, mut in enumerate(mut_index):
        
        if mut not in seq_index: continue
        index = seq_index.index(mut)
        mask[index] += 2

        if sequence[index] == ori_aa[i]:
            # print('mut from', ori_aa[i], 'to', mut_aa[i], 'at', mut)
            sequence = sequence[:index] + mut_aa[i] + sequence[index + 1: ]
        else:
            print('sequence[index]:', sequence[index])
            print('ori_aa:', ori_aa[i])
            raise ValueError("not match the origin aa")

    return ori_sequence, mask, sequence, coords

def get_chains_dict(mutation_info, strct):

    cplx_muts = get_mut_pos(mutation_info)  

    pairs = []
    for chain in strct:
        pair = {}
        # pair['No'] = flag_new
        # pair['ddG'] = cplx['ddG']
        if chain in cplx_muts:
            pair['ori_seq'], pair['mask'], pair['mut_seq'], pair['coords'] = form_mut_chain(strct[chain], cplx_muts[chain])
        else:
            pair['ori_seq'], pair['mask'], pair['mut_seq'], pair['coords'] = strct[chain]['sequence'], strct[chain]['mask'], [], strct[chain]['coords']
        pairs.append(pair)

    return pairs


def completize(protein, esm = None):

    l_max = 0
    for chain in protein:
        l_max = max(l_max, len(chain['mask']))
    chain_num = len(protein)

    M = np.zeros([chain_num, l_max], dtype=np.float32)
    L = np.zeros([chain_num, l_max], dtype=np.float32)
    X = np.zeros([chain_num, l_max, 4, 3])
    if esm == None:
        oS = np.zeros([chain_num, l_max], dtype=np.float32)
        mS = np.zeros([chain_num, l_max], dtype=np.float32)
    else:
        oS = np.ones([chain_num, l_max + 2], dtype=np.int32)
        mS = np.ones([chain_num, l_max + 2], dtype=np.int32)
        oS[:,0], mS[:,0] = 0, 0

    for i, chain in enumerate(protein):
        N = len(chain['mask'])
        # print('l_max:', self.l_max)
        M[i,:N] = [1 if mask_val >0 else 0 for mask_val in chain['mask']]
        L[i,:N] = [1 if mask_val >1 else 0 for mask_val in chain['mask']]
        X[i,:N,:] = np.stack([chain['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        if esm == None:
            oS[i,:N] = [alphabet.index(a) for a in chain['ori_seq']]
            if chain['mut_seq'] == []: mS[i,:N] = copy.deepcopy(oS[i,:N])
            else: mS[i,:N] = [alphabet.index(a) if a in alphabet else alphabet.index('-') for a in chain['mut_seq']]
        else:
            alphabet_esm = '----LAGVSERTIDPKQNFYMHWCXBUZO'
            oS[i,1:N+1] = [alphabet_esm.index(a) for a in chain['ori_seq']]
            oS[i,N+1] = 2
            if chain['mut_seq'] == []: mS[i,1:N+2] = copy.deepcopy(oS[i,1:N+2])
            else: 
                mS[i,1:N+1] = [alphabet_esm.index(a) for a in chain['mut_seq']]
                mS[i,N+1] = 2
            # print('oS/mS:', oS.shape, mS.shape)

    X = torch.from_numpy(X).float().cuda()        # [C, L_max, 4, 3]
    oS = torch.from_numpy(oS).long().cuda()       # [C, L_max]
    mS = torch.from_numpy(mS).long().cuda()       # [C, L_max]
    L = torch.from_numpy(L).float().cuda()      # [C, L_max]
    M = torch.from_numpy(M).float().cuda()      # [C, L_max]

    return [X.unsqueeze(0), oS.unsqueeze(0), mS.unsqueeze(0), L.unsqueeze(0), M.unsqueeze(0)]
