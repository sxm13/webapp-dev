from __future__ import print_function, division
import os
import json
import functools
from torch.utils.data import Dataset,DataLoader

import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

import warnings
import numpy as np
import pymatgen.core as mg
from CifFile import ReadCif
from ase.io import read,write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import shutil

import importlib
import sys
source = importlib.import_module('')
sys.modules['model.utils'] = source
sys.modules['model'] = source
sys.modules['model4pre'] = source
sys.modules['source'] = source
sys.modules['source.utils'] = source


class Normalizer(object):
	def __init__(self, tensor):
		self.mean = torch.mean(tensor)
		self.std = torch.std(tensor)
	def norm(self, tensor):
		return (tensor - self.mean) / self.std
	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean
	def state_dict(self):
		return {'mean': self.mean,'std': self.std}
	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']

def mae(prediction, target):
	return torch.mean(torch.abs(target - prediction))

def sampling(csv_path):
    import csv
    with open(csv_path,'r') as f:
        reader = csv.reader(f)
        x = [row for row in reader]
    result = []
    for i in range(len(x)):
        temp = x[i]
        result.append(float(temp[1]))
    return torch.Tensor(result)

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self,val,n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state,is_best,chk_name,best_name):
	torch.save(state, chk_name)
	if is_best:
		shutil.copyfile(chk_name,best_name)
          
class ConvLayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len):
        super(ConvLayer,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.tanh_e = nn.Tanh()
        self.tanh_v = nn.Tanh()
        self.bn_v = nn.BatchNorm1d(self.atom_fea_len)
        self.phi_e = nn.Sequential(nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
			        nn.Linear(self.atom_fea_len,self.atom_fea_len))
        self.phi_v = nn.Sequential(nn.Linear(2*self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len))
    def forward(self,atom_in_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        N,M = atom_in_fea.shape
        atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1,:]
        atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2,:]
        nbr_num_fea = num_nbrs[nbr_fea_idx1].view(-1,1)
        total_nbr_fea = torch.cat([atom_nbr_fea1,atom_nbr_fea2,nbr_fea],dim=1)
        ek = self.phi_e(total_nbr_fea)
        rho_e_v = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M))).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        total_node_fea = torch.cat([atom_in_fea,rho_e_v],dim=1)
        vi = self.phi_v(total_node_fea)		
        vi = self.bn_v(vi)
        ek = nbr_fea + ek
        vi = atom_in_fea + vi
        ek_sum = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M)) ).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nbr_fea = torch.cat([vi,ek_sum],dim=1)
        global_fea = Variable(torch.zeros((Ncrys,2*M)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,2*M)) ).scatter_add(0,crystal_atom_idx.view(-1,1).repeat(1,2*M),atom_nbr_fea)
        return ek,vi,global_fea

class GCN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h):    
        super(GCN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len).to(device)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len).to(device)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len) for _ in range(n_conv)]).to(device)
        self.phi_u = nn.Sequential(nn.Linear(2*atom_fea_len,h_fea_len).to(device),nn.LeakyReLU(0.2).to(device),
				   nn.Linear(h_fea_len,h_fea_len).to(device),nn.Tanh().to(device))
        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len).to(device)
        self.conv_to_fc_lrelu = nn.LeakyReLU(0.2).to(device)
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len).to(device) for _ in range(n_h-1)])
            self.activations = nn.ModuleList([nn.LeakyReLU(0.2).to(device) for _ in range(n_h-1)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(h_fea_len).to(device) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len,1).to(device)
    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        z = self.Encoding(atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbrs, crystal_atom_idx)
        out = self.Regressor(z)
        return out
    def Encoding(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        nbr_fea_idx1 = nbr_fea_idx1.cuda() if torch.cuda.is_available() else nbr_fea_idx1
        nbr_fea_idx2 = nbr_fea_idx2.cuda() if torch.cuda.is_available() else nbr_fea_idx2
        num_nbrs = num_nbrs.cuda() if torch.cuda.is_available() else num_nbrs
        crystal_atom_idx = crystal_atom_idx.cuda() if torch.cuda.is_available() else crystal_atom_idx
        atom_fea = atom_fea.cuda() if torch.cuda.is_available() else atom_fea
        nbr_fea = nbr_fea.cuda() if torch.cuda.is_available() else nbr_fea 

        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,_ = atom_fea.shape
        
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nums_ = Variable(torch.ones((N,1)).cuda() if torch.cuda.is_available() else torch.ones((N,1)) )
        atom_nums = Variable(torch.zeros((Ncrys,1)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,1)) ).scatter_add(0,crystal_atom_idx.view(-1,1),atom_nums_)
        N,_ = atom_fea.shape
        for conv_func in self.convs:
            nbr_fea,atom_fea,global_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx)          
        global_fea = global_fea / atom_nums
        z = self.phi_u(global_fea)
        return z
    def Regressor(self,z):
        crys_fea = self.conv_to_fc_lrelu(self.conv_to_fc(z))
        if hasattr(self,'fcs') and hasattr(self,'activations'):
            for fc,activation,_ in zip(self.fcs,self.activations,self.bns):
                crys_fea = activation(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out

class ConvLayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len):
        super(ConvLayer,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.tanh_e = nn.Tanh()
        self.tanh_v = nn.Tanh()
        self.bn_v = nn.BatchNorm1d(self.atom_fea_len)
        self.phi_e = nn.Sequential(nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len))
        self.phi_v = nn.Sequential(nn.Linear(2*self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len))
    def forward(self,atom_in_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        N,M = atom_in_fea.shape
        atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1,:]
        atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2,:]
        nbr_num_fea = num_nbrs[nbr_fea_idx1].view(-1,1)
        total_nbr_fea = torch.cat([atom_nbr_fea1,atom_nbr_fea2,nbr_fea],dim=1)
        ek = self.phi_e(total_nbr_fea)
        rho_e_v = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M)) ).scatter_add(0, nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        total_node_fea = torch.cat([atom_in_fea,rho_e_v],dim=1)
        vi = self.phi_v(total_node_fea)		
        vi = self.bn_v(vi)
        ek = nbr_fea + ek
        vi = atom_in_fea + vi
        ek_sum = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M))).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nbr_fea = torch.cat([vi,ek_sum],dim=1) 
        global_fea = Variable(torch.zeros((Ncrys,2*M)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,2*M)) ).scatter_add(0,crystal_atom_idx.view(-1,1).repeat(1,2*M),atom_nbr_fea)
        return ek,vi,global_fea,atom_nbr_fea

class SemiFullGN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,n_feature):    
        super(SemiFullGN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len) for _ in range(n_conv)])
        self.feature_embedding = nn.Sequential(nn.Linear(n_feature,512))
        self.atom_nbr_fea_embedding = nn.Sequential(nn.Linear(2*atom_fea_len,128))
        self.phi_pos = nn.Sequential(nn.Linear(512+128,512),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(nn.Conv1d(64,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,1,kernel_size=4,stride=1,padding=0))
        self.cell_embedding = nn.Sequential(nn.Linear(9,128)) # remove
    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,atom_idx,structure_feature):
        nbr_fea_idx1 = nbr_fea_idx1.cuda() if torch.cuda.is_available() else nbr_fea_idx1
        nbr_fea_idx2 = nbr_fea_idx2.cuda() if torch.cuda.is_available() else nbr_fea_idx2
        num_nbrs = num_nbrs.cuda() if torch.cuda.is_available() else num_nbrs
        atom_idx = atom_idx.cuda() if torch.cuda.is_available() else atom_idx
        atom_fea = atom_fea.cuda() if torch.cuda.is_available() else atom_fea
        nbr_fea = nbr_fea.cuda() if torch.cuda.is_available() else nbr_fea 
        structure_feature = structure_feature.cuda() if torch.cuda.is_available() else structure_feature
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,_ = atom_fea.shape 
        for conv_func in self.convs:
            nbr_fea,atom_fea,_,atom_nbr_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,atom_idx)
        feature = structure_feature[atom_idx]
        feature = self.feature_embedding(feature)
        atom_nbr_fea = self.atom_nbr_fea_embedding(atom_nbr_fea)
        final_feature = torch.cat((atom_nbr_fea,feature),dim=-1)
        charge = self.phi_pos(final_feature)
        charge = charge.view(N,64,8)
        charge = self.conv(charge).squeeze()
        return charge


def get_data_loader(dataset,collate_fn,batch_size=64,num_workers=0,pin_memory=False):
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=collate_fn,pin_memory=pin_memory)
    return data_loader

def collate_pool(dataset_list):
    batch_atom_fea = [] 
    batch_nbr_fea =[]
    batch_nbr_fea_idx1 = []
    batch_nbr_fea_idx2 = []
    batch_num_nbr = []
    crystal_atom_idx = []
    batch_pos = []
    batch_dij_ = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr, dij_), (pos))\
        in enumerate(dataset_list):       
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea);batch_dij_.append(dij_)
        tt1 = np.array(nbr_fea_idx1)+base_idx
        tt2 = np.array(nbr_fea_idx2)+base_idx
        batch_nbr_fea_idx1.append(torch.LongTensor(tt1.tolist()))
        batch_nbr_fea_idx2.append(torch.LongTensor(tt2.tolist()))
        batch_num_nbr.append(num_nbr)
        crystal_atom_idx.append(torch.LongTensor([i]*n_i))
        batch_pos.append(pos)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx1, dim=0),torch.cat(batch_nbr_fea_idx2, dim=0),
            torch.cat(batch_num_nbr, dim=0),torch.cat(crystal_atom_idx,dim=0), torch.cat(batch_dij_,dim=0),
            torch.cat(batch_pos,dim=0))

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]
    
class AtomCustomJSONInitializer(AtomInitializer):
		def __init__(self, elem_embedding_file):
				elem_embedding = json.load(open(elem_embedding_file))
				elem_embedding = {int(key): value for key, value in elem_embedding.items()}
				atom_types = set(elem_embedding.keys())
				super(AtomCustomJSONInitializer, self).__init__(atom_types)
				for key in range(101):
						zz = np.zeros((101,))
						zz[key] = 1.0
						self._embedding[key] = zz.reshape(1,-1)
    
class CIFData(Dataset):
    def __init__(self,crystal_data,pos,radius=6,dmin=0,step=0.2):
        self.crystal_data = crystal_data
        self.pos = pos
        self.radius = radius
        atom_init_file = os.path.join('./model4pre/' + 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return 1
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self,_):
        cif_id = self.mof.split('.cif')[0]
        with open(os.path.join(cif_id+'.json')) as f:
            crystal_data = json.load(f)
        nums = crystal_data['numbers']
        atom_fea = np.vstack([self.ari.get_atom_fea(nn) for nn in nums])
        pos = np.load(self.pos+cif_id+'_pos.npy')
        index1 = np.array(crystal_data['index1'])
        nbr_fea_idx = np.array(crystal_data['index2'])
        dij = np.array(crystal_data['dij']); dij_ = torch.Tensor(dij)
        nbr_fea = self.gdf.expand(dij)
        num_nbr = np.array(crystal_data['nn_num'])
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx1 = torch.LongTensor(index1)
        nbr_fea_idx2 = torch.LongTensor(nbr_fea_idx)
        num_nbr = torch.Tensor(num_nbr)
        pos = torch.Tensor(pos)

        return (atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr,dij_), (pos),cif_id
    

class CIFData(Dataset):
    def __init__(self,crystal_data,pos,radius,dmin,step):
        self.pos = pos
        self.radius = radius
        self.crystal_data = crystal_data
        atom_init_file = os.path.join('./PACMAN/' + 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return 1
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self,_):
        nums = self.crystal_data['numbers']
        atom_fea = np.vstack([self.ari.get_atom_fea(nn) for nn in nums])
        pos = self.pos
        index1 = np.array(self.crystal_data['index1'])
        nbr_fea_idx = np.array(self.crystal_data['index2'])
        dij = np.array(self.crystal_data['dij']); dij_ = torch.Tensor(dij)
        nbr_fea = self.gdf.expand(dij)
        num_nbr = np.array(self.crystal_data['nn_num'])
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx1 = torch.LongTensor(index1)
        nbr_fea_idx2 = torch.LongTensor(nbr_fea_idx)
        num_nbr = torch.Tensor(num_nbr)
        pos = torch.Tensor(pos)
       
        return (atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr,dij_), (pos)


def ensure_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if not lines[1].strip().startswith('data_'):
        lines.insert(1, 'data_struc\n')
        with open(file_path, 'w') as file:
            file.writelines(lines)

def n_atom(mof):
    try:
        structure = read(mof)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(mof, occupancy_tolerance=10)
        struct.get_structures()
    elements = [str(site.specie) for site in structure.sites]
    return len(elements)

def ase_format(mof):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # mof_temp = Structure.from_file(mof,primitive=True)
            mof_temp = Structure.from_file(mof)
            mof_temp.to(filename=mof, fmt="cif")
            struc = read(mof)
            write(mof, struc)
            # print('Reading by ase: ' + mof)
    except:
        try:
            struc = read(mof)
            write(mof, struc)
            print('Reading by ase: ' + mof)
        except:
            ensure_data(mof)

periodic_table_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]

def CIF2json(mof):
    try:
        structure = read(mof)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(mof, occupancy_tolerance=10)
        struct.get_structures()
    _c_index, _n_index, _, n_distance = struct.get_neighbor_list(r=6, numerical_tol=0, exclude_self=True)
    _nonmax_idx = []
    for i in range(len(structure)):
        idx_i = (_c_index == i).nonzero()[0]
        idx_sorted = np.argsort(n_distance[idx_i])[: 200]
        _nonmax_idx.append(idx_i[idx_sorted])
    _nonmax_idx = np.concatenate(_nonmax_idx)
    index1 = _c_index[_nonmax_idx]
    index2 = _n_index[_nonmax_idx]
    dij = n_distance[_nonmax_idx]
    numbers = []
    elements = [str(site.specie) for site in struct.sites]
    for i in range(len(elements)):
        ele = elements[i]
        atom_index = periodic_table_symbols.index(ele)
        numbers.append(int(int(atom_index)+1))
    nn_num = []
    for i in range(len(structure)):
        j = 0
        for idx in range(len(index1)):
            if index1[idx] == i:
                    j += 1
            else:
                    pass
        nn_num.append(j)
    data = {"rcut": 6.0,
            "numbers": numbers,
            "index1": index1.tolist(),
            "index2":index2.tolist(),
            "dij": dij.tolist(),
            "nn_num": nn_num}
    return data

def pre4pre(mof):
    try:
        try:
            structure = mg.Structure.from_file(mof)
        except:
            try:
                atoms = read(mof)
                structure = AseAtomsAdaptor.get_structure(atoms)
            except:
                structure = CifParser(mof, occupancy_tolerance=10)
                structure.get_structures()
        coords = structure.frac_coords
        try:
            elements = [str(site.specie) for site in structure.sites]
        except:
            elements = [str(site.species) for site in structure.sites]
        pos = []
        for i in range(len(elements)):
            x = coords[i][0]
            y = coords[i][1]
            z = coords[i][2]
            pos.append([float(x),float(y),float(z)])
    except Exception as e:
        pass
    return pos

def average_and_replace(numbers, di):
    groups = defaultdict(list)
    for i, number in enumerate(numbers):
        if di ==3:
            key = format(number, '.3f')
            groups[key].append(i)
        elif di ==2:
            key = format(number, '.2f')
            groups[key].append(i)
        elif di ==1:
            key = format(number, '.1f')
            groups[key].append(i)
        elif di ==0:
            key = format(number, '.0f')
            groups[key].append(i)
    for key, indices in groups.items():
        avg = sum(numbers[i] for i in indices) / len(indices)
        for i in indices:
            numbers[i] = avg
    return numbers

def write4cif(name, chg, digits, atom_type_option, neutral_option, charge_name, connect_option):
    new_content = []
    dia = int(digits)
    if atom_type_option:
        gcn_charge = chg.numpy()
        sum_chg = sum(gcn_charge)
        if neutral_option:
            charge = average_and_replace(gcn_charge, di=3)
            sum_chg = sum(charge)
            charges_1 = []
            for c in charge:
                cc = c - sum_chg / len(charge)
                charges_1.append(round(cc, dia))
            charge_2 = average_and_replace(charges_1, di=2)
            sum_chg = sum(charge_2)
            charges = []
            for c in charge_2:
                cc = c - sum_chg / len(charge_2)
                charges.append(round(cc, dia))
        else:
            charge = average_and_replace(gcn_charge, di=3)
            charges_1 = []
            for c in charge:
                charges_1.append(round(c, dia))
            charge_2 = average_and_replace(charges_1, di=2)
            charges = []
            for c in charge_2:
                charges.append(round(c, dia))
        net_charge = sum(charges)
        structure = read(name + ".cif")
        struct = AseAtomsAdaptor.get_structure(structure)
        atoms = [str(site.specie) for site in struct.sites]
        unique_counts = {}
        for char, index in zip(atoms, charges):
            if char not in unique_counts:
                unique_counts[char] = set()
            unique_counts[char].add(index)
        result = {char: len(indices) for char, indices in unique_counts.items()}
        atom_type = result
    else:
        gcn_charge = chg.numpy()
        sum_chg = sum(gcn_charge)
        if neutral_option:
            charges = [round(c - sum_chg / len(gcn_charge), dia) for c in gcn_charge]
        else:
            charges = []
            for c in gcn_charge:
                charges.append(round(c, dia))
        net_charge = sum(charges)
        atom_type = "Failure to check like atoms"


    if connect_option:
        mof = ReadCif(name + ".cif")
        mof.first_block().AddToLoop("_atom_site_type_symbol",{'_atom_site_charge':[str(q) for q in charges]})
       
        return ("# " + charge_name + " charges by PACMAN v1.3 (https://github.com/mtap-research/PACMAN-charge/)\n" +
                    f"data_{name.split('/')[-1]}" + str(mof.first_block())), atom_type, net_charge
            

    else:
        with open(name + ".cif", 'r') as file:
            lines = file.readlines()
        
        new_content.append("# " + charge_name + " charges by PACMAN v1.3 (https://github.com/mtap-research/PACMAN-charge) \n")
        charge_inserted = False
        charge_index = 0
        
        for line in lines:
            line = line.replace('_space_group_name_H-M_alt', '_symmetry_space_group_name_H-M')
            line = line.replace('_space_group_IT_number', '_symmetry_Int_Tables_number')
            line = line.replace('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
            
            if '_atom_site_occupancy' in line and not charge_inserted:
                new_content.append(line)
                new_content.append("  _atom_site_charge\n")
                charge_inserted = True
            elif charge_inserted and charge_index < len(charges):
                parts = line.strip().split()
                formatted_parts = []
                for part in parts[:-1]:
                    try:
                        formatted_parts.append(format(float(part), ".8f"))
                    except ValueError:
                        formatted_parts.append(part)
                formatted_parts.append(parts[-1])
                new_content.append(" ".join(formatted_parts) + " " + format(charges[charge_index], f".{dia}f") + "\n")
                charge_index += 1
            else:
                new_content.append(line)
        
        return ''.join(new_content), atom_type, net_charge


def predict_with_model(charge_name, file, name, digits, atom_type_option, neutral_option, connect_option):

    result = None
    atom_type_count = None
    net_charge = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if charge_name=="DDEC6":
        charge_model_name = "./PACMAN/pth/best_ddec/ddec.pth"
        nor_name = "./PACMAN/pth/best_ddec/normalizer-ddec.pkl"
    elif charge_name=="Bader":
        charge_model_name = "./PACMAN/pth/best_bader/bader.pth"
        nor_name = ".PACMAN//pth/best_bader/normalizer-bader.pkl"
    elif charge_name=="CM5":
        charge_model_name = "./PACMAN/pth/best_cm5/cm5.pth"
        nor_name = "./PACMAN/pth/best_cm5/normalizer-cm5.pkl"
    elif charge_name=="REPEAT":
        charge_model_name = "./PACMAN/pth/best_repeat/repeat.pth"
        nor_name = "./PACMAN/pth/best_repeat/normalizer-repeat.pkl"
    with open(nor_name, 'rb') as f:
        charge_nor = pickle.load(f)
    f.close()
    if connect_option:
        pass
    else:
        ase_format(file)
    data = CIF2json(file)
    pos = pre4pre(file)
    batch_size = 1
    num_workers = 0
    pin_memory = False
    pre_dataset = CIFData(data,pos,6,0,0.2)
    collate_fn = collate_pool
    pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
    for batch in pre_loader:
        chg_1 = batch[0].shape[-1]+3
        chg_2 = batch[1].shape[-1]
    gcn = GCN(chg_1-3, chg_2, 128, 7, 256,5) 
    chkpt = torch.load(charge_model_name, map_location=torch.device(device))
    model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
    model4chg.to(device)
    model4chg.load_state_dict(chkpt['state_dict'])
    model4chg.eval()
    for _, (input) in enumerate(pre_loader):
        with torch.no_grad():
            input_var = (input[0].to(device),
                        input[1].to(device),
                        input[2].to(device),
                        input[3].to(device),
                        input[4].to(device),
                        input[5].to(device))
            encoder_feature = gcn.Encoding(*input_var)
            atoms_fea = torch.cat((input[0],input[7]),dim=-1)
            input_var2 = (atoms_fea.to(device),
                        input[1].to(device),
                        input[2].to(device),
                        input[3].to(device),
                        input[4].to(device),
                        input[5].to(device),
                        encoder_feature.to(device))
            chg = model4chg(*input_var2)
            charge_pre = charge_nor.denorm(chg.data.cpu())
            result,atom_type_count,net_charge = write4cif(name, charge_pre, digits, atom_type_option, neutral_option, charge_name, connect_option)
    return result,atom_type_count,net_charge 
