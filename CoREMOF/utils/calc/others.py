from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from pymatgen.core.structure import Structure

def SpaceGroup(struc):
    result_sg = {}
    result_sg["unit"]="nan"
    atoms = read(struc, format="cif")
    structure = AseAtomsAdaptor.get_structure(atoms)
    result_ = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
    hall_symbol = result_.get_hall()
    space_group_number = result_.get_space_group_number()
    crystal_system = result_.get_crystal_system()
    result_sg["hall_symbol"]=hall_symbol
    result_sg["space_group_number"]=space_group_number
    result_sg["crystal_system"]=crystal_system

    return result_sg

def Mass(struc):
    result_m = {}
    result_m["unit"]="amu"
    atoms = read(struc, format="cif")
    total_mass = atoms.get_masses().sum()
    result_m["total_mass"]=float(total_mass)

    return result_m

def Volume(struc):
    result_v = {}
    result_v["unit"]="Å^3"
    atoms = Structure.from_str(struc, fmt="cif")
    total_volume = atoms.volume
    result_v["total_volume"]=total_volume

    return result_v

def n_atom(struc):
    result_na = {}
    result_na["unit"]="nan"
    atoms = read(struc, format="cif")
    number_atoms = len(atoms)
    result_na["number_atoms"]=number_atoms

    return result_na

