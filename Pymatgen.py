from pymatgen import MPRester
from pymatgen.io.cif import CifParser
from pprint import pprint
from pandas import DataFrame


MAPI_KEY =  "Ql4SEMZykVwuqeF7" #My API Key

#defining the material object
m= MPRester(MAPI_KEY)

# Get the data of all compounds containing Li, Mn, Ni and O.
data = m.query(criteria={"elements": {"$all": ["Li", "Mn", "Ni", "O"]}, "nelements": 4},
               properties=["material_id", "pretty_formula","unit_cell_formula", "formation_energy_per_atom", "crystal_system", "spacegroup.symbol", "nsites","volume", "density", "band_gap", "e_above_hull"])



df = DataFrame(data, columns=["material_id","pretty_formula","unit_cell_formula", "formation_energy_per_atom", "crystal_system", "spacegroup.symbol", "nsites","volume", "density", "band_gap", "e_above_hull"])

#save data in csv file
print (df.to_csv('LMNO.csv', index=False))



