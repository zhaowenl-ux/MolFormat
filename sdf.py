import os
import re
import numpy as np
import psycopg2 
import pandas as pd
import dotenv
dotenv.load_dotenv()

IN = "ks_compound.sdf"
OUT = "output1.txt"
OUT3 = "output3.txt"
MAP="map_1.csv"

host=os.getenv("DB_HOST"),
database=os.getenv("DB_NAME"),
user=os.getenv("DB_USER"),
password=os.getenv("DB_PASSWORD"),
port=os.getenv("DB_PORT", "5432")

connection_params = {
'dbname': database,
'user': user,
'password': password,
'host': host,
'port': port
}

TABLE_NAME = "chemistry.compound_70_sorted"
feature_dim = 70
connection = psycopg2.connect(**connection_params)
cursor = connection.cursor()
insert_query = "INSERT INTO " + TABLE_NAME + " (COMPOUND_ID, x, y, bond) VALUES (%s, %s, %s, %s);"
def read_map():
    df = pd.read_csv(MAP, header=None)    
    return df.to_numpy()

bond_map = read_map()

def map_atom(atom1, atom2, bond_type):
    for row in bond_map:
        if ((row[0] == atom1.upper() and row[1] == atom2.upper()) or (row[0] == atom2.upper() and row[1] == atom1.upper())) and row[2] == bond_type:
            return row[3]
    return -1

# DEFINE THE KEY PATTERNS
p_end = re.compile('M + END')

with open(IN, 'r') as file:
    lines = file.readlines()

f = open(OUT,"w")
f3 = open(OUT3,"w")
START = True
empty_array = np.zeros((3, 70))
num_atoms = 0
num_bonds = 0
atom_map = dict()
bonds = []
bond3 = []
pos = 0
ID_LINE = False
rec = 0
NO_READ = False
for line in lines:
    #print(line.strip())
    
    if line.find('$$$$') >= 0:
        #print('find end of molecule')
        #x = [0] * feature_dim
        #y = [0] * feature_dim
        #z = [0] * feature_dim
        x = []
        y = []
        z = []
        index = 0
        bond3.sort(key = lambda item : (item[0], item[1]) )
        for bond in bond3:
            atom1 = bond[3]
            atom2 = bond[4]
            bond_type = bond[2]
            mapped_value = map_atom(atom1, atom2, int(bond_type))
            #x[index] = bond[0]
            #y[index] = bond[1]
            #z[index] = mapped_value
            x.append(bond[0])
            y.append(bond[1])
            z.append(mapped_value)
            index += 1
            if index >= feature_dim:
                break
            #print(f'{atom1},{atom2},{bond_type},{mapped_value}')
            #f.write(f'{atom1},{atom2},{bond_type},{mapped_value}\n')
            #f3.write(f'{atom1},{atom2},{bond_type},{atom_map[atom1-1]},{atom_map[atom2-1]},{mapped_value}\n')
        #f.write(' '.join(map(str, empty_array.flatten())) + '\n')
        #f.write('\n'.join([str(','.join(map(str, bond))) for bond in bonds]))
        #f3.write('\n'.join([str(','.join(map(str, bond))) for bond in bond3]) + '\n')
        #f.write('\nMOL END\n')
        #f3.write('MOL END\n')
        vx = ",".join(map(str, x))
        vy = ",".join(map(str, y))
        vz = ",".join(map(str, z))
        if num_atoms <= 100 and not NO_READ:
            cursor.execute(insert_query, (compound_id, vx, vy, vz))
            rec += 1
            if rec % 1000 == 0:
                connection.commit()
                print(f'Inserted {rec} records')
        bond3 = []
        bonds = []
        #OUT.write('$$$$\n')
        #OUT.flush()
        pos = -1; # new molecule
        #START = False
    elif pos == 3 : # count line read number of atoms and bonds
        print('count line', line[0:3])
        try:
            num_atoms = int(line[0:3])
            num_bonds = int(line[3:6])
        except ValueError:
            num_atoms = 0
            num_bonds = 0
            NO_READ = True
        print(f'num_atoms: {num_atoms}, num_bonds: {num_bonds}')
    elif pos > 3 and pos < 4 + num_atoms and not NO_READ:
        atom_map[pos - 4] = line[31:33].strip() 
    elif pos >= 4 + num_atoms and pos < 4 + num_atoms + num_bonds and not NO_READ:
        #print('bond line', line.strip())
        atom1 = int(line[0:3].strip())
        atom2 = int(line[3:6].strip())
        bond_type = line[6:9].strip()
        bonds.append((atom1, atom2, bond_type))
        bond3.append((atom1, atom2, bond_type,atom_map[atom1-1], atom_map[atom2-1]))
        #f3.write(','.join(map(str, (atom1, atom2, bond_type))
        #print((atom1, atom2, bond_type))
    elif pos >= 4 + num_atoms + num_bonds and ID_LINE == False:
        if line.find('COMPOUND_ID') >= 0:
            ID_LINE = True
            #print('found COMPOUND_ID line' + str(pos))
    elif ID_LINE :
        #print('reading compound id line: ' + str(pos))
        compound_id = line.strip()
        ID_LINE = False
        print(f'compound_id: {compound_id}')
    pos += 1
    
    connection.commit() # commit remaining records
    print(f'Inserted total {rec} records')