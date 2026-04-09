import pandas as pd
import psycopg2 
import rdkit
from rdkit import Chem
import dotenv
import os   

dotenv.load_dotenv()
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

TABLE = "chemistry.logp"
MAP="map_1.csv"
FEATURE_DIM = 70
SQL_SELECT = "SELECT id, logp, smiles FROM " + TABLE + ";"
SQL_UPDATE = "UPDATE  " + TABLE + " set x = %s, y = %s, bond = %s, num_atoms=%s where id= %s"

def read_map():
    df = pd.read_csv(MAP, header=None)    
    return df.to_numpy()

bond_map = read_map()
def map_atom(atom1, atom2, bond_type):
    for row in bond_map:
        if ((row[0] == atom1.upper() and row[1] == atom2.upper()) or (row[0] == atom2.upper() and row[1] == atom1.upper())) and row[2] == bond_type:
            return row[3]
    return 100  # default value if not found

connection = psycopg2.connect(**connection_params)
cursor = connection.cursor()

cursor.execute(SQL_SELECT)
rec = 0
for row in cursor.fetchall():
    print(row)
    mol = Chem.MolFromSmiles(row[2])
    if mol is None:
        print(f"Invalid SMILES for ID {row[0]}: {row[2]}")
        continue
    id = row[0]
    x = [0] * FEATURE_DIM
    y = [0] * FEATURE_DIM
    z = [0] * FEATURE_DIM
    index = 0
    num_atoms = mol.GetNumHeavyAtoms()
    for b in mol.GetBonds():
        a1 = b.GetBeginAtom()
        atom1= a1.GetIdx()
        a2 = b.GetEndAtom()
        atom2= a2.GetIdx()
        bond_type = int(b.GetBondTypeAsDouble())
        if b.GetIsAromatic() and bond_type == 2:
            bond_type = 4  # Aromatic bond
        mapped_value = map_atom(a1.GetSymbol(), a2.GetSymbol(), int(bond_type))
        print(f"Bond between {a1.GetSymbol()} and {a2.GetSymbol()} of type {b.GetBondType()}")  
        x[index] = atom1 + 1
        y[index] = atom2
        z[index] = mapped_value
        index += 1
        if index >= FEATURE_DIM:
            break
    vx = ",".join(map(str, x))
    vy = ",".join(map(str, y))
    vz = ",".join(map(str, z))
    
    if num_atoms <= 100:
            cursor.execute(SQL_UPDATE, (vx, vy, vz,num_atoms, id))
            #print(f"Updating ID {id} with x: {vx}, y: {vy}, bond: {vz}")
            rec += 1
            if rec % 1000 == 0:
                connection.commit()
                print(f'Inserted {rec} records')    
    
connection