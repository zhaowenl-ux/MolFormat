# Repository for [A New Molecule Format for Transformer Models](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001000/v1).
## File List
- sdf.py Covert SDF file to new format and save data into PostgreSQL database.  The max features is 70.
- smiles_to_link.py Covert SMILES to the new format and save data back to postgreSQL database.  SMILES are already loaded in the database.
- map_1.csv : Mapping chemical bonds to token
