# Repository for [A New Molecule Format for Transformer Models](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001000/v1).
## File List
- sdf.py Covert SDF file to new format and save data into PostgreSQL database.  The max features is 70.
- smiles_to_link.py Covert SMILES to the new format and save data back to postgreSQL database.  SMILES are already loaded in the database.
- map_1.csv : Mapping chemical bonds to token
- build.sql - SQL to create tables to store the molecule data, columns
  - x : one of 2D coordinates of the bond - the 1st atom number
  - y : another 2D coordinates of the bond - the 2nd atom number
  - bond : bond list represented by the numeric token
- train_ep.py: Using Transformer and 2D RoPE encoding to train LogP dataset
