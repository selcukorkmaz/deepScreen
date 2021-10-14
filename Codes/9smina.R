### TERMINAL #####
#
cd deepDrug/deepScreen/imageData/pubchem
awk '{ print $2 ,$1 "_active " }' smiles/actives.txt | \
sed 's/\. Cl//' > smiles/actives.smi

awk '{ print $2 ,$1 "_decoys " }' smiles/decoys.txt | \
sed 's/\. Cl//' > smiles/decoys.smi


wget http://bits.csb.pitt.edu/tdtCDPK1/rdconf.py
chmod +x rdconf.py

## install conda
## install rdkit

conda activate py37_rdkit_beta
cd /tmp/rdkit/rdkit/Chem  # <- replace this with the real name of the directory
export RDBASE=`pwd`
export PYTHONPATH="$RDBASE"

python3 rdconf.py --maxconfs 1 smiles/decoys.smi sdf/decoys.sdf
python3 rdconf.py --maxconfs 1 smiles/actives.smi sdf/actives.sdf

cat sdf/actives.sdf sdf/decoys.sdf > sdf/combined.sdf


obabel sdf/combined.sdf -xr -O sdf/combined.pdb

obabel receptor/1bpy.pdb -xr -O receptor/1bpy.pdbqt

conda install -c bioconda smina

#####################

setwd("deepDrug/deepScreen/")


files = list.files("imageData/pubchem/ligands/")

for(i in 1:length(files)){

  f = paste0(gsub('.pdb','',files[i]),".pdbqt")
  f2 = gsub('.pdb','',files[i])

  obabel = paste0("obabel imageData/pubchem/ligands/",files[i], " -O imageData/pubchem/pdbqt/",f)

  system(obabel)

  old_path <- Sys.getenv("PATH")
  Sys.setenv(PATH = paste(old_path, "/opt/anaconda3/bin/", sep = ":"))

  smina = paste0("smina -r imageData/pubchem/receptor/1bpy.pdbqt -l imageData/pubchem/pdbqt/",f, " --seed 0 --autobox_ligand imageData/pubchem/pdbqt/",f," --autobox_add 12 --exhaustiveness 40 -o imageData/pubchem/redock/1bpy-redock_",f)

  system(smina)

  pymol = paste0("pymol -c imageData/pubchem/receptor/1bpy.pdb imageData/pubchem/redock/1bpy-redock_",f," -d 'remove (hydro);remove resn hoh;remove solvent;' -g imageData/pubchem/pose/",f2,".png")

  system(pymol)

  print(i)
}



