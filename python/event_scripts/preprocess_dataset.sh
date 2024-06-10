#!/bin/bash

# You should be running this script from toy4b/python/event_scripts. 

DATA=MG3   # Modify this to the nickname of your dataset.


TTREE_PATH=../../events/$DATA/TTree   # This script assumes this path exists already.
DF_PATH=../../events/$DATA/dataframes
PT_PATH=../../events/$DATA/PtEtaPhi
PTM_PATH=../../events/$DATA/PtEtaPhiM

# mkdir $DF_PATH
# mkdir $PT_PATH
# mkdir $PTM_PATH

# # Split ROOT files by region.
# python tree_regions.py -pi $TTREE_PATH/bbbj.root -po $TTREE_PATH/bbbj_
# python tree_regions.py -pi $TTREE_PATH/bbbb_large.root -po $TTREE_PATH/bbbb_large
# python tree_regions.py -pi $TTREE_PATH/HH4b.root -po $TTREE_PATH/HH4b_

# # ROOT --> PtEtaPhi.
# python tree_to_PtEtaPhi.py -pi $TTREE_PATH/bbbb_large.root -po $PT_PATH/ -bs 4
# python tree_to_PtEtaPhi.py -pi $TTREE_PATH/bbbj.root -po $PT_PATH/ -bs 3 

# ROOT --> h5.
# python tree_to_df.py -pi $TTREE_PATH/bbbb.root -po $DF_PATH/bbbb.h5 -f True
# python tree_to_df.py -pi $TTREE_PATH/bbbb_large.root -po $DF_PATH/bbbb_large.h5 -f True
# python tree_to_df.py -pi $TTREE_PATH/HH4b.root -po $DF_PATH/HH4b.h5 -f True
# python tree_to_df.py -pi $TTREE_PATH/bbbj.root -po $DF_PATH/bbbj.h5 

# python tree_to_PtEtaPhiM.py -pi $TTREE_PATH/bbbb_large.root -po $PTM_PATH/ -bs 4
# python tree_to_PtEtaPhiM.py -pi $TTREE_PATH/bbbj.root -po $PTM_PATH/ -bs 3

# # ROOT --> h5.
python tree_to_df.py -pi $TTREE_PATH/HH4b_picoAOD.root -po $DF_PATH/HH4b_picoAOD.h5 -f True
python tree_to_df.py -pi $TTREE_PATH/fourTag_10x_picoAOD.root -po $DF_PATH/fourTag_10x_picoAOD.h5 -f True
python tree_to_df.py -pi $TTREE_PATH/fourTag_picoAOD.root -po $DF_PATH/fourTag_picoAOD.h5 -f True
python tree_to_df.py -pi $TTREE_PATH/threeTag_picoAOD.root -po $DF_PATH/threeTag_picoAOD.h5 -f False