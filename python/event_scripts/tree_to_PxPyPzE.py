import ROOT
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin', default='../../events/MG1/TTree/bbjj.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG1/PxPyPzE/', type=str, help='Path to output directory.')
parser.add_argument('-bs', '--bquarks', default=2, type=int, help='Number of b-quarks (typically 2, 3 or 4).')

args = parser.parse_args()

tree_file = ROOT.TFile(args.pathin,"READ")

tree = tree_file.Get("Tree")
tree.SetName("t")

lSR = []
lCR = []
lSB = []

lSR2bDer = []

vec2b = []
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())


for i in range(tree.GetEntries()):
    tree.GetEntry(i)
    
    temp = []
    
    vec2b[0].SetPtEtaPhiE(tree.jetPt[0], tree.jetEta[0], tree.jetPhi[0], tree.jetEnergy[0])
    vec2b[1].SetPtEtaPhiE(tree.jetPt[1], tree.jetEta[1], tree.jetPhi[1], tree.jetEnergy[1])
    vec2b[2].SetPtEtaPhiE(tree.jetPt[2], tree.jetEta[2], tree.jetPhi[2], tree.jetEnergy[2])
    vec2b[3].SetPtEtaPhiE(tree.jetPt[3], tree.jetEta[3], tree.jetPhi[3], tree.jetEnergy[3])

    for j in range(0, 4):
        temp.append(vec2b[j].Px())
        temp.append(vec2b[j].Py())
        temp.append(vec2b[j].Pz())
        temp.append(vec2b[j].E())  
    
    if tree.SR == 1:
        lSR.append(temp)

    if tree.CR == 1:
        lCR.append(temp)
        
    if tree.SB == 1:
        lSB.append(temp)        

    i += 1
        
SB = np.array(lSB)        
CR = np.array(lCR)
SR = np.array(lSR)

np.save(args.pathout + "/SB" + str(args.bquarks) + "b.npy", SB)
np.save(args.pathout + "/CR" + str(args.bquarks) + "b.npy", CR)
np.save(args.pathout + "/SR" + str(args.bquarks) + "b.npy", SR)


