### Scale of upsampled 4b data
pi_4L = 0.0998996846167015


### Signal fractions
### Taken from Table 8.1 2016 HH data at https://cds.cern.ch/record/2644551?ln=en
SB_sig_fraction = 1.1/10420 
CR_sig_fraction = 1.8/7553
SR_sig_fraction = 3.8/7134

### Signal scaling
pi_S = 100

### IDs of main methods.
method_ids = {
        "fvt"       : "HH_FvT__cl_np799_l0_01_e10",
        "comb"      : "HH_Comb_FvT__pl_emd_p1_R0_4__cl_np799_l0_01_e10",
        "ot1"       : "HH_OT__pl_emd_p1_R0_4__K_1",
        "ot5"       : "HH_OT__pl_emd_p1_R0_4__K_5",
        "ot10"      : "HH_OT__pl_emd_p1_R0_4__K_10",
        "ot10"      : "HH_OT__pl_emd_p1_R0_4__K_10",
        "ot20"      : "HH_OT__pl_emd_p1_R0_4__K_20",
        "benchmark" : "benchmark",
        }

### Legends 
method_legends = {
        "fvt"      : "FvT Model",
        "comb"     : "OT-FvT Model",
        "ot1"      : "OT-1NN Model ",
        "ot5"      : "OT-5NN Model ",
        "ot10"     : "OT-10NN Model",
        "ot20"     : "OT-20NN Model",
        "benchmark": "Scaled Three-Tag Data",
        "signal"   : "SM HH #times " + str(int(pi_S)),
        }

### Selection constants
minPt  = 40
maxEta = 2.5

### Region Definitions
leadHH, sublHH = 125.0, 125.0

xHHSR = 1.6
rHHCR = 30
sHHCR = 1.03
rHHSB = 45
sHHSB = 1.05

