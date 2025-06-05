# Preprocessing constants
REGIONS = [(0, 500), (1500, 4000)]
INTERPOLATION_N_POINTS = 600
SMOOTHING_WINDOW_LENGTH = 5
SMOOTHING_POLYORDER = 2
BASELINE_SMOOTHNESS = 1e6
BASELINE_ASMMETRY = 0.01
BASELINE_MAX_ITERATIONS = 10
FUNCTIONAL_GROUP_SMARTS = {
    #'alkane': '[CX4;H0,H1,H2,H4]',
    #'methyl': '[CH3]',
    #'alkene': '[CX3]=[CX3]',
    #'alkyne': '[CX2]#C',
    'alcohols': '[#6][OX2H]',
    #'amines': '[NX3;H2,H1;!$(NC=O)]',
    #'nitriles': '[NX1]#[CX2]',
    'aromatics': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    #'alkyl halides': '[#6][F,Cl,Br,I]',
    'esters': '[#6][CX3](=O)[OX2H0][#6]',
    #'ketones': '[#6][CX3](=O)[#6]',
    'aldehydes': '[CX3H1](=O)[#6]',
    'carboxylic acids': '[CX3](=O)[OX2H1]',
    #'ether': '[OD2]([#6;!$(C=O)])([#6;!$(C=O)])',
    #'ether': '[OD2]([#6])([#6])',
    #'acyl halides': '[CX3](=[OX1])[F,Cl,Br,I]',
    #'amides': '[NX3][CX3](=[OX1])[#6]',
    #'amides': '[NX3][CX3](=[OX1])',
    #'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
}