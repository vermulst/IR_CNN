import orjson
from data_handling.SpectraSample import SpectraSample
from data_handling.func_group_identif import FunctionalGroupIdentifier

func_grp_smarts = {
    'alkane': '[CX4;H0,H1,H2,H4]',
    'methyl': '[CH3]',
    'alkene': '[CX3]=[CX3]',
    'alkyne': '[CX2]#C',
    'alcohols': '[#6][OX2H]',
    'amines': '[NX3;H2,H1;!$(NC=O)]',
    'nitriles': '[NX1]#[CX2]',
    'aromatics': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    'alkyl halides': '[#6][F,Cl,Br,I]',
    'esters': '[#6][CX3](=O)[OX2H0][#6]',
    'ketones': '[#6][CX3](=O)[#6]',
    'aldehydes': '[CX3H1](=O)[#6]',
    'carboxylic acids': '[CX3](=O)[OX2H1]',
    'ether': '[OD2]([#6;!$(C=O)])([#6;!$(C=O)])',
    # 'ether': '[OD2]([#6])([#6])',
    'acyl halides': '[CX3](=[OX1])[F,Cl,Br,I]',
    'amides': '[NX3][CX3](=[OX1])[#6]',
    # 'amides': '[NX3][CX3](=[OX1])'
    'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
}
func_group_id = FunctionalGroupIdentifier(func_grp_smarts)


# load a sample from a file name (JCAMP file)
def load_sample(datafolder, identifier):
    path = f"{datafolder}/samples/{identifier}"
    try:
        return SpectraSample(path)
    except Exception as e:
        print(f"Error in {identifier}: {e}")
        return None
    
def load_samples(datafolder):
    """Load spectral samples with functional group labels, optimized version"""
    with open(f"{datafolder}/meta_data.json", "rb") as f:
        compounds = orjson.loads(f.read())

    samples = []
    valid_count = 0
    
    for compound in compounds:
        if not (smiles := compound.get("cano_smiles")):
            continue  # skip compounds without SMILES early
            
        # Process attachments in compound scope to preserve SMILES context
        attachments = (
            attachment
            for dataset in compound.get("datasets", [])
            for attachment in dataset.get("attacments", [])
        )
        
        for attachment in attachments:
            try:
                identifier = attachment["identifier"].split("/", 1)[1]
                if (sample := load_sample(datafolder, identifier)) and \
                   (label_vec := func_group_id.encode(smiles)):
                    sample.labels = label_vec
                    samples.append(sample)
                    valid_count += 1
            except Exception as e:
                print(f"Skipping {identifier}: {str(e)}")
                continue

    print(f"Loaded {valid_count}/{len(samples)} valid samples with labels.")
    return samples

def load_first_sample(datafolder):
    # Load the JSON metadata
    with open(f"{datafolder}/meta_data.json", "rb") as f:
        compounds = orjson.loads(f.read())
    
    # Find the first available filename
    for compound in compounds:
        for dataset in compound.get("datasets", []):
            for attachment in dataset.get("attacments", []):
                try:
                    identifier = attachment["identifier"].split("/", 1)[1]
                    path = f"{datafolder}/samples/{identifier}"
                    return SpectraSample(path)
                except Exception as e:
                    print(f"Error loading sample {identifier}: {e}")
                    continue
    
    return None  # Return None if no samples were found