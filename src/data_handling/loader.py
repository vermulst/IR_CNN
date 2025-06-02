import orjson
from data_handling.SpectraSample import SpectraSample
from data_handling.func_group_identif import FunctionalGroupIdentifier
from tqdm import tqdm
from rich import print as rprint

func_grp_smarts = {
    #'alkane': '[CX4;H0,H1,H2,H4]',
    #'methyl': '[CH3]',
    #'alkene': '[CX3]=[CX3]',
    #'alkyne': '[CX2]#C',
    'alcohols': '[#6][OX2H]',
    #'amines': '[NX3;H2,H1;!$(NC=O)]',
    #'nitriles': '[NX1]#[CX2]',
    'aromatics': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    #'alkyl halides': '[#6][F,Cl,Br,I]',
    #'esters': '[#6][CX3](=O)[OX2H0][#6]',
    #'ketones': '[#6][CX3](=O)[#6]',
    'aldehydes': '[CX3H1](=O)[#6]',
    #'carboxylic acids': '[CX3](=O)[OX2H1]',
    #'ether': '[OD2]([#6;!$(C=O)])([#6;!$(C=O)])',
    #'ether': '[OD2]([#6])([#6])',
    #'acyl halides': '[CX3](=[OX1])[F,Cl,Br,I]',
    #'amides': '[NX3][CX3](=[OX1])[#6]',
    #'amides': '[NX3][CX3](=[OX1])',
    #'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
}
func_group_id = FunctionalGroupIdentifier(func_grp_smarts)


# load a sample from a file name (JCAMP file)
def load_sample(datafolder, identifier):
    path = f"{datafolder}/samples/{identifier}"
    try:
        sample = SpectraSample(path)
        if sample.skip: # skip if encountered error while reading data
            return None 
        return sample
    except Exception as e:
        print(f"Error in {identifier}: {e}")
        return None
    
def load_samples(datafolder):
    """Load spectral samples with functional group labels, optimized version"""
    with open(f"{datafolder}/meta_data.json", "rb") as f:
        compounds = orjson.loads(f.read())


    attachments = [
        (compound.get("cano_smiles"), attachment["identifier"].split("/", 1)[1])
        for compound in compounds if compound.get("cano_smiles")
        for dataset in compound.get("datasets", [])
        for attachment in dataset.get("attacments", [])
    ]

    samples = []
    pbar = tqdm(total=len(attachments), desc="Loading samples", colour="yellow")

    for smiles, identifier in attachments:
        try:
            sample = load_sample(datafolder, identifier)
            label_vec = func_group_id.encode(smiles)
            if (sample is not None) and (label_vec is not None):
                sample.labels = label_vec
                samples.append(sample)
        except Exception as e:
            rprint(f"[red]Error in {identifier}[/red]: {e}")
        pbar.update(1)
    pbar.close()
    rprint(f"[bold green]Loaded {len(samples)}/{len(attachments)} valid samples with labels.[/bold green]")
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