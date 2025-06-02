# for loading
import orjson
from data_handling.SpectraSample import SpectraSample
from data_handling.func_group_identif import FunctionalGroupIdentifier

# for printing
from tqdm import tqdm
from rich import print as rprint

# for optimizations
import concurrent.futures
import os

from config import FUNCTIONAL_GROUP_SMARTS
    

def calculate_max_workers():
    cpu_count = os.cpu_count() or 1
    # For I/O-heavy tasks with light CPU
    return min(32, cpu_count * 4)


def load_samples(datafolder):
    func_group_id = FunctionalGroupIdentifier(FUNCTIONAL_GROUP_SMARTS)

    with open(f"{datafolder}/meta_data.json", "rb") as f:
        compounds = orjson.loads(f.read())

    # Precompute all unique SMILES labels
    smiles_labels = {}
    unique_smiles = set()

    # First pass: Collect all unique SMILES
    attachments = []
    for compound in compounds:
        if smiles := compound.get("cano_smiles"):
            for dataset in compound.get("datasets", []):
                for attachment in dataset.get("attacments", []):
                    identifier = attachment["identifier"].split("/", 1)[1]
                    attachments.append((smiles, identifier))
                    unique_smiles.add(smiles)


    # Precompute labels with progress bar
    with tqdm(total=len(unique_smiles), desc="Precomputing labels", colour="yellow") as pbar:
        for smiles in unique_smiles:
            if label_vec := func_group_id.encode(smiles):
                smiles_labels[smiles] = label_vec
            pbar.update(1)

     # Parallel loading
    samples = []
    max_workers = calculate_max_workers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures
        future_to_id = {}
        for smiles, identifier in attachments:
            future = executor.submit(load_sample, datafolder, identifier)
            future_to_id[future] = (smiles, identifier)

        # Process results with single progress bar
        with tqdm(total=len(attachments), desc="Loading samples", colour="yellow") as pbar:
            for future in concurrent.futures.as_completed(future_to_id):
                smiles, identifier = future_to_id[future]
                try:
                    sample = future.result()
                    if sample and (label_vec := smiles_labels.get(smiles)):
                        sample.labels = label_vec
                        samples.append(sample)
                except Exception as e:
                    rprint(f"[red]Error processing {identifier}[/red]: {e}")
                finally:
                    pbar.update(1)  # Ensure progress bar updates
    rprint(f"[bold green]Loaded {len(samples)}/{len(attachments)} valid samples with labels.[/bold green]")
    return samples


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