# for loading
import orjson
from data_handling.SpectraSample import SpectraSample
from data_handling.func_group_identif import FunctionalGroupIdentifier

# for printing
from tqdm import tqdm
from rich import print as rprint
import time

# for optimizations
import os
import multiprocessing as mp

from config import FUNCTIONAL_GROUP_SMARTS
    


def load_samples(datafolder):
    # Obtain metadata
    with open(f"{datafolder}/meta_data.json", "rb") as f:
        metadata = orjson.loads(f.read())

    # Read out metadata
    paths, smiles, path_to_smiles = read_metadata(metadata, datafolder)

    # Compute functional groups
    smiles_to_functional_group = compute_functional_groups(smiles) # SMILES -> LABEL_VEC

    # Read IR spectra data
    samples = read_spectra_samples(paths, path_to_smiles, smiles_to_functional_group)

    return samples


def read_metadata(metadata, datafolder):
    # Precompute all SMILES labels
    paths = [] # PATHS TO FILES 
    smiles = [] # SMILES
    path_to_smiles = {} # PATH -> SMILES

    # Read out metadata
    for compound in metadata:
        smile = compound.get("cano_smiles")
        if not smile:
            continue
        for dataset in compound.get("datasets", []):
            for attachment in dataset.get("attacments", []):
                identifier = attachment["identifier"].split("/", 1)[1]
                path = f"{datafolder}/samples/{identifier}"
                paths.append(path)
                smiles.append(smile)
                path_to_smiles[path] = smile
    return paths, smiles, path_to_smiles


def compute_functional_groups(smiles):
    func_group_identifier = FunctionalGroupIdentifier(FUNCTIONAL_GROUP_SMARTS)
    smiles_labels = {}

    start_time = time.time()
    for smile in tqdm(smiles, desc="Encoding SMILES", colour="yellow"):
        label_vec = func_group_identifier.encode(smile)
        if label_vec is not None:
            smiles_labels[smile] = label_vec
    label_time = time.time() - start_time
    rprint(f"[cyan]Functional groups computed in {label_time:.2f}s[/cyan]")
    return smiles_labels

def read_spectra_samples(paths, path_to_smiles, smiles_to_functional_group):
    # Parallel loading of spectra samples
    start_time = time.time()
    samples = []
    with mp.Pool(processes=calculate_max_workers()) as pool:
        # Create a progress bar that updates as we get results
        for result in tqdm(pool.imap_unordered(load_sample_parallel, paths), total=len(paths), desc="Loading samples", colour="yellow"):
            if not result:
                continue
            sample, path = result
            
            smile = path_to_smiles.get(path)
            if not smile:
                continue

            label_vec = smiles_to_functional_group.get(smile)
            if not label_vec:
                continue
            sample.labels = label_vec

            samples.append(sample)
    load_time = time.time() - start_time
    rate = len(paths) / load_time if load_time > 0 else 0
    rprint(f"[bold green]Loaded {len(samples)}/{len(paths)} samples at {rate:.0f} samples/s[/bold green]")
    return samples

def calculate_max_workers():
    cpu_count = os.cpu_count() or 1
    return min(32, int(cpu_count / 2))

# Parallel file loader
def load_sample_parallel(path):
    sample = SpectraSample(path)
    return (sample, path) if not sample.skip else None
