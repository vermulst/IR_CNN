from csv_to_jcamp import csv_to_jcamp

from os import listdir
from rdkit import Chem
from PIL import Image
import os
import numpy as np
import csv
import cv2
import re
import orjson
from typing import List, Dict

from pubchempy import get_compounds
import cirpy 
from py2opsin import py2opsin

# Define paths.
nist_inchi_path = '/nist_dataset/inchi/'
nist_jdx_path = '../nist_dataset/jdx/'
sdbs_gif_path = './data/public/sdbs/sdbs_dataset/gif/'
sdbs_png_path = './data/public/sdbs/sdbs_dataset/png/'
sdbs_other_path = './data/public/sdbs/sdbs_dataset/other/'
save_path = './data/public/sdbs/processed/samples/'
metadata_save_path = './data/public/sdbs/processed/'

# Function to convert name to SMILES
def name_to_smiles(name):
    """
    Convert a chemical compound name to its SMILES string using a cascade of methods.
    Returns the SMILES string or None if not found by any method.
    """
    smiles = None # Initialize smiles to None

    try:
        # try numerous methods
        smiles = py2opsin(name)
        if smiles:
            #print(f"Resolved '{name}' via py2opsin: {smiles}")
            return smiles # Return if successful
        smiles = cirpy.resolve(name, 'smiles')
        if smiles:
            #print(f"Resolved '{name}' via cirpy: {smiles}")
            return smiles # Return if successful
        smiles = get_compounds(name)
        if smiles:
            return smiles
        print(f"Could not resolve: {name}")
        return None
    except Exception as e:
        print(f"py2opsin failed for '{name}': {e}")


def name_to_smiles_and_inchi(name):
    smiles = cirpy.resolve(name, 'smiles')
    if smiles is None:
        print(f"Could not resolve SMILES for: {name}")
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"RDKit could not parse SMILES for: {name}")
        return smiles, None
    inchi = Chem.MolToInchi(mol)
    return smiles, inchi

def convert_x(x_in, unit_from, unit_to):
    """Convert between micrometer and wavenumber."""
    if unit_to == 'micrometers' and x_out == 'MICROMETERS':
        x_out = x_in
        return x_out
    elif unit_to == 'cm-1' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = x_in
        return x_out
    elif unit_to == 'micrometers' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out
    elif unit_to == 'cm-1' and unit_from == 'MICROMETERS':
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out


def convert_y(y_in, unit_from, unit_to):
    """Convert between absorbance and trasmittance."""
    if unit_to == 'transmittance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = y_in
        return y_out
    elif unit_to == 'absorbance' and unit_from == 'ABSORBANCE':
        y_out = y_in
        return y_out
    elif unit_to == 'transmittance' and unit_from == 'ABSORBANCE':
        y_out = np.array([1 / 10 ** j for j in y_in])
        return y_out
    elif unit_to == 'absorbance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = np.array([np.log10(1 / j) for j in y_in])
        return y_out


def get_png():
    """Convert GIF to PNG."""
    # Check if PNG folder exists.
    if not os.path.exists(sdbs_png_path):
        os.makedirs(sdbs_png_path)
    files = listdir(sdbs_gif_path)
    for file in files:
        if not file.startswith('.'):
            from_path = sdbs_gif_path + file
            img = Image.open(from_path)
            file_name = os.path.splitext(file)[0]
            to_path = sdbs_png_path + file_name + '.png'
            img.save(to_path, 'png')


def get_unique(x_in, y_in):
    """Removes duplicates in x and takes smallest y value for each x value."""
    x_out = sorted(list(set(x_in)), reverse=True)
    y_out = []
    for i in x_out:
        y_temp = []
        for ii, j in zip(x_in, y_in):
            if i == ii:
                y_temp.append(j)
        y_out.append(min(y_temp))
    return x_out, y_out


def get_sdbs():
    print('Start SDBS processing')
    metadata: List[Dict] = []
    for file in os.listdir(sdbs_png_path):
        if ('KBr' not in file) and ('liquid' not in file) and ('nujol' not in file):
            continue
        if file.startswith('.'):
            continue
        
        # Image reading and filtering for width
        original_image = cv2.imread(os.path.join(sdbs_png_path, file), 0)
        if original_image is None:
            print(f"Failed to read image: {file}")
            continue
        _, width = original_image.shape
        if width != 715:
            print(f"Skipping {file}: width is not 715")
            continue

        # Metadata extraction
        sdbs_id = file.split('_')[0]
        other_file = os.path.join(sdbs_other_path, sdbs_id + '_other.txt')
        if not os.path.exists(other_file):
            print(f"Missing metadata: {other_file}")
            continue
        
        # Compound extraction
        compound_name = extract_compound_name(other_file)
        if not compound_name:
            continue
        
        # Spectrum Processing
        x, y = np.linspace(4000, 400, 600), np.random.rand(600)

        # --- Save Data ---
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", compound_name)
        save_spectrum_data(x, y, safe_name)

        # ADD TO CONFIG
        smiles = name_to_smiles(compound_name)
        if smiles:
            metadata.append({
                "smiles": smiles,
                "filename": f"{safe_name}.jdx",
                "compound_name": compound_name,
                "sdbs_id": sdbs_id
            })
    return metadata
            
# Helper functions (would be defined elsewhere)
def extract_compound_name(path):
    """Extracts compound name from metadata file."""
    with open(path) as f:
        for line in f:
            if match := re.match('Name: (.*)', line):
                return match.group(1).strip()
    return None

def sanitize_filename(name: str) -> str:
    """Makes filenames filesystem-safe."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def save_spectrum_data(x, y, base_name):
    """Saves spectrum data in CSV and JCAMP formats."""
    csv_path = os.path.join(save_path, f"{base_name}.csv") 
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Wavenumber', 'Intensity'])
        writer.writerows(zip(x, y))
    
    csv_to_jcamp(
        csv_path=str(csv_path),
        jcamp_path=str(os.path.join(save_path, f"{base_name}.jdx")),
        title=base_name,
        origin="SDBS",
        sampling_procedure="ATR"
    )

def save_metadata(metadata: List[Dict], filename: str = "metadata.json") -> None:
    """Save metadata using orjson for better performance."""
    metadata_path = os.path.join(metadata_save_path, filename)
    
    unique_entries = {}
    for entry in metadata:
        sdbs_id = entry.get("sdbs_id")
        if sdbs_id and sdbs_id not in unique_entries:
            unique_entries[sdbs_id] = entry
    try:
        with open(metadata_path, 'wb') as f:
            f.write(orjson.dumps(
                list(unique_entries.values()),  # Convert back to list
                option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            ))
        print(f"Saved {len(unique_entries)} unique entries to {metadata_path}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")

if __name__ == '__main__':
    get_png() 
    metadata = get_sdbs()
    save_metadata(metadata)
    
    #get_nist(fg_list_extended)