import os
import json
import requests
import time
from urllib.parse import quote

# Configuration
INPUT_DIR = "data/public/nist_dataset/samples"
OUTPUT_FILE = "data/public/nist_dataset/metadata.json"
REQUEST_DELAY = 0.05  # 20ms between requests

def extract_title(content):
    """Extract TITLE property from JCAMP content"""
    for line in content.splitlines():
        if line.startswith('##TITLE='):
            return line[8:].strip()
    return None

def get_smiles_from_title(title):
    """Get SMILES from title using PubChem API"""
    try:
        # URL encode the title
        encoded_title = quote(title)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_title}/property/IsomericSMILES/JSON"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass
    return None

def process_files():
    """Process all JDX files and return SMILES metadata"""
    metadata = []
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.jdx'):
            continue
            
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        title = extract_title(content)
        if not title:
            continue
            
        smiles = get_smiles_from_title(title)
        if smiles:
            metadata.append({
                "smiles": smiles,
                "filename": filename
            })
            print(f"Converted: {title} -> {smiles}")
        
        # Respect PubChem rate limits
        time.sleep(REQUEST_DELAY)
    
    return metadata

def save_metadata(metadata):
    """Save metadata to JSON file"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {len(metadata)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    print("Starting processing...")
    metadata = process_files()
    save_metadata(metadata)
    print("Done!")