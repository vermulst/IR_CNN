"""Download all IR spectra and structure files available from nist chemistry webbook."""

from bs4 import BeautifulSoup
import csv
import os
import requests
import re
import lxml.html
import glob


# Define paths.
nist_path = "data/public/nist_dataset/"
jdx_path = f'{nist_path}jdx/'
name_path = f'{nist_path}name/'

# Define url for get requests.
nist_url = 'http://webbook.nist.gov/cgi/cbook.cgi'

# Define regex for nist IDs.
id_re = re.compile('/cgi/cbook.cgi\?ID=(.*?)&')


def check_dir():
    """Check if file directories exist and create them if they do not."""
    if not os.path.exists(jdx_path):
        os.makedirs(jdx_path)
    if not os.path.exists(name_path):
        os.makedirs(name_path)

def get_jdx(nistid, indices):
    """Download ir jdx file for specified nist id, if not already downloaded."""
    for index in range(0, indices + 1):
        filepath = os.path.join(jdx_path, '%s_%s.jdx' % (nistid, index))
        if os.path.isfile(filepath):
            print('%s_%s.jdx: already exists' % (nistid, index))
            continue
        params = {'JCAMP': nistid, 'Type': 'IR', 'Index': index}    
        response = requests.get(nist_url, params=params)
        if index == 0 and response.text.splitlines()[0] == '##TITLE=Spectrum not found.':
            print('%s_%s.jdx: file not found' % (nistid, index))
            return
        elif index > 0 and response.text.splitlines()[0] == '##TITLE=Spectrum not found.':
            return
        print('%s_%s.jdx: downloading' % (nistid,index))
        with open(filepath, 'wb') as file:
            file.write(response.content)



def get_name(nistid):
    """Get all names for specified nist id and store in a file, if not already done so."""
    filepath = os.path.join(name_path, '%s_name.txt' % nistid)
    if os.path.isfile(filepath):
        print('%s_name.txt: already exists' % nistid)
        return
    
    # Count existing JCAMP files for this ID
    jcamp_count = len(glob.glob(os.path.join(jdx_path, f'{nistid}_*.jdx')))
    
    params = {'ID': nistid, 'Units': 'SI'}
    response = requests.get(nist_url, params=params, stream=True)
    response.raw.decode_content = True
    html = lxml.html.parse(response.raw)
    formula = html.xpath('/html/body/main/ul[1]/li[1]/strong/a/text() | /html/body/main/ul[1]/li[1]/text() | /html/body/main/ul[1]/li[1]/*/text()')
    
    with open(filepath, 'w+') as file:
        file.write('ID: %s\n' % nistid)
        file.write('JCAMP_files: %d\n' % jcamp_count)  # Add JCAMP count
        if 'Formula' in ''.join(formula):
            file.write(''.join(formula))


def search_formula(formula):
    """Single nist search using the specified formula query and return the matching nist ids."""
    print('Searching formula: %s' % formula)
    params = {'Formula': formula, 'Units': 'SI', 'NoIon': 'on', 'cIR': 'on'}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text)
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result: %s' % ids)
    return ids


def search_mw(mw):
    """Single nist search using the specfied molecular weight query and returns the matching nist ids."""
    print('Searching molecular weight: %s' % mw)
    params = {'Value': mw, 'VType': 'MW', 'Units': 'SI', 'cIR': 'on', "Formula": ''}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text, features='lxml')
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result %s' % ids)
    return ids


def search_name(name):
    """Single nist search using the specified name query and return the matching nist ids."""
    print('Searching name: %s' % name)
    params = {'Name': name, 'Units': 'SI', 'cIR': 'on'}
    response = requests.get(nist_url, params=params)
    soup = BeautifulSoup(response.text)
    ids = list(set([re.match(id_re, link['href']).group(1) for link in soup('a', href=id_re)]))
    print('Result: %s' % ids)
    return ids


def get_nistid_formula_name(file):
    """Return all nist ids matched using formula or name query. Search name only if formula does not exist."""
    nistids = []
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[1]:
                nistids += list(set(search_formula(row[1])))
            elif not row[1]:
                nistids += list(set(search_name(row[0])))
    return nistids


def get_nistid_mw(num, current):
    """Return all nist ids matched using the molecular weight query."""
    if os.path.exists(f'{nist_path}ids.txt'):
        print('ids.txt already exists')
        nistids = []
        file = open(f'{nist_path}ids.txt')
        for i, line in enumerate(file):
            if i > current:
                nistids.append(line.rstrip('\n'))
        file.close()
        return nistids
    elif not os.path.exists(f'{nist_path}ids.txt'):
        nistids = []
        for mw in range(1, num):
            nistids += list(set(search_mw(mw)))
            nistids = list(set(nistids))
        print('Total IDs: %s' % len(nistids))
        file = open(f'{nist_path}ids.txt', 'w+')
        for nistid in nistids:
            file.write(nistid)
            file.write('\n')
        file.close()
        return nistids


if __name__ == '__main__':
    """Search nist for all compounds with ir spectra and downloads jdx, mol, sdf, inchi, inchikey, name, molecular weight files."""
    # Molecular weight to start search from.
    num = 1

    # Differenciates multiple entries of same molecule.
    indices = 10

    # Position of the ID to start search from.
    current = 0

    check_dir()

    nistids = get_nistid_mw(num, current)

    print('Downloading files')

    for nistid in nistids:
        print('ID count: %s' % (current - 1))
        get_jdx(nistid, indices)
        get_name(nistid)
        current += 1

    print('Download complete')