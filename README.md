### Setup

## Option 1: Virtual Python Environment

- Run `setup/venv.bat` to create the virtual environment.
- VS Code is configured to use it via `.vscode/settings.json`.
- All required libraries will be installed in the environment.

## Option 2: Install Python Libraries System-Wide

- Run `setup/system.bat`.
- This may fail if multiple Python installations are present on your system.

## Option 3: Anaconda Virtual Enviorment from Command Prompt
- Install Anaconda
- Go to repository directory
- Run conda create --name ir python=3.11 (answer yes too al questions)
- Run conda activate ir

---

# JCAMP-DX Format Overview

The standard format for spectroscopic data found online and used by the Chemotion repository is JCAMP-DX file. Therefore I looked into the format to understand how to extract the data. JCAMP-DX files use only printable ASCII characters, it consists of a data label and an associated data set. It starts with ##TITLE= and ending with ##END=;

## Relevant Fields to Extract

The relevant information that we should extract from a JCAMP-DX file for our purposes:
- ##TITLE= : Description of the spectrum (required as first LDR)
- ##DATA TYPE= : Specifies the type of data (e.g., INFRARED SPECTRUM, MASS SPECTRUM)
- ##SAMPLING PROCEDURE= : Instrument details
- ##XUNITS= : Defines abscissa units (e.g., 1/CM for wavenumber)
- ##YUNITS= : Defines ordinate units (e.g., ABSORBANCE, TRANSMITTANCE)
- ##XFACTOR=, ##YFACTOR= : Scaling factors
- ##FIRSTX=, ##LASTX= : First and last abscissa values
- ##NPOINTS= : Number of data points
- ##XYDATA= : For ordinates at equal X-intervals
- (X++(Y..Y)) : For data with equally spaced X values
- ##DELTAX= spacing of X-intervals

---

# SMILES and SMARTS

The SMILES name for the compound of the respective spectra is in the JSON metadata (for Chemotion JCAMP files). The 2025 Punjabi J paper provides the code for extracting spectroscopic data from JCAMP files into their custom SpectraCarrier() class. I would suggest to use their code to extract the data from JCAMP files of the chemotion repository. In this code they include a lot of data preprocessing that should be looked into. We can then format the data into the class or other data structure we will use in our algorithm.

Additionally since our goal is to determine the functional groups we can should convert the SMILES name in SMARTS format which makes it much easier to search for the present functional groups.

From the meetings we concluded we would want to collect our own data, talking with the chemists of our group I concluded that the data would come in a .spc format. I tried to find code or a program online which extracted that relevant data but I didn’t manage to make this work. Although this is not necessary right now it could serve as an SSA in the future.


# How to run
- Run `setup/venv.bat` to start the virtual environment.
- Run main.py to train the model (if altered).
- Run test.py to test specific JCAMP files on the model.
