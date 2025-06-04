
import csv
import os
from datetime import datetime

def csv_to_jcamp(csv_path, jcamp_path, title="Sample Spectrum", origin="Python Script", 
                 sampling_procedure="", xunit="1/CM", yunit="TRANSMITTANCE", 
                 data_type="INFRARED SPECTRUM", values_per_line=6):
    
    # Read CSV data
    x = []
    y = []
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                xi = float(row[0])
                yi = float(row[1])
                x.append(xi)
                y.append(yi)
            except Exception as e:
                print(f"Skipping row {row}: {e}")

    if not x or not y:
        print("No data found in CSV.")
        return

    # Calculate JCAMP header values
    npoints = len(x)
    firstx = x[0]
    lastx = x[-1]
    deltax = (lastx - firstx) / (npoints - 1) if npoints > 1 else 0
    maxy = max(y)
    miny = min(y)
    firsty = y[0]
    resolution = abs(deltax)
    xfactor = 1
    yfactor = 1

    # Get current date
    current_date = datetime.now().strftime("%d/%m/%Y")

    # Write JCAMP-DX file
    with open(jcamp_path, 'w') as f:
        f.write(f"##TITLE={title}\n")
        f.write(f"##JCAMP-DX=4.24\n")
        f.write(f"##DATA TYPE={data_type}\n")
        f.write(f"##DATE={current_date}\n")
        if sampling_procedure:
            f.write(f"##SAMPLING PROCEDURE={sampling_procedure}\n")
        f.write(f"##ORIGIN={origin}\n")
        f.write(f"##XUNITS={xunit}\n")
        f.write(f"##YUNITS={yunit}\n")
        f.write(f"##RESOLUTION={resolution}\n")
        f.write(f"##FIRSTX={firstx}\n")
        f.write(f"##LASTX={lastx}\n")
        f.write(f"##DELTAX={deltax}\n")
        f.write(f"##MAXY={maxy}\n")
        f.write(f"##MINY={miny}\n")
        f.write(f"##XFACTOR={xfactor}\n")
        f.write(f"##YFACTOR={yfactor}\n")
        f.write(f"##NPOINTS={npoints}\n")
        f.write(f"##FIRSTY={firsty}\n")
        f.write(f"##XYDATA=(X++(Y..Y))\n")

        # Write data in JCAMP format: X+Y1+Y2+Y3+Y4+Y5+Y6
        i = 0
        while i < npoints:
            # Start line with X value
            line = f"{x[i]:.0f}"
            
            # Add Y values for this line (up to values_per_line)
            for j in range(values_per_line):
                if i + j < npoints:
                    # Convert Y values to integers (multiply by large factor for precision)
                    y_val = int(y[i + j] * 1e9)  # Scale factor like in your example
                    line += f"+{y_val}"
            
            f.write(line + "\n")
            i += values_per_line

        f.write("##END=\n")

# Example usage:

from sdbs_preprocessing import name_to_smiles


smiles = name_to_smiles("2-furoic acid")
csv_to_jcamp(
    csv_path="./data/public/sdbs/processed_dataset/2-furoic acid.csv", 
    jcamp_path="output.jdx",
    title="CHS-424",
    origin="Your Name",
    sampling_procedure="Diamant-ATR",
    values_per_line=1  # Number of Y values per line (6 in your example)
)
