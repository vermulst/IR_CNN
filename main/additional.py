import os
from jcamp import jcamp_readfile
import matplotlib.pyplot as plt

# Relative to script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, "..", "data", "public", "samples", "00a0ee9a-ac02-4c9e-96a6-656d069fb80a")
filepath = os.path.normpath(relative_path)

# Load and plot
data = jcamp_readfile(filepath)
x = data['x']
y = data['y']

plt.plot(x, y)
plt.title('IR Spectrum')
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
plt.gca().invert_xaxis()
plt.show()
