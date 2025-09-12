#!/usr/bin/env python3

"""
Author: Ankita Biswas
Date: 09/11/2025
Version: 1.0
"""

# Extract DOS data with material IDs
all_dos = []
material_ids = []

for i in mos_results:
    dos_i = i['dos']
    material_id = i['metadata']['material_id']  # Get the material ID
    all_dos.append(dos_i)
    material_ids.append(material_id)

# Filter both lists together
filtered_all_dos = []
filtered_material_ids = []
for dos, mid in zip(all_dos, material_ids):
    if dos is not None:
        filtered_all_dos.append(dos)
        filtered_material_ids.append(mid)

# Now save files with material IDs
for i, (dos_data, material_id) in enumerate(zip(filtered_all_dos, filtered_material_ids)):
    # Use material ID in filename
    filename_prefix = f"dos_{material_id}"
    save_mp_dos_to_txt(dos_data, filename_prefix=filename_prefix, spin=1, index=None)
    
    # Also store material ID in the dos data for future reference
    dos_data['material_id'] = material_id

# Alternative: Modified save function to handle material IDs directly
def save_mp_dos_to_txt_with_id(mp_dos_data, material_id, filename_prefix="dos", spin=1):
    """
    Convert Materials Project DOS format to simple text file with material ID.
    
    Args:
        mp_dos_data: DOS data from Materials Project
        material_id: Material ID for filename
        filename_prefix: Prefix for output filename
        spin: Which spin to use (1 for up, -1 for down)
    """
    energies = mp_dos_data['energies']
    densities = mp_dos_data['densities']
    
    # Handle spin selection
    if isinstance(densities, dict):
        if spin in densities:
            dos_values = densities[spin]
        else:
            dos_values = list(densities.values())[0]
            print(f"Warning: Spin {spin} not found, using first available spin")
    else:
        dos_values = densities
    
    # Shift energies so Fermi level is at 0
    energies_shifted = energies - mp_dos_data['efermi']
    
    # Create filename with material ID
    filename = f"{filename_prefix}_{material_id}.txt"
    
    # Combine into 2D array and save
    data_to_save = np.column_stack((energies_shifted, dos_values))
    np.savetxt(filename, data_to_save, fmt='%.6f', delimiter=' ')
    print(f"Saved DOS data to {filename}")
    print(f"Fermi level was at {mp_dos_data['efermi']:.6f} eV (now shifted to 0 eV)")

# Use the modified function
for dos_data, material_id in zip(filtered_all_dos, filtered_material_ids):
    save_mp_dos_to_txt_with_id(dos_data, material_id)
