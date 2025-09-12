import uuid
import json
from io import StringIO
from pymatgen.io.cif import CifParser
from mp_api.client import MPRester

from mp_fetch import collect_materials_by_species
from feature_extraction import extract_material_features
from dos_description import get_dos_features



    
def generate_unique_id(prefix="mat"):
    """
    Generate a unique ID string with an optional prefix.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_database(MP_API_KEY, species: list,  cutoff=5.0, output_file="materials_database.json"):
    """
    Collect structural and DOS features from MP and save as JSON database.

    Args:
        MP_API_KEY (str): Your Materials Project API key
        species (list): List of element symbols to query (e.g., ["Mo", "S"])
        output_file (str): Path to JSON file to save output
    """
    with MPRester(MP_API_KEY) as mpr:
        raw_results = collect_materials_by_species(mpr, species=species)
        filtered_results = [r for r in raw_results if r['dos'] is not None]

    print(f"Found {len(filtered_results)} stable materials containing {', '.join(species)} with available DOS")

    overall_data = {}

    for i, result in enumerate(filtered_results):
        try:
            uid = generate_unique_id()
    
            # DOS features
            dos_info, desc_dict,description = get_dos_features(result['dos'], spin=1, smoothing=1.0)

            # Structural features
            cif_text = result['cif']
            parser = CifParser(StringIO(cif_text))
            structure_info = extract_material_features(parser, cutoff=cutoff, primitive=True)
    
            # Combine and store
            overall_data[uid] = {
                "structure": structure_info,
                "dos": {
                    "dos_features": dos_info,
                    "dos_description": description,
                    "dos_description_dict": desc_dict
                }
            }
        except Exception as e:
            print(f"[Warning] Skipping material {i} due to error: {e}")

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(overall_data, f)

    print(f"\nSaved database of {len(overall_data)} materials to {output_file}")
