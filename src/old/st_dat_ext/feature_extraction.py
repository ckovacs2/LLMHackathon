from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from io import StringIO
import numpy as np

def extract_material_features(mp_entry, cutoff=3.0, primitive=False):
    """
    Return a string formatted like your original file example, with // explanations visible.
    mp_entry: dict with keys 'metadata' and 'cif' (CIF text).
    """
    cif_text = mp_entry["cif"]
    parser = CifParser(StringIO(cif_text))
    structure = parser.parse_structures(primitive=primitive)[0]

    # Space group + crystal system
    sga = SpacegroupAnalyzer(structure)
    sg_symbol = sga.get_space_group_symbol()
    sg_number = sga.get_space_group_number()
    crystal_system = sga.get_crystal_system()

    # Volume per atom, density
    volume_per_atom = structure.volume / len(structure.sites)
    density = mp_entry.get("metadata", {}).get("density", None)

    # Coordination numbers and bond lengths
    cnn = CrystalNN()
    cn_list = []
    bond_lengths = []
    for i, site in enumerate(structure.sites):
        try:
            cn = cnn.get_cn(structure, i)
        except Exception:
            cn = None
        cn_list.append(cn if cn is not None else np.nan)

        neighbors = structure.get_neighbors(site, cutoff)
        for n in neighbors:
            # handle different neighbor object shapes safely
            d = None
            try:
                # neighbor may be (site, dist)
                d = n[1]
            except Exception:
                try:
                    d = n.distance
                except Exception:
                    pass
            if d is not None:
                bond_lengths.append(d)

    avg_cn = float(np.nanmean(cn_list)) if len(cn_list) > 0 else None
    mean_bond_length = float(np.mean(bond_lengths)) if bond_lengths else None
    bond_length_std = float(np.std(bond_lengths)) if bond_lengths else None

    # Valence electron count (per formula unit) and electronegativity
    comp = structure.composition
    valence_electron_count = 0
    en_list = []
    for elem, amt in comp.get_el_amt_dict().items():
        e = Element(elem)
        # get valence electrons by summing outer-shell electrons (best-effort)
        valence = None
        try:
            full_es = e.full_electronic_structure
            max_n = max([t[0] for t in full_es])
            valence = sum([t[2] for t in full_es if t[0] == max_n])
        except Exception:
            # fallback: try e.valence if available
            valence = getattr(e, "valence", None)
        if valence is None:
            valence = 0
        valence_electron_count += int(valence) * int(amt)

        en = getattr(e, "X", None)
        if en is not None:
            en_list.extend([en] * int(amt))

    electronegativity_mean = float(np.mean(en_list)) if en_list else None
    electronegativity_difference = (
        float(max(en_list) - min(en_list)) if len(en_list) >= 2 else None
    )

    # prepare text_summary (concise human-readable sentence)
    atoms_per_cell = len(structure.sites)
    try:
        # round avg_cn to integer for summary if available
        avg_cn_rounded = int(round(avg_cn)) if avg_cn is not None and not np.isnan(avg_cn) else "N/A"
    except Exception:
        avg_cn_rounded = "N/A"

    text_summary = (
        f"{crystal_system.capitalize()} {comp.formula} ({sg_symbol}), "
        f"{atoms_per_cell} atoms per cell, "
        f"avg CN ≈ {avg_cn_rounded}, "
        f"mean bond length ≈ {mean_bond_length:.2f} Å." if mean_bond_length is not None else
        f"{crystal_system.capitalize()} {comp.formula} ({sg_symbol}), {atoms_per_cell} atoms per cell."
    )

    # Format the output string with // comments (matching your original style)
    # Use indenting for nested structural_features block
    def fmt_val(v):
        if v is None:
            return "null"
        if isinstance(v, float):
            return f"{v:.4g}"
        return repr(v)

    s = []
    s.append(f"\"formula\": {repr(comp.formula)}, // Composition")
    s.append(f"\"text_summary\": {repr(text_summary)},")
    s.append("\"structural_features\": {")
    s.append(f"  \"space_group_number\": {fmt_val(sg_number)}, // Numeric space group ID")
    s.append(f"  \"space_group_symbol\": {repr(sg_symbol)}, // Symbolic space group name")
    s.append(f"  \"crystal_system\": {repr(crystal_system)}, // crystal system (e.g. tetragonal)")
    s.append(f"  \"volume_per_atom\": {fmt_val(volume_per_atom)}, // Å³/atom, relates to orbital overlap and bandwidth")
    s.append(f"  \"density\": {fmt_val(density)}, // g/cm³, affects electronic overlap")
    s.append(f"  \"valence_electron_count\": {fmt_val(valence_electron_count)}, // Total valence electrons per formula unit")
    s.append(f"  \"avg_coordination_number\": {fmt_val(avg_cn)}, // Crystal-field splitting & band shape")
    s.append(f"  \"mean_bond_length\": {fmt_val(mean_bond_length)}, // Average nearest-neighbor bond length (Å)")
    s.append(f"  \"bond_length_std\": {fmt_val(bond_length_std)}, // Bond length variation → DOS broadening")
    s.append(f"  \"electronegativity_mean\": {fmt_val(electronegativity_mean)}, // Mean electronegativity of constituent elements")
    s.append(f"  \"electronegativity_difference\": {fmt_val(electronegativity_difference)} // EN difference → gap size trend")
    s.append("}")

    return "\n".join(s)

# -------------------------
# Example usage:
mp_entry = mos_results[4]
description = extract_material_features(mp_entry)
print(description)



