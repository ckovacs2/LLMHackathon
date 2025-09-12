import numpy as np
from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from dotenv import load_dotenv


def collect_materials_by_species(mpr: MPRester, species: str | list[str]):
    """
    Collect metadata, DOS data, and CIF string for ALL materials
    containing the given species, but only include those that have DOS available.

    Args:
        mpr: Authenticated MPRester client.
        species: One element symbol or list of elements (e.g. "Si" or ["Mo", "S"]).

    Returns:
        results: list of dicts, each with keys 'metadata', 'dos', 'cif'
    """
    if not species:
        raise ValueError("Must provide at least one species.")

    if isinstance(species, str):
        species = [species]

    # -------------------------------
    # Search for all matching material_ids
    # -------------------------------
    docs = mpr.materials.summary.search(
        chemsys="-".join(species),
        fields=["material_id"]
    )

    if not docs:
        raise ValueError(f"No materials found with species={species}")

    results = []

    for doc in docs:
        mid = doc.material_id

        # --- DOS ---
        dos_data = None
        try:
            dos_doc = mpr.get_dos_by_material_id(mid)
            if dos_doc:
                efermi = dos_doc.efermi
                en = dos_doc.energies
                dens = dos_doc.densities
                idx = np.argmin(np.abs(en - efermi))
                dos_at_ef = sum(dens[spin][idx] for spin in dens)
                dos_data = {
                    "efermi": efermi,
                    "energies": en,
                    "densities": dens,
                    "DOS@E_F": dos_at_ef,
                }
        except MPRestError:
            continue  # Skip this material if DOS is not available

        if dos_data is None:
            continue  # Skip if DOS still not populated

        # --- Metadata ---
        try:
            summary = mpr.materials.summary.get_data_by_id(mid)
            metadata = {
                "material_id": summary.material_id,
                "formula": summary.formula_pretty,
                "spacegroup": summary.symmetry.symbol if summary.symmetry else None,
                "spacegroup_number": summary.symmetry.number if summary.symmetry else None,
                "density": summary.density,
                "volume": summary.volume,
                "band_gap": summary.band_gap,
                "e_above_hull": summary.energy_above_hull,
                "is_stable": summary.is_stable,
            }
        except MPRestError:
            continue  # Skip if metadata fetch fails

        # --- CIF ---
        try:
            struct = mpr.get_structure_by_material_id(mid)
            cif_str = struct.to(fmt="cif") if struct else None
        except MPRestError:
            cif_str = None

        results.append({
            "metadata": metadata,
            "dos": dos_data,
            "cif": cif_str,
        })

    return results
