import os
import glob
import argparse
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d



#-------- utility----------------

def align_to_ef(E, dos, ef=0.0):
    """Shift energy values so that Fermi level (EF) is at 0."""
    return E - ef, dos

def window_mask(E, low, high):
    """Create a boolean mask for energy values between low and high."""
    return (E >= low) & (E <= high)

def finite_diff(y, x):
    """Calculate first and second derivatives of y with respect to x."""
    dy = np.gradient(y, x)  # First derivative (slope)
    d2y = np.gradient(dy, x)  # Second derivative (curvature)
    return dy, d2y



def largest_gap(E, dos, threshold=1e-3, span=(-6, 6)):
    """
    Find the largest energy range where DOS is below threshold.
    Returns: (gap_size, Valence Band Max, Conduction Band Min)
    """
    # Create mask for the energy range we're interested in
    mask = window_mask(E, span[0], span[1])
    if not np.any(mask):
        return 0.0, None, None

    E_masked = E[mask]
    dos_masked = dos[mask]

    # Find where DOS is below our threshold
    low_dos_regions = dos_masked < threshold

    # Find continuous regions of low DOS
    starts, ends = [], []
    in_gap = False

    for i, is_low in enumerate(low_dos_regions):
        if is_low and not in_gap:
            starts.append(i)  # Start of a gap
            in_gap = True
        elif not is_low and in_gap:
            ends.append(i-1)  # End of a gap
            in_gap = False

    if in_gap:  # If we ended while still in a gap
        ends.append(len(low_dos_regions)-1)

    # If no gaps found, return zeros
    if not starts or not ends:
        return 0.0, None, None

    # Find the widest gap
    gap_widths = [E_masked[end] - E_masked[start] for start, end in zip(starts, ends)]
    widest_idx = np.argmax(gap_widths)

    gap_size = max(0.0, float(gap_widths[widest_idx]))
    VBM = float(E_masked[ends[widest_idx]])  # Valence Band Maximum
    CBM = float(E_masked[starts[widest_idx]])  # Conduction Band Minimum

    return gap_size, VBM, CBM




def find_peaks_summary(E, dos, energy_range, prominence=0.05):
    """
    Find prominent peaks in the DOS within specified energy range.
    Returns list of peak properties (position, height, width).
    """
    # Create mask for the energy range
    mask = window_mask(E, energy_range[0], energy_range[1])
    if np.sum(mask) < 5:  # Need enough points for good peak finding
        return []

    E_range = E[mask]
    dos_range = dos[mask]

    # Set minimum prominence for a peak to be considered significant
    min_prominence = prominence * (np.max(dos_range) - np.min(dos_range) + 1e-12)

    # Find peaks using scipy's function
    peak_indices, peak_properties = find_peaks(dos_range, prominence=min_prominence)

    peaks = []
    if len(peak_indices) == 0:
        return peaks

    # Extract information for each peak found
    for i in range(len(peak_indices)):
        peak_energy = float(E_range[peak_indices[i]])
        peak_height = float(dos_range[peak_indices[i]])
        peak_prominence = float(peak_properties["prominences"][i])

        # Calculate approximate width if available
        if "widths" in peak_properties:
            width_in_samples = float(peak_properties["widths"][i])
            # Convert width from samples to energy units
            energy_per_sample = (E_range[-1] - E_range[0]) / len(E_range)
            fwhm = width_in_samples * energy_per_sample
        else:
            fwhm = np.nan

        peaks.append({
            "energy": peak_energy,
            "height": peak_height,
            "prominence": peak_prominence,
            "width": fwhm
        })

    # Sort peaks by prominence (most prominent first)
    peaks.sort(key=lambda x: x["prominence"], reverse=True)
    return peaks[:3]  # Return top 3 most prominent peaks



def detect_superconducting_gap(E, dos, gap_span=0.1, peak_threshold=5.0):
    """
    Detect signature of superconducting gap.
    Superconductors have near-zero DOS at Fermi level with sharp peaks nearby.
    Returns: (is_superconductor, gap_width, peak_ratio)
    """
    # Look at small region around Fermi level
    gap_mask = window_mask(E, -gap_span, gap_span)
    if not np.any(gap_mask):
        return False, 0.0, 0.0

    E_gap = E[gap_mask]
    dos_gap = dos[gap_mask]

    # Check if DOS is very low at Fermi level
    dos_at_ef = np.interp(0.0, E_gap, dos_gap)
    if dos_at_ef > 1e-3:  # Not low enough for superconductor
        return False, 0.0, 0.0

    # Find typical DOS value away from Fermi level
    far_mask = window_mask(E, -2*gap_span, -gap_span) | window_mask(E, gap_span, 2*gap_span)
    if not np.any(far_mask):
        return False, 0.0, 0.0

    typical_dos = np.median(dos[far_mask])

    # Find where DOS rises above 50% of typical value (gap edges)
    edge_threshold = 0.5 * typical_dos
    above_threshold = dos_gap > edge_threshold
    edge_indices = np.where(above_threshold)[0]

    if len(edge_indices) < 2:
        return False, 0.0, 0.0

    # Calculate gap width
    left_edge = E_gap[edge_indices[0]]
    right_edge = E_gap[edge_indices[-1]]
    gap_width = right_edge - left_edge

    # Look for coherence peaks just outside the gap
    left_peak_mask = window_mask(E, left_edge - 0.02, left_edge)
    right_peak_mask = window_mask(E, right_edge, right_edge + 0.02)

    left_peak = np.max(dos[left_peak_mask]) if np.any(left_peak_mask) else 0
    right_peak = np.max(dos[right_peak_mask]) if np.any(right_peak_mask) else 0
    avg_peak = (left_peak + right_peak) / 2

    # Calculate how much taller peaks are compared to normal DOS
    peak_ratio = avg_peak / typical_dos if typical_dos > 0 else 0

    # Superconductor signature: deep gap + tall peaks nearby
    is_superconductor = (peak_ratio > peak_threshold) and (gap_width > 0)

    return is_superconductor, gap_width, peak_ratio



# ---------- MAIN FEATURE EXTRACTION ----------



def extract_dos_features(E, dos, ef=0.0, smoothing=1.0):
    """
    Extract all important features from DOS data.
    Returns dictionary with all calculated properties.
    """
    # Align energy so Fermi level is at 0
    E, dos = align_to_ef(E, dos, ef=ef)

    # Apply slight smoothing to reduce noise
    if smoothing > 0:
        dos_smooth = gaussian_filter1d(dos, smoothing)
    else:
        dos_smooth = dos.copy()

    # Calculate derivatives (slope and curvature)
    slope, curvature = finite_diff(dos_smooth, E)

    # DOS value at Fermi level
    N_EF = float(np.interp(0.0, E, dos_smooth))

    # Slope and curvature at Fermi level
    slope_EF = float(np.interp(0.0, E, slope))
    curvature_EF = float(np.interp(0.0, E, curvature))

    # Check for pseudogap (partial gap near Fermi level)
    near_ef = window_mask(E, -0.3, 0.3)
    conduction = window_mask(E, 0.5, 2.0)

    min_near_ef = float(np.min(dos_smooth[near_ef])) if np.any(near_ef) else np.nan
    median_conduction = float(np.median(dos_smooth[conduction])) if np.any(conduction) else np.nan
    pseudogap_score = min_near_ef / (median_conduction + 1e-12) if not np.isnan(min_near_ef) and not np.isnan(median_conduction) else np.nan

    # Check symmetry around Fermi level
    left_side = window_mask(E, -1.0, 0.0)
    right_side = window_mask(E, 0.0, 1.0)

    left_avg = float(np.mean(dos_smooth[left_side])) if np.any(left_side) else np.nan
    right_avg = float(np.mean(dos_smooth[right_side])) if np.any(right_side) else np.nan

    if not np.isnan(left_avg) and not np.isnan(right_avg):
        asymmetry = (right_avg - left_avg) / (right_avg + left_avg + 1e-12)
    else:
        asymmetry = np.nan

    # Find band gap properties
    band_gap, VBM, CBM = largest_gap(E, dos_smooth)

    # Check for superconducting gap
    is_sc, sc_gap_width, sc_peak_ratio = detect_superconducting_gap(E, dos_smooth)

    # Find peaks in valence and conduction bands
    valence_peaks = find_peaks_summary(E, dos_smooth, (-6, 0))
    conduction_peaks = find_peaks_summary(E, dos_smooth, (0, 6))

    # Return all features as a dictionary
    return {
        "N_EF": N_EF,
        "slope_EF": slope_EF,
        "curvature_EF": curvature_EF,
        "pseudogap_score": pseudogap_score,
        "asymmetry": asymmetry,
        "band_gap": band_gap,
        "VBM": VBM,
        "CBM": CBM,
        "is_superconductor": is_sc,
        "sc_gap_width": sc_gap_width,
        "sc_peak_ratio": sc_peak_ratio,
        "valence_peaks": valence_peaks,
        "conduction_peaks": conduction_peaks
    }


def generate_description(features, metal_threshold=0.05, gap_threshold=0.05, sc_threshold=5.0):
    """
    Generate structured description from DOS features.
    Returns:
        - description_dict: dict with individual components (classification, shape, peaks, etc.)
        - description_text: single combined string
    """
    description_parts = []
    desc_dict = {}

    # 1. Material type
    if features["is_superconductor"] and features["sc_peak_ratio"] > sc_threshold:
        gap_meV = features["sc_gap_width"] * 1000
        main_type = "superconducting"
        type_text = (
            f"Superconducting with gap of {gap_meV:.1f} meV and coherence peaks "
            f"({features['sc_peak_ratio']:.1f}× normal DOS)."
        )
    elif features["band_gap"] > gap_threshold and features["N_EF"] < 1e-3:
        main_type = "semiconducting"
        type_text = (
            f"Semiconducting with a band gap of {features['band_gap']:.2f} eV "
            f"(VBM: {features['VBM']:+.2f} eV, CBM: {features['CBM']:+.2f} eV)."
        )
    elif features["N_EF"] >= metal_threshold:
        main_type = "metallic"
        type_text = f"Metallic with high DOS at EF: {features['N_EF']:.2f} states/eV."
    else:
        main_type = "pseudogap"
        type_text = f"Pseudogap-like behavior with low DOS at EF: {features['N_EF']:.3f} states/eV."

    description_parts.append(type_text)
    desc_dict["material_classification"] = main_type

    # 2. Shape near EF
    if features["curvature_EF"] < -0.1:
        shape_text = "V-shaped suppression near EF."
    elif features["curvature_EF"] > 0.1:
        shape_text = "U-shaped DOS around EF."
    else:
        shape_text = "Relatively flat DOS near EF."

    description_parts.append(shape_text)
    desc_dict["overall_dos_shape"] = shape_text

    # 3. Asymmetry
    asymmetry_text = None
    if not np.isnan(features["asymmetry"]):
        if features["asymmetry"] > 0.2:
            asymmetry_text = "Conduction side (0–1 eV) has higher DOS than valence (−1–0 eV)."
        elif features["asymmetry"] < -0.2:
            asymmetry_text = "Valence side (−1–0 eV) has higher DOS than conduction (0–1 eV)."

    if asymmetry_text:
        description_parts.append(asymmetry_text)
    desc_dict["asymmetry_comment"] = asymmetry_text

    # 4. Peak descriptions
    def describe_peaks(peaks, band_name):
        if not peaks:
            return None
        main_peak = peaks[0]
        desc = {
            "main_peak_energy": main_peak["energy"],
            "main_peak_height": main_peak["height"],
            "other_peaks": [p["energy"] for p in peaks[1:]]
        }
        return desc

    desc_dict["valence_band_peaks"] = describe_peaks(features["valence_peaks"], "Valence")
    desc_dict["conduction_band_peaks"] = describe_peaks(features["conduction_peaks"], "Conduction")

    # Turn peaks into text
    def peaks_to_text(peaks, band):
        if not peaks:
            return None
        text = f"{band} band peak at {peaks['main_peak_energy']:+.2f} eV"
        if peaks["other_peaks"]:
            text += f" with additional features at {', '.join(f'{e:+.2f}' for e in peaks['other_peaks'])} eV."
        else:
            text += "."
        return text

    valence_text = peaks_to_text(desc_dict["valence_band_peaks"], "Valence")
    conduction_text = peaks_to_text(desc_dict["conduction_band_peaks"], "Conduction")

    if valence_text:
        description_parts.append(valence_text)
    if conduction_text:
        description_parts.append(conduction_text)

    # 5. Pseudogap score
    if not np.isnan(features["pseudogap_score"]):
        pseudo_text = f"Pseudogap score: {features['pseudogap_score']:.2f}."
        description_parts.append(pseudo_text)
        desc_dict["pseudogap_score"] = features["pseudogap_score"]

    # Return as structured output
    return desc_dict, " ".join(description_parts)




def get_dos_features(mp_dos_data, spin=1, smoothing=1.0):
    """
    Process a Materials Project DOS dictionary and return extracted features + description.

    Args:
        mp_dos_data (dict): Contains 'energies', 'densities', 'efermi'
        spin (int): Spin channel to extract (1 for up, -1 for down)
        smoothing (float): Optional smoothing for feature extraction

    Returns:
        tuple:
            - features (dict): Computed DOS features
            - description (str): Textual interpretation of the DOS
    """
    try:
        # === Extract raw data ===
        energies = np.array(mp_dos_data['energies'])
        densities = mp_dos_data['densities']
        fermi_energy = float(mp_dos_data['efermi'])

        # === Select spin channel ===
        if isinstance(densities, dict):
            if spin in densities:
                dos_values = np.array(densities[spin])
            else:
                dos_values = np.array(list(densities.values())[0])
                print(f"Warning: Spin {spin} not found; using first available spin.")
        else:
            dos_values = np.array(densities)

        # === Shift energy axis ===
        energies_shifted = energies - fermi_energy

        # === Extract features & generate description ===
        features = extract_dos_features(energies_shifted, dos_values, fermi_energy, smoothing=smoothing)
        desc_dict, description = generate_description(features)

        return features, desc_dict,description

    except Exception as error:
        return None, f"ERROR processing DOS: {error}"
