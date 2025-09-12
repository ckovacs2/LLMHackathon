#!/usr/bin/env python3

"""
Author: Ankita Biswas
Date: 09/11/2025
Version: 1.0


Converts Density of States (DOS) data into textual descriptions.
This script analyzes DOS vs. Energy data and generates plain English descriptions
of material properties (metal, insulator, semiconductor, superconductor).

Usage: python dos_to_text.py "path/to/dos_files/*.csv" --out_csv results.csv
"""

import os
import glob
import argparse
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ---------- UTILITY FUNCTIONS ----------

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

# ---------- TEXT DESCRIPTION GENERATION ----------

def generate_description(features, metal_threshold=0.05, gap_threshold=0.05, sc_threshold=5.0):
    """
    Generate English description from DOS features.
    Uses simple rules to interpret the numerical features.
    """
    description_parts = []
    
    # 1. Determine main material type
    if features["is_superconductor"] and features["sc_peak_ratio"] > sc_threshold:
        gap_meV = features["sc_gap_width"] * 1000  # Convert to meV
        description_parts.append(
            f"This material shows a clear superconducting gap of {gap_meV:.1f} meV. "
            f"The deep suppression at the Fermi level and sharp coherence peaks "
            f"(~{features['sc_peak_ratio']:.1f}x normal DOS) are characteristic of superconductivity."
        )
    elif features["band_gap"] > gap_threshold and features["N_EF"] < 1e-3:
        description_parts.append(
            f"Semiconducting or insulating with a band gap of {features['band_gap']:.2f} eV. "
            f"Valence band maximum at {features['VBM']:+.2f} eV, "
            f"conduction band minimum at {features['CBM']:+.2f} eV."
        )
    elif features["N_EF"] >= metal_threshold:
        description_parts.append(f"Metallic with high density of states at Fermi level: {features['N_EF']:.2f} states/eV.")
    else:
        description_parts.append(f"Pseudogap behavior with low density at Fermi level: {features['N_EF']:.3f} states/eV.")
    
    # 2. Describe shape near Fermi level
    if features["curvature_EF"] < -0.1:
        description_parts.append("V-shaped suppression at Fermi level.")
    elif features["curvature_EF"] > 0.1:
        description_parts.append("U-shaped density of states around Fermi level.")
    else:
        description_parts.append("Relatively flat density of states near Fermi level.")
    
    # 3. Describe asymmetry
    if not np.isnan(features["asymmetry"]):
        if features["asymmetry"] > 0.2:
            description_parts.append("Conduction band (0-1 eV) has higher DOS than valence band (-1-0 eV).")
        elif features["asymmetry"] < -0.2:
            description_parts.append("Valence band (-1-0 eV) has higher DOS than conduction band (0-1 eV).")
    
    # 4. Helper function to describe peaks
    def describe_peaks(peaks, band_name):
        if not peaks:
            return None
        
        main_peak = peaks[0]
        description = f"{band_name} band has a strong peak at {main_peak['energy']:+.2f} eV"
        
        if not np.isnan(main_peak['width']):
            description += f" (width ~{main_peak['width']:.2f} eV)"
        
        if len(peaks) > 1:
            other_peaks = ", ".join(f"{p['energy']:+.2f}" for p in peaks[1:])
            description += f" with additional features at {other_peaks} eV."
        else:
            description += "."
        
        return description
    
    # Add valence and conduction peak descriptions
    valence_desc = describe_peaks(features["valence_peaks"], "Valence")
    conduction_desc = describe_peaks(features["conduction_peaks"], "Conduction")
    
    if valence_desc:
        description_parts.append(valence_desc)
    if conduction_desc:
        description_parts.append(conduction_desc)
    
    # 5. Mention pseudogap score if available
    if not np.isnan(features["pseudogap_score"]):
        description_parts.append(f"Pseudogap score: {features['pseudogap_score']:.2f}.")
    
    return " ".join(description_parts)

# ---------- FILE HANDLING ----------

def load_dos_data(filepath):
    """
    Load DOS data from various file formats.
    Supports: .csv, .txt, .npy
    Returns: (energy, dos, fermi_energy)
    """
    extension = os.path.splitext(filepath)[1].lower()
    
    try:
        if extension == ".npy":
            # NPY files should contain a dictionary with E, DOS, and optionally EF
            data_dict = np.load(filepath, allow_pickle=True).item()
            energy = np.array(data_dict["E"])
            dos = np.array(data_dict["DOS"])
            fermi_energy = float(data_dict.get("EF", 0.0))
        else:
            # CSV or TXT files: first column energy, second column DOS
            if extension == ".csv":
                data = np.loadtxt(filepath, delimiter=",")
            else:
                data = np.loadtxt(filepath)  # Whitespace delimiter
            
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"File {filepath} should have at least 2 columns")
            
            energy, dos = data[:, 0], data[:, 1]
            fermi_energy = 0.0  # Assume Fermi level is at 0 if not specified
        
        return energy, dos, fermi_energy
    
    except Exception as error:
        print(f"Error loading {filepath}: {error}")
        return None, None, None

def process_dos_file(filepath, smoothing=1.0):
    """Process a single DOS file and return features and description."""
    E, dos, ef = load_dos_data(filepath)
    
    if E is None or dos is None:
        return None, f"ERROR: Failed to load data from {os.path.basename(filepath)}"
    
    try:
        features = extract_dos_features(E, dos, ef, smoothing)
        description = generate_description(features)
        return features, description
    except Exception as error:
        return None, f"ERROR processing {os.path.basename(filepath)}: {error}"
