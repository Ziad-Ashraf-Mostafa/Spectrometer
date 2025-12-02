"""
Generate synthetic spectrum images for testing the spectrum analyzer.
This creates images with known wavelength-to-color mappings and intensity patterns.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def wavelength_to_rgb(wavelength):
    """
    Convert wavelength (in nm) to approximate RGB values.
    Based on visible spectrum (380-750 nm).
    
    Args:
        wavelength: Wavelength in nanometers (380-750)
    
    Returns:
        RGB tuple (0-255 range)
    """
    wavelength = float(wavelength)
    
    if wavelength < 380:
        R, G, B = 0, 0, 0
    elif 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 750:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R, G, B = 0, 0, 0
    
    # Intensity correction for visibility
    if wavelength < 380:
        factor = 0.0
    elif 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength < 700:
        factor = 1.0
    elif 700 <= wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 0.0
    
    R = int(255 * R * factor)
    G = int(255 * G * factor)
    B = int(255 * B * factor)
    
    return (R, G, B)


def generate_continuous_spectrum(width=1200, height=100, wavelength_range=(400, 700)):
    """
    Generate a synthetic continuous spectrum image (like from a prism).
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        wavelength_range: Tuple of (start_wavelength, end_wavelength) in nm
    
    Returns:
        RGB image array
    """
    start_wl, end_wl = wavelength_range
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient from start to end wavelength
    for x in range(width):
        wavelength = start_wl + (end_wl - start_wl) * x / width
        r, g, b = wavelength_to_rgb(wavelength)
        image[:, x] = [b, g, r]  # OpenCV uses BGR
    
    return image


def generate_emission_spectrum(width=1200, height=100, emission_lines=None,
                               wavelength_range=(400, 700), background_intensity=20):
    """
    Generate a synthetic emission line spectrum (like from a gas discharge).
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        emission_lines: List of (wavelength_nm, intensity_0_to_1) tuples
        wavelength_range: Tuple of (start_wavelength, end_wavelength) in nm
        background_intensity: Background brightness (0-255)
    
    Returns:
        RGB image array
    """
    if emission_lines is None:
        # Default: Hydrogen Balmer lines (approximate)
        emission_lines = [
            (434, 0.6),   # H-gamma (violet)
            (486, 0.8),   # H-beta (cyan)
            (656, 1.0),   # H-alpha (red)
        ]
    
    start_wl, end_wl = wavelength_range
    image = np.ones((height, width, 3), dtype=np.uint8) * background_intensity
    
    # Add emission lines
    for wavelength, intensity in emission_lines:
        if start_wl <= wavelength <= end_wl:
            # Calculate pixel position
            x_pos = int((wavelength - start_wl) / (end_wl - start_wl) * width)
            
            # Get color for this wavelength
            r, g, b = wavelength_to_rgb(wavelength)
            
            # Create Gaussian profile for the line
            line_width = width * 0.015  # ~1.5% of image width
            x_coords = np.arange(width)
            gaussian = np.exp(-0.5 * ((x_coords - x_pos) / line_width) ** 2)
            
            # Apply line to image
            for c in range(3):
                color_intensity = [b, g, r][c] * intensity
                line_profile = background_intensity + color_intensity * gaussian
                image[:, :, c] = np.maximum(image[:, :, c], line_profile.astype(np.uint8))
    
    return image


def generate_spectrum_with_noise(base_image, noise_level=0.1):
    """
    Add realistic noise to spectrum image.
    
    Args:
        base_image: Original spectrum image
        noise_level: Noise strength (0.0 to 1.0)
    
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, 255 * noise_level, base_image.shape)
    noisy_image = np.clip(base_image.astype(float) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_background_and_borders(spectrum_image, total_height=400):
    """
    Add black borders above and below the spectrum (realistic photo).
    
    Args:
        spectrum_image: The spectrum stripe image
        total_height: Total height of output image
    
    Returns:
        Image with borders
    """
    spec_height = spectrum_image.shape[0]
    width = spectrum_image.shape[1]
    
    # Create black canvas
    full_image = np.zeros((total_height, width, 3), dtype=np.uint8)
    
    # Place spectrum in the middle
    y_offset = (total_height - spec_height) // 2
    full_image[y_offset:y_offset + spec_height, :] = spectrum_image
    
    return full_image


def create_test_spectrum(spectrum_type='continuous', output_path='test_spectrum.jpg',
                        show_preview=True):
    """
    Create a test spectrum image with known properties.
    
    Args:
        spectrum_type: 'continuous' or 'emission'
        output_path: Where to save the image
        show_preview: Whether to display the image
    
    Returns:
        Dictionary with image metadata
    """
    # Generate base spectrum
    if spectrum_type == 'continuous':
        spectrum = generate_continuous_spectrum(
            width=1200, 
            height=80, 
            wavelength_range=(400, 700)
        )
        title = "Continuous Spectrum (400-700 nm)"
    elif spectrum_type == 'emission':
        # Hydrogen-like emission lines
        emission_lines = [
            (410, 0.5),   # Violet
            (434, 0.7),   # Blue-violet
            (486, 0.9),   # Cyan
            (589, 0.6),   # Yellow (like sodium D-line)
            (656, 1.0),   # Red
        ]
        spectrum = generate_emission_spectrum(
            width=1200,
            height=80,
            emission_lines=emission_lines,
            wavelength_range=(400, 700)
        )
        title = "Emission Line Spectrum"
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}")
    
    # Add some realistic noise
    spectrum = generate_spectrum_with_noise(spectrum, noise_level=0.05)
    
    # Add borders to simulate real photo
    full_image = add_background_and_borders(spectrum, total_height=300)
    
    # Save image
    cv2.imwrite(output_path, full_image)
    print(f"Test spectrum saved to: {output_path}")
    print(f"Image size: {full_image.shape[1]}x{full_image.shape[0]} pixels")
    print(f"Wavelength range: 400-700 nm")
    
    # Display preview
    if show_preview:
        plt.figure(figsize=(12, 4))
        plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Return metadata for verification
    metadata = {
        'type': spectrum_type,
        'wavelength_start': 400,
        'wavelength_end': 700,
        'width_pixels': full_image.shape[1],
        'height_pixels': full_image.shape[0],
        'spectrum_height': 80,
        'path': output_path
    }
    
    if spectrum_type == 'emission':
        metadata['emission_lines'] = emission_lines
    
    return metadata


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPECTRUM TEST IMAGE GENERATOR")
    print("=" * 70)
    
    # Generate continuous spectrum with ONLY green and red bands visible
    print("\n1. Generating CONTINUOUS spectrum with GREEN and RED bands only...")
    
    width = 1200
    height = 80
    start_wl = 400
    end_wl = 700
    
    # Create base black image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define green and red continuous ranges
    green_range = (510, 580)  # Green band
    red_range = (620, 700)    # Red band
    
    # Fill in the spectrum
    for x in range(width):
        wavelength = start_wl + (end_wl - start_wl) * x / width
        
        # Check if wavelength is in green or red range
        if green_range[0] <= wavelength <= green_range[1]:
            # Green band - continuous gradient
            r, g, b = wavelength_to_rgb(wavelength)
            image[:, x] = [b, g, r]  # OpenCV uses BGR
        elif red_range[0] <= wavelength <= red_range[1]:
            # Red band - continuous gradient
            r, g, b = wavelength_to_rgb(wavelength)
            image[:, x] = [b, g, r]  # OpenCV uses BGR
        else:
            # Black for all other wavelengths (violet, blue, cyan, orange)
            image[:, x] = [0, 0, 0]
    
    # Add slight noise for realism
    spectrum_combined = generate_spectrum_with_noise(image, noise_level=0.03)
    full_image_combined = add_background_and_borders(spectrum_combined, total_height=300)
    cv2.imwrite('test_green_red_continuous.jpg', full_image_combined)
    
    print(f"✓ Green+Red continuous spectrum saved to: test_green_red_continuous.jpg")
    print(f"  Full wavelength range: 400-700 nm")
    print(f"  Visible continuous bands:")
    print(f"    - GREEN band: 510-580 nm (continuous gradient)")
    print(f"    - RED band: 620-700 nm (continuous gradient)")
    print(f"  Dark/Black regions: 400-510 nm, 580-620 nm (violet, blue, cyan, orange)")
    print(f"  Calibration points suggestion:")
    print(f"    - Left edge (pixel ~0): 400 nm (black region)")
    print(f"    - Green start: 510 nm")
    print(f"    - Green middle: 545 nm")
    print(f"    - Green end: 580 nm")
    print(f"    - Red start: 620 nm")
    print(f"    - Red middle: 660 nm")
    print(f"    - Right edge (pixel ~1200): 700 nm")
    
    print("\n" + "=" * 70)
    print("CALIBRATION INSTRUCTIONS")
    print("=" * 70)
    print("\nFor GREEN+RED continuous spectrum (test_green_red_continuous.jpg):")
    print("  1. Load the image in the analyzer")
    print("  2. Enter calibration mode (Ctrl+M)")
    print("  3. Click at the LEFT edge (black region) and enter: 400")
    print("  4. Click at the start of green band and enter: 510")
    print("  5. Click at the middle of green band and enter: 545")
    print("  6. Click at the start of red band and enter: 620")
    print("  7. Click at the RIGHT edge and enter: 700")
    print("  8. Expected: Continuous green band (510-580nm) and red band (620-700nm)")
    print("              Black in violet, blue, cyan, and orange regions")
    
    print("\n" + "=" * 70)
    print("SPECTRUM CHARACTERISTICS")
    print("=" * 70)
    print("\nGreen+Red continuous spectrum (400-700 nm full range):")
    print("  - Full visible spectrum range (400-700 nm)")
    print("  - GREEN continuous band: 510-580 nm")
    print("    (cyan-green → pure green → yellow-green → yellow)")
    print("  - RED continuous band: 620-700 nm")
    print("    (orange-red → red → deep red → far red)")
    print("  - BLACK regions: 400-510 nm (violet, blue)")
    print("                  580-620 nm (yellow-orange gap)")
    print("  - Simulates filtered continuous spectrum or bandpass filters")
    
    print("\n✓ Continuous green+red spectrum image generated successfully!")
    print("=" * 70)