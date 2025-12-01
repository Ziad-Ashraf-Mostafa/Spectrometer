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
    
    # Generate continuous spectrum (like from a prism or diffraction grating)
    print("\n1. Generating continuous spectrum...")
    metadata_continuous = create_test_spectrum(
        spectrum_type='continuous',
        output_path='test_continuous_spectrum.jpg',
        show_preview=True
    )
    
    # Generate emission line spectrum (like from a gas discharge tube)
    print("\n2. Generating emission line spectrum...")
    metadata_emission = create_test_spectrum(
        spectrum_type='emission',
        output_path='test_emission_spectrum.jpg',
        show_preview=True
    )
    
    print("\n" + "=" * 70)
    print("VERIFICATION INSTRUCTIONS")
    print("=" * 70)
    print("\nTo verify the spectrum analyzer:")
    print("\n1. Use the generated test images with the analyzer:")
    print("   - test_continuous_spectrum.jpg (should show smooth rainbow)")
    print("   - test_emission_spectrum.jpg (should show discrete peaks)")
    print("\n2. Expected results for continuous spectrum:")
    print("   - Wavelength range: 400-700 nm")
    print("   - Intensity should be relatively uniform across spectrum")
    print("   - Colors: violet -> blue -> cyan -> green -> yellow -> red")
    print("\n3. Expected results for emission spectrum:")
    print("   - Should show distinct peaks at:")
    for wl, intensity in metadata_emission['emission_lines']:
        print(f"     {wl} nm (relative intensity: {intensity:.1f})")
    print("\n4. Run the analyzer and check if:")
    print("   - Auto-detection finds the spectrum stripe")
    print("   - Wavelength calibration matches 400-700 nm range")
    print("   - Peak positions match expected wavelengths (for emission)")