"""
Spectrum Image Analyzer
Analyzes light spectrum images to extract wavelength vs intensity data.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SpectrumAnalyzer:
    """Analyzes spectrum images and extracts wavelength-intensity data."""
    
    def __init__(self, image_path=None, wavelength_calibration=None):
        """
        Initialize the spectrum analyzer.

        Args:
            image_path: Optional path to a static spectrum image file. If None,
                        the analyzer can accept frames via `process_and_update`.
            wavelength_calibration: Dict with 'points' (list of [pixel, wavelength_nm])
                                   or 'linear' with [start_nm, end_nm]
        """
        self.image_path = image_path
        self.image = None
        self.spectrum_region = None
        self.intensity_profile = None
        self.wavelengths = None
        self.calibration = wavelength_calibration or {
            'linear': [400, 700]  # Default visible spectrum range
        }
        # Realtime plotting attributes
        self.fig = None
        self.ax = None
        self.line = None
        self.realtime_initialized = False
    
    def load_image(self):
        """Load the spectrum image from file."""
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")
        print(f"Image loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return self.image
    
    def crop_spectrum(self, roi=None, auto_detect=True):
        """
        Crop or isolate the spectral stripe from the image.
        
        Args:
            roi: Region of interest as [x, y, width, height]. If None, uses auto-detection or full image.
            auto_detect: If True, attempts to automatically detect the brightest horizontal band.
        
        Returns:
            Cropped spectrum region
        """
        if roi is not None:
            x, y, w, h = roi
            self.spectrum_region = self.image[y:y+h, x:x+w]
        elif auto_detect:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Sum intensity along horizontal axis to find brightest vertical region
            vertical_profile = np.sum(gray, axis=1)
            
            # Smooth the profile to reduce noise
            kernel_size = max(5, len(vertical_profile) // 50)
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(vertical_profile, kernel, mode='same')
            
            # Find the peak region (brightest band)
            peak_idx = np.argmax(smoothed)
            
            # Define region height as 20% of image height or 50 pixels, whichever is larger
            region_height = max(50, self.image.shape[0] // 5)
            
            # Center the region around the peak
            y_start = max(0, peak_idx - region_height // 2)
            y_end = min(self.image.shape[0], y_start + region_height)
            
            self.spectrum_region = self.image[y_start:y_end, :]
            print(f"Auto-detected spectrum region: rows {y_start}-{y_end}")
        else:
            # Use the full image
            self.spectrum_region = self.image.copy()
        
        return self.spectrum_region
    
    def correct_perspective(self, src_points=None, dst_points=None):
        """
        Correct perspective distortion in the spectrum region.
        
        Args:
            src_points: Four corner points in the source image [[x,y], ...]
            dst_points: Four corner points for the corrected rectangle [[x,y], ...]
        
        Returns:
            Perspective-corrected spectrum region
        """
        if src_points is None or dst_points is None:
            print("No perspective correction applied (points not specified)")
            return self.spectrum_region
        
        src_pts = np.float32(src_points)
        dst_pts = np.float32(dst_points)
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transformation
        height, width = self.spectrum_region.shape[:2]
        self.spectrum_region = cv2.warpPerspective(
            self.spectrum_region, matrix, (width, height)
        )
        print("Perspective correction applied")
        return self.spectrum_region
    
    def extract_intensity(self, method='average', channel='gray'):
        """
        Extract intensity profile by collapsing spectrum along vertical axis.
        
        Args:
            method: 'average', 'max', or 'median' for combining vertical pixels
            channel: 'gray', 'red', 'green', 'blue', or 'all' for RGB
        
        Returns:
            1D array of intensity values (one per horizontal pixel position)
        """
        if self.spectrum_region is None:
            raise ValueError("Spectrum region not set. Run crop_spectrum() first.")
        
        # Select color channel
        if channel == 'gray':
            data = cv2.cvtColor(self.spectrum_region, cv2.COLOR_BGR2GRAY)
        elif channel == 'red':
            data = self.spectrum_region[:, :, 2]  # BGR format
        elif channel == 'green':
            data = self.spectrum_region[:, :, 1]
        elif channel == 'blue':
            data = self.spectrum_region[:, :, 0]
        elif channel == 'all':
            # Use average of all channels
            data = np.mean(self.spectrum_region, axis=2).astype(np.uint8)
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        # Collapse along vertical axis using specified method
        if method == 'average':
            self.intensity_profile = np.mean(data, axis=0)
        elif method == 'max':
            self.intensity_profile = np.max(data, axis=0)
        elif method == 'median':
            self.intensity_profile = np.median(data, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Intensity profile extracted: {len(self.intensity_profile)} points")
        return self.intensity_profile
    
    def calibrate_wavelength(self):
        """
        Convert pixel positions to wavelengths using calibration data.
        
        Returns:
            Array of wavelength values (in nm) corresponding to each pixel
        """
        if self.intensity_profile is None:
            raise ValueError("Intensity profile not extracted yet.")
        
        num_pixels = len(self.intensity_profile)
        
        if 'linear' in self.calibration:
            # Linear calibration: map first pixel to start_nm, last to end_nm
            start_nm, end_nm = self.calibration['linear']
            self.wavelengths = np.linspace(start_nm, end_nm, num_pixels)
            print(f"Linear calibration: {start_nm}-{end_nm} nm")
        
        elif 'points' in self.calibration:
            # Calibration using known reference points
            points = np.array(self.calibration['points'])  # [[pixel, wavelength], ...]
            pixels = points[:, 0]
            wavelengths = points[:, 1]
            
            # Interpolate and EXTRAPOLATE wavelength for all pixel positions
            pixel_positions = np.arange(num_pixels)
            
            # Use linear extrapolation beyond the calibration points
            # This ensures the full spectrum is shown, not just the calibrated region
            self.wavelengths = np.interp(pixel_positions, pixels, wavelengths, 
                                        left=None, right=None)
            
            # For pixels outside calibration range, extrapolate linearly
            if pixel_positions[0] < pixels[0]:
                # Extrapolate to the left using first two calibration points
                if len(pixels) >= 2:
                    slope = (wavelengths[1] - wavelengths[0]) / (pixels[1] - pixels[0])
                    for i in range(num_pixels):
                        if pixel_positions[i] < pixels[0]:
                            self.wavelengths[i] = wavelengths[0] + slope * (pixel_positions[i] - pixels[0])
                        else:
                            break
            
            if pixel_positions[-1] > pixels[-1]:
                # Extrapolate to the right using last two calibration points
                if len(pixels) >= 2:
                    slope = (wavelengths[-1] - wavelengths[-2]) / (pixels[-1] - pixels[-2])
                    for i in range(num_pixels - 1, -1, -1):
                        if pixel_positions[i] > pixels[-1]:
                            self.wavelengths[i] = wavelengths[-1] + slope * (pixel_positions[i] - pixels[-1])
                        else:
                            break
            
            print(f"Point calibration: {len(points)} reference points")
            print(f"  Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")
        
        else:
            raise ValueError("Invalid calibration configuration")
        
        return self.wavelengths
    
    def plot_spectrum(self, title="Spectrum Analysis", save_path=None, 
                     show_peaks=False, figsize=(12, 6)):
        """
        Plot wavelength vs intensity spectrum.
        
        Args:
            title: Plot title
            save_path: If provided, save the plot to this path
            show_peaks: If True, mark prominent peaks
            figsize: Figure size as (width, height)
        """
        if self.wavelengths is None or self.intensity_profile is None:
            raise ValueError("Run calibrate_wavelength() first.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.wavelengths, self.intensity_profile, linewidth=1.5)
        
        # Optionally mark peaks
        if show_peaks:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(
                self.intensity_profile, 
                prominence=np.std(self.intensity_profile)
            )
            plt.plot(self.wavelengths[peaks], self.intensity_profile[peaks], 
                    'rx', markersize=10, label='Peaks')
            plt.legend()
        
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Intensity (arbitrary units)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

    # ---------------------- Real-time support -------------------------------
    def init_realtime_plot(self, title="Realtime Spectrum", figsize=(12, 6)):
        """Initialize an interactive matplotlib plot for live updates."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.line, = self.ax.plot([], [], linewidth=1.5)
        self.ax.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax.set_ylabel('Intensity (arbitrary units)', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()
        plt.show(block=False)
        self.realtime_initialized = True

    def update_realtime_plot(self):
        """Update the realtime plot with the latest wavelength and intensity data."""
        if not self.realtime_initialized:
            self.init_realtime_plot()

        if self.wavelengths is None or self.intensity_profile is None:
            return

        self.line.set_data(self.wavelengths, self.intensity_profile)
        self.ax.relim()
        self.ax.autoscale_view()
        # Use draw_idle + pause to keep UI responsive
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close_realtime(self):
        """Close the realtime plot cleanly."""
        try:
            plt.ioff()
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass
        self.realtime_initialized = False

    def process_and_update(self, frame, roi=None, auto_detect=True,
                           intensity_method='average', channel='gray',
                           update_plot=True):
        """Process a single video frame and optionally update realtime plot.

        Args:
            frame: BGR image frame (numpy array) from camera or video.
            roi: Region of interest [x, y, width, height]
            auto_detect: Whether to auto-detect spectrum stripe
            intensity_method: 'average', 'max', or 'median'
            channel: Color channel to analyze ('gray','red','green','blue','all')
            update_plot: If True, update the interactive plot

        Returns:
            dict with 'wavelengths' and 'intensities'
        """
        # Accept frames directly (do not try to load from disk)
        self.image = frame.copy()
        self.crop_spectrum(roi=roi, auto_detect=auto_detect)
        self.extract_intensity(method=intensity_method, channel=channel)
        self.calibrate_wavelength()

        if update_plot:
            self.update_realtime_plot()

        return {'wavelengths': self.wavelengths, 'intensities': self.intensity_profile}
    
    def save_data(self, csv_path):
        """
        Save wavelength and intensity data to CSV file.
        
        Args:
            csv_path: Path for output CSV file
        """
        if self.wavelengths is None or self.intensity_profile is None:
            raise ValueError("No data to save. Run analysis first.")
        
        data = np.column_stack((self.wavelengths, self.intensity_profile))
        np.savetxt(
            csv_path, 
            data, 
            delimiter=',', 
            header='Wavelength(nm),Intensity',
            comments='',
            fmt='%.3f'
        )
        print(f"Data saved to: {csv_path}")
    
    def analyze(self, roi=None, auto_detect=True, intensity_method='average',
                channel='gray', plot=True, save_plot=None, save_csv=None):
        """
        Run complete analysis pipeline.
        
        Args:
            roi: Region of interest [x, y, width, height]
            auto_detect: Auto-detect spectrum region
            intensity_method: 'average', 'max', or 'median'
            channel: Color channel to analyze
            plot: Whether to display the plot
            save_plot: Path to save plot image
            save_csv: Path to save CSV data
        
        Returns:
            Dictionary with wavelengths and intensities
        """
        self.load_image()
        self.crop_spectrum(roi=roi, auto_detect=auto_detect)
        self.extract_intensity(method=intensity_method, channel=channel)
        self.calibrate_wavelength()
        
        if plot:
            self.plot_spectrum(save_path=save_plot)
        
        if save_csv:
            self.save_data(save_csv)
        
        return {
            'wavelengths': self.wavelengths,
            'intensities': self.intensity_profile
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # CONFIGURATION - Modify these parameters for your specific use case
    # -------------------------------------------------------------------------
    
    # Path to your spectrum image
    IMAGE_PATH = "test_emission_spectrum.jpg"
    
    # Wavelength calibration options:
    # Option 1: Linear calibration (assumes spectrum spans from start to end wavelength)
    calibration_config = {
        'linear': [400, 700]  # violet (400nm) to red (700nm)
    }
    
    # Option 2: Point calibration (if you know specific wavelength positions in pixels)
    # Uncomment and modify if you have reference points:
    # calibration_config = {
    #     'points': [
    #         [100, 450],   # pixel 100 = 450nm (blue)
    #         [500, 550],   # pixel 500 = 550nm (green)
    #         [900, 650]    # pixel 900 = 650nm (red)
    #     ]
    # }
    
    # Region of interest (optional): [x, y, width, height]
    # Set to None to use auto-detection or full image
    ROI = None  # Example: [50, 200, 1000, 100]
    
    # Analysis parameters
    AUTO_DETECT = True           # Automatically find brightest horizontal band
    INTENSITY_METHOD = 'average' # 'average', 'max', or 'median'
    COLOR_CHANNEL = 'gray'       # 'gray', 'red', 'green', 'blue', or 'all'
    
    # Output options
    SAVE_PLOT = "spectrum_plot.png"  # Set to None to not save
    SAVE_CSV = "spectrum_data.csv"   # Set to None to not save
    
    # -------------------------------------------------------------------------
    # RUN ANALYSIS
    # -------------------------------------------------------------------------
    
    try:
        # Create analyzer instance
        analyzer = SpectrumAnalyzer(IMAGE_PATH, wavelength_calibration=calibration_config)
        
        # Run complete analysis
        results = analyzer.analyze(
            roi=ROI,
            auto_detect=AUTO_DETECT,
            intensity_method=INTENSITY_METHOD,
            channel=COLOR_CHANNEL,
            plot=True,
            save_plot=SAVE_PLOT,
            save_csv=SAVE_CSV
        )
        
        print("\nAnalysis complete!")
        print(f"Wavelength range: {results['wavelengths'][0]:.1f} - {results['wavelengths'][-1]:.1f} nm")
        print(f"Number of data points: {len(results['wavelengths'])}")
        print(f"Intensity range: {results['intensities'].min():.1f} - {results['intensities'].max():.1f}")
        
    except FileNotFoundError:
        print(f"Error: Image file '{IMAGE_PATH}' not found.")
        print("Please update IMAGE_PATH with the correct path to your spectrum image.")
    except Exception as e:
        print(f"Error during analysis: {e}")