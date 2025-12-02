import sys
import os
import glob
import re
import cv2
import numpy as np
from SpectrumAnalyzer import SpectrumAnalyzer

# PySide6 GUI imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QToolBar, QSplitter, QSizePolicy, QFrame,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox,
    QFileDialog, QSlider, QGroupBox
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QImage, QPixmap, QAction
import json

# Matplotlib for embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Polygon


class VideoLabel(QLabel):
    """QLabel that displays video frames and supports drawing a crop rectangle."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self.start_pos = None
        self.end_pos = None
        self.drawing = False
        self.show_rect = False
        self.rect = None  # QRectF in widget coordinates

    def mousePressEvent(self, event):
        if self.show_rect and event.button() == Qt.LeftButton:
            self.start_pos = event.position().toPoint()
            self.end_pos = self.start_pos
            self.drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.end_pos = event.position().toPoint()
            self.drawing = False
            self._update_rect()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_rect and (self.drawing or self.rect is not None):
            from PySide6.QtGui import QPainter, QPen
            painter = QPainter(self)
            pen = QPen(Qt.yellow)
            pen.setWidth(2)
            painter.setPen(pen)
            if self.drawing and self.start_pos and self.end_pos:
                r = QRectF(self.start_pos, self.end_pos)
                painter.drawRect(r)
            elif self.rect is not None:
                painter.drawRect(self.rect)
            painter.end()

    def _update_rect(self):
        if self.start_pos and self.end_pos:
            x1 = min(self.start_pos.x(), self.end_pos.x())
            y1 = min(self.start_pos.y(), self.end_pos.y())
            x2 = max(self.start_pos.x(), self.end_pos.x())
            y2 = max(self.start_pos.y(), self.end_pos.y())
            self.rect = QRectF(x1, y1, x2 - x1, y2 - y1)

    def clear_rect(self):
        self.rect = None
        self.update()

    def get_roi_on_frame(self, frame_shape):
        """Convert the rectangle in widget coordinates to ROI in frame pixel coordinates.
        Returns [x, y, w, h] in frame coordinates or None if no rect.
        """
        if self.rect is None:
            return None
        w_widget = self.width()
        h_widget = self.height()
        img_h, img_w = frame_shape[0], frame_shape[1]

        # The QLabel scaledContents stretches image to widget size; map accordingly
        sx = img_w / w_widget if w_widget > 0 else 1
        sy = img_h / h_widget if h_widget > 0 else 1

        x = int(self.rect.x() * sx)
        y = int(self.rect.y() * sy)
        w = int(self.rect.width() * sx)
        h = int(self.rect.height() * sy)

        # clamp
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        return [x, y, w, h]


class MainWindow(QMainWindow):
    def __init__(self, video_url):
        super().__init__()
        self.setWindowTitle("Spectrometer Live Analyzer")
        self.video_url = video_url
        self.cap = cv2.VideoCapture(self.video_url)
        
        # Calibration state
        self.calibration_mode = False
        self.calibration_points = []  # List of [pixel_index, wavelength_nm]
        self.calibration_markers = []  # matplotlib artists for calibration points
        self.load_calibration()  # Load saved calibration if exists
        
        # Flip state
        self.flip_enabled = True  # Default to flipped (current behavior)
        
        # HSV filter thresholds (S and V channels, H is fixed per range)
        self.lower_s = 40
        self.lower_v = 40
        self.upper_s = 255
        self.upper_v = 255
        
        # Initialize analyzer with loaded or default calibration
        if len(self.calibration_points) >= 2:
            calib_config = {'points': self.calibration_points}
        else:
            calib_config = {'linear': [400, 700]}
        self.analyzer = SpectrumAnalyzer(image_path=None, wavelength_calibration=calib_config, verbose=False)

        # Main widgets
        self.canvas = FigureCanvas(Figure(figsize=(8, 5)))
        self.ax = self.canvas.figure.subplots()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Video display area
        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setMaximumSize(640, 480)
        self.video_label.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")

        # Toolbar
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        # Pause / Resume button -> toggles pause
        self.pause_action = QAction("⏸ Pause", self)
        self.pause_action.setCheckable(True)
        self.pause_action.triggered.connect(self.toggle_pause)
        toolbar.addAction(self.pause_action)
        
        # Flip toggle button
        self.flip_action = QAction("🔄 Flip", self)
        self.flip_action.setCheckable(True)
        self.flip_action.setChecked(True)  # Start with flip enabled
        self.flip_action.triggered.connect(self.toggle_flip)
        toolbar.addAction(self.flip_action)

        # Snapshot button
        snap_action = QAction("📷 Snapshot", self)
        snap_action.triggered.connect(self.save_snapshot)
        toolbar.addAction(snap_action)

        # Toggle crop box
        self.crop_action = QAction("✂ Crop Box", self)
        self.crop_action.setCheckable(True)
        self.crop_action.triggered.connect(self.toggle_crop_box)
        toolbar.addAction(self.crop_action)

        # Fold/unfold video button in toolbar
        self.fold_action = QAction("🔽 Fold Video", self)
        self.fold_action.setCheckable(True)
        self.fold_action.triggered.connect(self.toggle_fold)
        toolbar.addAction(self.fold_action)
        
        toolbar.addSeparator()
        
        # Calibration mode button
        self.calib_action = QAction("📐 Calibrate", self)
        self.calib_action.setCheckable(True)
        self.calib_action.triggered.connect(self.toggle_calibration)
        toolbar.addAction(self.calib_action)
        
        # Clear calibration button
        clear_calib_action = QAction("🗑️ Clear Calib", self)
        clear_calib_action.triggered.connect(self.clear_calibration)
        toolbar.addAction(clear_calib_action)
        
        # Save calibration button
        save_calib_action = QAction("💾 Save Calib", self)
        save_calib_action.triggered.connect(self.save_calibration)
        toolbar.addAction(save_calib_action)
        
        # Load calibration button
        load_calib_action = QAction("📂 Load Calib", self)
        load_calib_action.triggered.connect(self.load_calibration_dialog)
        toolbar.addAction(load_calib_action)

        # Layout using splitter so user can resize video vs graph
        splitter = QSplitter(Qt.Horizontal)

        # Left side: Graph (main display)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(self.canvas)

        # Right side: Video display and threshold controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(self.video_label)
        
        # HSV Threshold Controls
        threshold_group = QGroupBox("HSV Filter Thresholds")
        threshold_layout = QVBoxLayout()
        
        # Lower Saturation
        lower_s_layout = QHBoxLayout()
        lower_s_label = QLabel("Lower S:")
        self.lower_s_slider = QSlider(Qt.Horizontal)
        self.lower_s_slider.setRange(0, 255)
        self.lower_s_slider.setValue(self.lower_s)
        self.lower_s_slider.setTickPosition(QSlider.TicksBelow)
        self.lower_s_slider.setTickInterval(25)
        self.lower_s_value_label = QLabel(str(self.lower_s))
        self.lower_s_value_label.setMinimumWidth(30)
        self.lower_s_slider.valueChanged.connect(self._update_lower_s)
        lower_s_layout.addWidget(lower_s_label)
        lower_s_layout.addWidget(self.lower_s_slider)
        lower_s_layout.addWidget(self.lower_s_value_label)
        
        # Lower Value
        lower_v_layout = QHBoxLayout()
        lower_v_label = QLabel("Lower V:")
        self.lower_v_slider = QSlider(Qt.Horizontal)
        self.lower_v_slider.setRange(0, 255)
        self.lower_v_slider.setValue(self.lower_v)
        self.lower_v_slider.setTickPosition(QSlider.TicksBelow)
        self.lower_v_slider.setTickInterval(25)
        self.lower_v_value_label = QLabel(str(self.lower_v))
        self.lower_v_value_label.setMinimumWidth(30)
        self.lower_v_slider.valueChanged.connect(self._update_lower_v)
        lower_v_layout.addWidget(lower_v_label)
        lower_v_layout.addWidget(self.lower_v_slider)
        lower_v_layout.addWidget(self.lower_v_value_label)
        
        # Upper Saturation
        upper_s_layout = QHBoxLayout()
        upper_s_label = QLabel("Upper S:")
        self.upper_s_slider = QSlider(Qt.Horizontal)
        self.upper_s_slider.setRange(0, 255)
        self.upper_s_slider.setValue(self.upper_s)
        self.upper_s_slider.setTickPosition(QSlider.TicksBelow)
        self.upper_s_slider.setTickInterval(25)
        self.upper_s_value_label = QLabel(str(self.upper_s))
        self.upper_s_value_label.setMinimumWidth(30)
        self.upper_s_slider.valueChanged.connect(self._update_upper_s)
        upper_s_layout.addWidget(upper_s_label)
        upper_s_layout.addWidget(self.upper_s_slider)
        upper_s_layout.addWidget(self.upper_s_value_label)
        
        # Upper Value
        upper_v_layout = QHBoxLayout()
        upper_v_label = QLabel("Upper V:")
        self.upper_v_slider = QSlider(Qt.Horizontal)
        self.upper_v_slider.setRange(0, 255)
        self.upper_v_slider.setValue(self.upper_v)
        self.upper_v_slider.setTickPosition(QSlider.TicksBelow)
        self.upper_v_slider.setTickInterval(25)
        self.upper_v_value_label = QLabel(str(self.upper_v))
        self.upper_v_value_label.setMinimumWidth(30)
        self.upper_v_slider.valueChanged.connect(self._update_upper_v)
        upper_v_layout.addWidget(upper_v_label)
        upper_v_layout.addWidget(self.upper_v_slider)
        upper_v_layout.addWidget(self.upper_v_value_label)
        
        threshold_layout.addLayout(lower_s_layout)
        threshold_layout.addLayout(lower_v_layout)
        threshold_layout.addLayout(upper_s_layout)
        threshold_layout.addLayout(upper_v_layout)
        threshold_group.setLayout(threshold_layout)
        right_layout.addWidget(threshold_group)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 300])  # Graph gets more space

        self.setCentralWidget(splitter)

        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(30)  # ~33 fps
        self.paused = False

        # store last frame
        self.current_frame = None

        # snapshot counter
        self.snapshot_index = self._find_next_snapshot_index()

        # init plot line and gradient
        self.line, = self.ax.plot([], [], color='black', linewidth=1.5)
        self.gradient_image = None
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity (a.u.)', fontsize=11)
        self.ax.set_title('Live Spectrum Analysis', fontsize=13)
        self.ax.grid(True, alpha=0.3)
        
        # Performance tuning: throttle plot updates to reduce redraw overhead
        self.plot_update_interval = 5  # update plot every N frames (set to 1 for every frame)
        self._frame_counter = 0
        
        # Cache for gradient (avoid recreating every frame)
        self._gradient_cache = None
        self._cached_ylim = None
        
        # Pre-allocate buffers to avoid repeated allocation
        self.hsv_buffer = None
        
        # Connect click event for calibration
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Update title with calibration status
        self.update_calibration_display()

    def _find_next_snapshot_index(self):
        files = glob.glob(os.path.join(os.getcwd(), "snappedshot*.png"))
        nums = []
        for f in files:
            name = os.path.basename(f)
            m = re.search(r"snappedshot(\d+)\.png", name)
            if m:
                nums.append(int(m.group(1)))
        return max(nums) + 1 if nums else 1

    def toggle_pause(self, checked):
        # If paused (checked true) we stop updating analyzer (freeze graph)
        if checked:
            self.paused = True
            self.pause_action.setText("▶ Resume")
        else:
            self.paused = False
            self.pause_action.setText("⏸ Pause")
    
    def toggle_flip(self, checked):
        """Toggle horizontal flip of video frames."""
        self.flip_enabled = checked
    
    def _update_lower_s(self, value):
        self.lower_s = value
        self.lower_s_value_label.setText(str(value))
    
    def _update_lower_v(self, value):
        self.lower_v = value
        self.lower_v_value_label.setText(str(value))
    
    def _update_upper_s(self, value):
        self.upper_s = value
        self.upper_s_value_label.setText(str(value))
    
    def _update_upper_v(self, value):
        self.upper_v = value
        self.upper_v_value_label.setText(str(value))

    def save_snapshot(self):
        if self.current_frame is None:
            return
        fname = f"snappedshot{self.snapshot_index:02d}.png"
        cv2.imwrite(fname, self.current_frame)
        print(f"Saved snapshot: {fname}")
        self.snapshot_index += 1

    def toggle_crop_box(self, checked):
        self.video_label.show_rect = checked
        if not checked:
            self.video_label.clear_rect()

    def toggle_fold(self, checked):
        # If checked -> hide video (fold), else show
        widget = self.centralWidget()
        if not isinstance(widget, QSplitter):
            return
        if checked:
            # collapse right widget
            widget.setSizes([10000, 0])
            self.fold_action.setText("🔼 Unfold Video")
        else:
            widget.setSizes([900, 300])
            self.fold_action.setText("🔽 Fold Video")
    
    def toggle_calibration(self, checked):
        """Enter/exit calibration mode."""
        self.calibration_mode = checked
        if checked:
            self.calib_action.setText("✅ Calibrating...")
            self.paused = True  # Pause during calibration
            self.pause_action.setChecked(True)
            self.pause_action.setText("▶ Resume")
            QMessageBox.information(self, "Calibration Mode",
                "Click on spectral peaks in the graph to mark known wavelengths.\n"
                "You'll be prompted to enter the wavelength for each point.\n"
                "Add at least 2 reference points for accurate calibration.")
        else:
            self.calib_action.setText("📐 Calibrate")
            self.update_calibration_display()
    
    def on_plot_click(self, event):
        """Handle clicks on the plot during calibration mode."""
        if not self.calibration_mode or event.inaxes != self.ax:
            return
        
        # Get pixel index from x-coordinate (wavelength displayed)
        if self.analyzer.wavelengths is None:
            return
        
        # Find closest pixel to clicked x position
        clicked_wavelength = event.xdata
        pixel_idx = np.argmin(np.abs(self.analyzer.wavelengths - clicked_wavelength))
        
        # Prompt user for actual wavelength
        dialog = QDialog(self)
        dialog.setWindowTitle("Enter Wavelength")
        layout = QFormLayout(dialog)
        
        wavelength_input = QLineEdit()
        wavelength_input.setPlaceholderText("e.g., 546.1 (mercury green)")
        layout.addRow("Wavelength (nm):", wavelength_input)
        
        info_label = QLabel(f"Pixel index: {pixel_idx}\nCurrent display: {clicked_wavelength:.1f} nm")
        layout.addRow(info_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            try:
                wavelength = float(wavelength_input.text())
                if 200 <= wavelength <= 1100:  # Reasonable range
                    # Add or update calibration point
                    # Remove existing point at similar pixel if any
                    self.calibration_points = [p for p in self.calibration_points 
                                               if abs(p[0] - pixel_idx) > 5]
                    # Convert to standard Python types to avoid JSON serialization issues
                    self.calibration_points.append([int(pixel_idx), float(wavelength)])
                    self.calibration_points.sort(key=lambda p: p[0])  # Sort by pixel
                    
                    # Update analyzer calibration and recalibrate wavelengths
                    if len(self.calibration_points) >= 2:
                        self.analyzer.calibration = {'points': self.calibration_points}
                        # Force recalibration of wavelengths
                        if self.analyzer.intensity_profile is not None:
                            self.analyzer.calibrate_wavelength()
                    
                    self.update_calibration_display()
                    print(f"Added calibration point: pixel {pixel_idx} = {wavelength} nm")
                    print(f"Total calibration points: {len(self.calibration_points)}")
                else:
                    QMessageBox.warning(self, "Invalid Wavelength", 
                                       "Wavelength must be between 200-1100 nm")
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number")
    
    def update_calibration_display(self):
        """Update the graph to show calibration markers."""
        # Remove old markers
        for marker in self.calibration_markers:
            try:
                marker.remove()
            except:
                pass
        self.calibration_markers.clear()
        
        # Add new markers at the ACTUAL wavelengths (from user input)
        if len(self.calibration_points) > 0:
            for pixel_idx, actual_wavelength in self.calibration_points:
                # Draw vertical line at the ACTUAL wavelength the user specified
                # This shows where that spectral line should be
                if self.analyzer.intensity_profile is not None and 0 <= pixel_idx < len(self.analyzer.intensity_profile):
                    # Get intensity at that pixel
                    intensity = self.analyzer.intensity_profile[pixel_idx]
                    
                    # Draw marker at the actual wavelength
                    vline = self.ax.axvline(actual_wavelength, 
                                           color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                    text = self.ax.text(actual_wavelength, 
                                       intensity if intensity > 0 else 100,
                                       f'{actual_wavelength:.1f}nm', 
                                       color='red', fontsize=9, ha='center',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    self.calibration_markers.extend([vline, text])
        
        # Update title
        n_points = len(self.calibration_points)
        if n_points >= 2:
            calib_type = f"Calibrated ({n_points} points)"
        elif n_points == 1:
            calib_type = "Partial calibration (need 1+ more)"
        else:
            calib_type = "Linear (default)"
        self.ax.set_title(f'Live Spectrum Analysis - {calib_type}', fontsize=13)
        
        # Force update the plot with new calibrated wavelengths
        if self.analyzer.wavelengths is not None and self.analyzer.intensity_profile is not None:
            self._update_plot(self.analyzer.wavelengths, self.analyzer.intensity_profile)
        else:
            self.canvas.draw_idle()
    
    def clear_calibration(self):
        """Clear all calibration points and reset to default."""
        reply = QMessageBox.question(self, 'Clear Calibration',
                                    'Are you sure you want to clear all calibration points?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.calibration_points.clear()
            self.analyzer.calibration = {'linear': [400, 700]}
            # Recalibrate with linear default
            if self.analyzer.intensity_profile is not None:
                self.analyzer.calibrate_wavelength()
            self.update_calibration_display()
            print("Calibration cleared, using default linear calibration")
    
    def save_calibration(self):
        """Save calibration points to a JSON file."""
        if len(self.calibration_points) < 2:
            QMessageBox.warning(self, "Insufficient Data",
                               "Need at least 2 calibration points to save.")
            return
        
        calib_file = os.path.join(os.getcwd(), "spectrometer_calibration.json")
        # Convert numpy types to standard Python types for JSON serialization
        calibration_data = [[int(p[0]), float(p[1])] for p in self.calibration_points]
        data = {
            'calibration_points': calibration_data,
            'timestamp': str(np.datetime64('now')),
            'num_points': len(calibration_data)
        }
        
        try:
            with open(calib_file, 'w') as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Calibration Saved",
                                   f"Calibration saved to:\n{calib_file}")
            print(f"Calibration saved: {len(self.calibration_points)} points")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save calibration:\n{e}")
            print(f"Save error: {e}")
    
    def load_calibration(self):
        """Load calibration points from JSON file if it exists (auto-load at startup)."""
        calib_file = os.path.join(os.getcwd(), "spectrometer_calibration.json")
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    data = json.load(f)
                self.calibration_points = data.get('calibration_points', [])
                print(f"Loaded calibration: {len(self.calibration_points)} points")
                if len(self.calibration_points) >= 2:
                    print("  Points:", self.calibration_points)
            except Exception as e:
                print(f"Warning: Could not load calibration file: {e}")
                self.calibration_points = []
        else:
            self.calibration_points = []
    
    def load_calibration_dialog(self):
        """Load calibration from a user-selected file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration File", 
            os.getcwd(),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.calibration_points = data.get('calibration_points', [])
                
                # Update analyzer with loaded calibration
                if len(self.calibration_points) >= 2:
                    self.analyzer.calibration = {'points': self.calibration_points}
                    # Force recalibration of wavelengths
                    if self.analyzer.intensity_profile is not None:
                        self.analyzer.calibrate_wavelength()
                    QMessageBox.information(self, "Calibration Loaded",
                                          f"Loaded {len(self.calibration_points)} calibration points from:\n{file_path}")
                    print(f"Loaded calibration: {len(self.calibration_points)} points")
                    print("  Points:", self.calibration_points)
                    self.update_calibration_display()
                else:
                    QMessageBox.warning(self, "Invalid Calibration",
                                       f"File contains only {len(self.calibration_points)} points. Need at least 2.")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Calibration",
                                    f"Failed to load calibration file:\n{e}")
                print(f"Error loading calibration: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # More robust HSV filtering for full spectrum including orange
        # Reuse HSV buffer to avoid allocation
        if self.hsv_buffer is None or self.hsv_buffer.shape != frame.shape:
            self.hsv_buffer = np.empty_like(frame)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv_buffer)
        
        # Create multiple masks for different hue ranges to catch all spectrum colors
        # Use runtime-adjustable saturation and value thresholds
        # Red/Orange range (wraps around 180)
        mask1 = cv2.inRange(hsv, np.array([0, self.lower_s, self.lower_v]), np.array([25, self.upper_s, self.upper_v]))
        # Orange to Green
        mask2 = cv2.inRange(hsv, np.array([25, self.lower_s, self.lower_v]), np.array([85, self.upper_s, self.upper_v]))
        # Cyan to Blue to Violet
        mask3 = cv2.inRange(hsv, np.array([85, self.lower_s, self.lower_v]), np.array([160, self.upper_s, self.upper_v]))
        # Red wrap-around (upper red)
        mask4 = cv2.inRange(hsv, np.array([160, self.lower_s, self.lower_v]), np.array([180, self.upper_s, self.upper_v]))
        
        # Combine masks efficiently using bitwise OR
        mask = mask1 | mask2 | mask3 | mask4
        
        # Clean up noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply mask to keep only colored spectrum regions (set rest to black)
        spectrum_only = cv2.bitwise_and(frame, frame, mask=mask)

        # Flip horizontally if enabled
        if self.flip_enabled:
            processed = cv2.flip(spectrum_only, 1)
        else:
            processed = spectrum_only
        self.current_frame = processed.copy()

        # Display processed frame in the QLabel
        self._display_frame(processed)

        # Prepare ROI if crop box enabled
        roi = None
        if self.video_label.show_rect and self.video_label.rect is not None:
            roi = self.video_label.get_roi_on_frame(frame.shape)

        # Update analyzer and plot if not paused
        if not self.paused:
            res = self.analyzer.process_and_update(processed, roi=roi, auto_detect=(roi is None),
                                                   intensity_method='average', channel='gray',
                                                   update_plot=False)

            # Throttle plot updates to improve performance
            self._frame_counter += 1
            if (self._frame_counter % self.plot_update_interval) == 0:
                self._update_plot(res['wavelengths'], res['intensities'])

    def _display_frame(self, frame):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def _update_plot(self, wavelengths, intensities):
        if wavelengths is None or intensities is None:
            return
        x = wavelengths
        # Ensure intensities are non-negative
        y = np.clip(intensities, 0.0, None)

        self.line.set_data(x, y)

        # Keep x-axis fixed at 400-700nm range
        self.ax.set_xlim(400, 700)

        # Compute ymax
        try:
            ymax = float(np.nanmax(y))
        except Exception:
            ymax = 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        # add margin
        ymax *= 1.05
        
        # Only recreate gradient if y-limits changed significantly
        current_ylim = (0.0, ymax)
        if self._cached_ylim is None or abs(self._cached_ylim[1] - ymax) > ymax * 0.1:
            self._create_gradient(0.0, ymax)
            self._cached_ylim = current_ylim
        
        self.ax.set_ylim(0.0, ymax)

        # Update polygon clip path
        verts = np.column_stack([x, y])
        verts = np.vstack([[x[0], 0.0], verts, [x[-1], 0.0]])
        poly = Polygon(verts, closed=True, transform=self.ax.transData)
        if self.gradient_image is not None:
            self.gradient_image.set_clip_path(poly)

        self.canvas.draw_idle()
    
    def _create_gradient(self, ymin, ymax):
        """Create gradient only when needed."""
        xmin, xmax = 400, 700
        
        # Use cached gradient if available
        if self._gradient_cache is None:
            N = 300
            wavelength_samples = np.linspace(400, 700, N)
            colors = np.zeros((1, N, 3), dtype=float)
            
            # Map each wavelength to its actual visible color
            for i, wl in enumerate(wavelength_samples):
                if wl < 380:
                    colors[0, i, :] = [0, 0, 0]
                elif wl < 440:
                    colors[0, i, :] = [(440 - wl) / (440 - 380) * 0.58, 0, 1.0]
                elif wl < 490:
                    colors[0, i, :] = [0, (wl - 440) / (490 - 440), 1.0]
                elif wl < 510:
                    colors[0, i, :] = [0, 1.0, (510 - wl) / (510 - 490)]
                elif wl < 580:
                    colors[0, i, :] = [(wl - 510) / (580 - 510), 1.0, 0]
                elif wl < 645:
                    colors[0, i, :] = [1.0, (645 - wl) / (645 - 580), 0]
                elif wl <= 700:
                    colors[0, i, :] = [1.0, 0, 0]
                else:
                    colors[0, i, :] = [0, 0, 0]
            
            self._gradient_cache = colors
        
        # Remove old gradient
        if self.gradient_image is not None:
            try:
                self.gradient_image.remove()
            except:
                pass
        
        # Create new gradient with updated extent
        self.gradient_image = self.ax.imshow(
            self._gradient_cache, 
            extent=[xmin, xmax, ymin, ymax], 
            aspect='auto', 
            zorder=1, 
            alpha=0.6
        )
        
        self.line.set_zorder(2)
        self.line.set_color('black')


def main():
    app = QApplication(sys.argv)
    url = "http://192.168.137.41:8080//video"
    # url = "test_emission_spectrum.jpg"  # for testing with static image
    w = MainWindow(url)
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
