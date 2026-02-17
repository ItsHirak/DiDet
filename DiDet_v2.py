# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 01:55:20 2026

@author: Hirak
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets
import pyqtgraph as pg

from skimage import io, morphology, exposure, filters
from scipy.interpolate import splprep, splev

warnings.filterwarnings("ignore")


class FiberDiameterGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Fiber Diameter – ROI-Based Measurement Tool")
        self.resize(1350, 850)

        self.image = None
        self.binary = None
        self.px_to_um = None

        self.roi_points = []
        self.centerline_pts = []

        self.roi_plot = None
        self.centerline_plot = None
        self.probe_items = []

        self.scale_points = None
        self.drawing_roi = False
        self.drawing_cl = False

        self._build_ui()

    # ==================================================
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.view = pg.ImageView()
        self.view.ui.histogram.hide()
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        self.view.getView().setMenuEnabled(False)
        layout.addWidget(self.view, 4)

        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel, 1)

        buttons = [
            ("Load SEM Image", self.load_image),
            ("Save ROI + Centerline", self.save_geometry),
            ("Load ROI + Centerline", self.load_geometry),
            ("Draw ROI", self.start_roi),
            ("Draw Centerline", self.start_centerline),
            ("Set Scale Bar", self.start_scale),
            ("Analyze & Export", self.analyze),
        ]

        for txt, func in buttons:
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(func)
            panel.addWidget(b)

        panel.addStretch()
        self.view.scene.sigMouseClicked.connect(self.on_mouse_click)

    # ==================================================
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SEM Image", "", "Images (*.tif *.png *.jpg)"
        )
        if not path:
            return

        img = io.imread(path)
        if img.ndim == 3:
            img = img.mean(axis=2)

        self.image = exposure.rescale_intensity(img)
        self.view.setImage(self.image.T)
        self.view.getView().invertY(True)

    # ==================================================
    def start_roi(self):
        self.roi_points = []
        self._remove(self.roi_plot)
        self.drawing_roi = True

    def start_centerline(self):
        self.centerline_pts = []
        self._remove(self.centerline_plot)
        self.drawing_cl = True

    def start_scale(self):
        self.scale_points = []

    # ==================================================
    def on_mouse_click(self, event):
        pos = self.view.getView().mapSceneToView(event.scenePos())
        x, y = pos.x(), pos.y()

        if self.drawing_roi:
            if event.double():
                self.drawing_roi = False
                return
            self.roi_points.append([x, y])
            self._plot(self.roi_points, 'r', 'roi_plot')
            return

        if self.drawing_cl:
            if event.double():
                self.drawing_cl = False
                return
            self.centerline_pts.append([x, y])
            self._plot(self.centerline_pts, 'y', 'centerline_plot')
            return

        if self.scale_points is not None and len(self.scale_points) < 2:
            self.scale_points.append([x, y])
            if len(self.scale_points) == 2:
                self.finish_scale()

    # ==================================================
    def _plot(self, pts, color, attr):
        self._remove(getattr(self, attr, None))
        arr = np.array(pts)
        item = pg.PlotDataItem(
            arr[:, 0], arr[:, 1],
            pen=pg.mkPen(color, width=2),
            symbol='o'
        )
        setattr(self, attr, item)
        self.view.addItem(item)

    def _remove(self, item):
        if item:
            self.view.removeItem(item)

    # ==================================================
    def finish_scale(self):
        p1, p2 = np.array(self.scale_points)
        px = np.linalg.norm(p2 - p1)
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Scale", "Scale length (µm):", decimals=4
        )
        if ok and px > 0:
            self.px_to_um = val / px
        self.scale_points = None

    # ==================================================
    def save_geometry(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Geometry", "", "Geometry (*.json)"
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump({
                "roi": self.roi_points,
                "centerline": self.centerline_pts,
                "px_to_um": self.px_to_um
            }, f, indent=2)

    def load_geometry(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Geometry", "", "Geometry (*.json)"
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)

        self.roi_points = data["roi"]
        self.centerline_pts = data["centerline"]
        self.px_to_um = data.get("px_to_um")

        self._plot(self.roi_points, 'r', 'roi_plot')
        self._plot(self.centerline_pts, 'y', 'centerline_plot')

    # ==================================================
    # Geometry
    # ==================================================
    def _line_segment_intersection(self, p, r, q, s):
        rxs = np.cross(r, s)
        if abs(rxs) < 1e-9:
            return None
        t = np.cross(q - p, s) / rxs
        u = np.cross(q - p, r) / rxs
        if 0 <= u <= 1:
            return p + t * r
        return None

    # ==================================================
    def analyze(self):
        if self.image is None or self.px_to_um is None:
            QtWidgets.QMessageBox.warning(self, "Missing", "Image or scale missing.")
            return

        if len(self.roi_points) < 3 or len(self.centerline_pts) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Missing", "ROI or centerline incomplete."
            )
            return

        n_probes, ok = QtWidgets.QInputDialog.getInt(
            self, "Probes", "Number of diameter probes:", 50, 5, 500
        )
        if not ok:
            return

        for p in self.probe_items:
            self.view.removeItem(p)
        self.probe_items = []

        # --- clean centerline ---
        cl = np.array(self.centerline_pts)
        d = np.sqrt(((cl[1:] - cl[:-1])**2).sum(axis=1))
        s = np.insert(np.cumsum(d), 0, 0) / np.sum(d)
        u = np.linspace(0, 1, n_probes)
        cx = np.interp(u, s, cl[:, 0])
        cy = np.interp(u, s, cl[:, 1])
        centerline = np.column_stack((cx, cy))

        roi = np.array(self.roi_points)
        roi_segments = list(zip(roi, np.roll(roi, -1, axis=0)))

        diam_um = []
        probe_rows = []

        for i in range(1, len(centerline) - 1):
            tvec = centerline[i + 1] - centerline[i - 1]
            tvec /= (np.linalg.norm(tvec) + 1e-9)
            nvec = np.array([-tvec[1], tvec[0]])

            p = centerline[i]
            hits = []

            for a, b in roi_segments:
                q = np.array(a)
                svec = np.array(b) - q
                inter = self._line_segment_intersection(p, nvec, q, svec)
                if inter is not None:
                    hits.append(inter)

            if len(hits) < 2:
                continue

            hits = np.array(hits)
            proj = np.dot(hits - p, nvec)
            left = hits[np.argmin(proj)]
            right = hits[np.argmax(proj)]

            d_px = np.linalg.norm(left - right)
            d_um = d_px * self.px_to_um

            diam_um.append(d_um)
            probe_rows.append({
                "probe": i,
                "diameter_um": d_um,
                "left_x": left[0],
                "left_y": left[1],
                "right_x": right[0],
                "right_y": right[1]
            })

            line = pg.PlotDataItem(
                [left[0], right[0]],
                [left[1], right[1]],
                pen=pg.mkPen((65, 105, 225), width=1.5)
            )
            self.view.addItem(line)
            self.probe_items.append(line)

        if not diam_um:
            QtWidgets.QMessageBox.critical(
                self, "Failed",
                "No valid probe intersections.\nCheck ROI geometry."
            )
            return

        pd.DataFrame({"Diameter (µm)": diam_um}).to_excel(
            "fiber_diameter_distribution.xlsx", index=False
        )
        pd.DataFrame(probe_rows).to_excel(
            "probe_coordinates.xlsx", index=False
        )

        QtWidgets.QMessageBox.information(
            self, "Done",
            "Full-width ROI-based diameters exported successfully."
        )


# ==================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FiberDiameterGUI()
    win.show()
    sys.exit(app.exec_())

