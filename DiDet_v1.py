import sys
import json
import warnings
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from skimage import io, morphology, exposure, filters
from scipy.interpolate import splprep, splev

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*remove_small_objects.*"
)


class FiberDiameterGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Fiber Diameter – Final SEM-Safe Tool")
        self.resize(1350, 850)

        self.image = None
        self.binary = None
        self.edges = None
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
            ("Preview Edges", self.preview_edges),
            ("Analyze & Export", self.analyze),
        ]

        for txt, func in buttons:
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(func)
            panel.addWidget(b)

        panel.addStretch()
        self.view.scene.sigMouseClicked.connect(self.on_mouse_click)

    # ==================================================
    # Image
    # ==================================================
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SEM Image", "", "Images (*.tif *.tiff *.png *.jpg)"
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
    # Drawing
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
    # Save / Load
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
    # Debug
    # ==================================================
    def preview_edges(self):
        if self.edges is not None:
            self.view.setImage((self.edges.astype(np.uint8) * 255).T)

    # ==================================================
    # Analysis
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

        # --- ROI mask ---
        h, w = self.image.shape
        mask = np.zeros((h, w), bool)
        poly = np.array(self.roi_points)
        from matplotlib.path import Path
        P = Path(poly)
        yy, xx = np.mgrid[:h, :w]
        coords = np.vstack((xx.ravel(), yy.ravel())).T
        mask = P.contains_points(coords).reshape(h, w)

        fiber = self.image * mask
        th = np.percentile(fiber[fiber > 0], 35)
        self.binary = fiber > th

        try:
            self.binary = morphology.remove_small_objects(
                self.binary, area_threshold=299
            )
        except TypeError:
            self.binary = morphology.remove_small_objects(
                self.binary, min_size=300
            )

        self.binary = morphology.closing(self.binary, morphology.disk(2))

        # --- Robust SEM edges ---
        self.edges = filters.sobel(self.binary) > 0
        self.edges = morphology.binary_dilation(self.edges, morphology.disk(2))

        # --- Clean centerline ---
        cl = np.array(self.centerline_pts)
        clean = [cl[0]]
        for p in cl[1:]:
            if np.linalg.norm(p - clean[-1]) > 1.0:
                clean.append(p)
        cl = np.array(clean)

        # --- Resample centerline ---
        if len(cl) >= 4:
            try:
                tck, _ = splprep([cl[:, 0], cl[:, 1]], s=0, k=3)
                u = np.linspace(0, 1, n_probes)
                cx, cy = splev(u, tck)
            except Exception:
                d = np.sqrt(((cl[1:] - cl[:-1])**2).sum(axis=1))
                s = np.insert(np.cumsum(d), 0, 0) / np.sum(d)
                u = np.linspace(0, 1, n_probes)
                cx = np.interp(u, s, cl[:, 0])
                cy = np.interp(u, s, cl[:, 1])
        else:
            d = np.sqrt(((cl[1:] - cl[:-1])**2).sum(axis=1))
            s = np.insert(np.cumsum(d), 0, 0) / np.sum(d)
            u = np.linspace(0, 1, n_probes)
            cx = np.interp(u, s, cl[:, 0])
            cy = np.interp(u, s, cl[:, 1])

        centerline = np.column_stack((cx, cy))

        # --- Probe intersections (edge-based) ---
        L = max(h, w)
        diam_um = []
        probe_rows = []

        for i in range(1, len(centerline) - 1):
            tvec = centerline[i + 1] - centerline[i - 1]
            tvec /= (np.linalg.norm(tvec) + 1e-9)
            nvec = np.array([-tvec[1], tvec[0]])

            c = centerline[i]
            q1 = c - L * nvec
            q2 = c + L * nvec

            xs = np.linspace(q1[0], q2[0], int(2 * L))
            ys = np.linspace(q1[1], q2[1], int(2 * L))

            hits = []
            for xk, yk in zip(xs, ys):
                xi, yi = int(xk), int(yk)
                if 0 <= yi < h and 0 <= xi < w:
                    if self.edges[yi, xi]:
                        hits.append([xk, yk])

            if len(hits) >= 2:
                pA, pB = hits[0], hits[-1]
                d_px = np.linalg.norm(np.array(pA) - np.array(pB))
                d_um = d_px * self.px_to_um

                diam_um.append(d_um)
                probe_rows.append({
                    "probe": i,
                    "center_x": c[0],
                    "center_y": c[1],
                    "left_x": pA[0],
                    "left_y": pA[1],
                    "right_x": pB[0],
                    "right_y": pB[1],
                    "diameter_um": d_um
                })

                line = pg.PlotDataItem(
                    [pA[0], pB[0]],
                    [pA[1], pB[1]],
                    pen=pg.mkPen((65, 105, 225), width=1.5)
                )
                self.view.addItem(line)
                self.probe_items.append(line)

        if not diam_um:
            QtWidgets.QMessageBox.critical(
                self, "Failed",
                "No valid probe intersections.\nCheck segmentation or ROI."
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
            "Diameters and probe coordinates exported successfully."
        )


# ==================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FiberDiameterGUI()
    win.show()
    sys.exit(app.exec_())
