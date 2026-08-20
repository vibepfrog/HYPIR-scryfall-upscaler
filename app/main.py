from __future__ import annotations

import os
import sys
import random
import traceback
from pathlib import Path

APP_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "HYPIR-Upscaler"
MODEL_DIR = APP_HOME / "models"
# PyInstaller puts the bundled HYPIR source under the application directory.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
HYPIR_SOURCE = BUNDLE_DIR / "vendor" / "HYPIR"

sys.path.insert(0, str(HYPIR_SOURCE))

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate.utils import set_seed
from huggingface_hub import hf_hub_download, snapshot_download
from PySide6.QtCore import Qt, QObject, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QProgressBar,
    QMessageBox, QFrame, QSpinBox, QPlainTextEdit, QCheckBox, QGroupBox,
)

from HYPIR.enhancer.sd2 import SD2Enhancer


LORA_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
    "conv_shortcut", "conv_out", "proj_in", "proj_out",
    "ff.net.2", "ff.net.0.proj",
]
BASE_MODEL = MODEL_DIR / "stable-diffusion-2-1-base"
WEIGHTS = MODEL_DIR / "HYPIR_sd2.pth"


def pil_to_pixmap(image: Image.Image, max_size: QSize) -> QPixmap:
    image = image.convert("RGB")
    image.thumbnail((max_size.width(), max_size.height()), Image.Resampling.LANCZOS)
    data = image.tobytes("raw", "RGB")
    qimg = QImage(data, image.width, image.height, image.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class DropLabel(QLabel):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__("Drop an image here\n\nor click Browse")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.file_dropped.emit("__browse__")
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()


class UpscaleWorker(QObject):
    finished = Signal(object, str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, image_path: str, prompt: str, upscale: int,
                 patch_size: int, stride: int, seed: int, auto_retry: bool):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.upscale = upscale
        self.patch_size = patch_size
        self.stride = stride
        self.seed = seed
        self.auto_retry = auto_retry

    def run(self):
        global MODEL
        try:
            self.status.emit("Loading HYPIR model into GPU..." if MODEL is None else "Processing image...")

            if MODEL is None:
                MODEL = SD2Enhancer(
                    base_model_path=str(BASE_MODEL),
                    weight_path=str(WEIGHTS),
                    lora_modules=LORA_MODULES,
                    lora_rank=256,
                    model_t=200,
                    coeff_t=200,
                    device="cuda",
                )
                MODEL.init_models()

            seed = self.seed if self.seed >= 0 else random.randint(0, 2**32 - 1)
            set_seed(seed)

            image = Image.open(self.image_path).convert("RGB")
            tensor = ToTensor()(image).unsqueeze(0)

            attempts = [(self.patch_size, self.stride)]
            if self.auto_retry:
                for candidate in ((512, 256), (384, 192)):
                    if candidate not in attempts and candidate[0] < attempts[-1][0]:
                        attempts.append(candidate)

            last_error = None
            for i, (patch, stride) in enumerate(attempts):
                try:
                    self.status.emit(f"Running HYPIR • tiles {patch}/{stride}…")
                    result = MODEL.enhance(
                        lq=tensor,
                        prompt=self.prompt,
                        upscale=self.upscale,
                        patch_size=patch,
                        stride=stride,
                        return_type="pil",
                    )[0]
                    self.finished.emit(result, f"Done • {result.width} × {result.height} • seed {seed} • tiles {patch}/{stride}")
                    return
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    last_error = exc
                    if not self.auto_retry or i == len(attempts) - 1 or "out of memory" not in str(exc).lower():
                        raise
                    self.status.emit(f"GPU memory was tight — retrying with smaller tiles ({attempts[i+1][0]}/{attempts[i+1][1]})…")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            raise last_error
        except Exception:
            self.failed.emit(traceback.format_exc())


MODEL = None


def models_ready() -> bool:
    return (MODEL_DIR / "HYPIR_sd2.pth").exists() and (
        MODEL_DIR / "stable-diffusion-2-1-base" / "model_index.json"
    ).exists()


def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weight = MODEL_DIR / "HYPIR_sd2.pth"
    base = MODEL_DIR / "stable-diffusion-2-1-base"

    if not weight.exists():
        hf_hub_download(
            repo_id="lxq007/HYPIR",
            filename="HYPIR_sd2.pth",
            local_dir=str(MODEL_DIR),
        )

    if not (base / "model_index.json").exists():
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-2-1-base",
            local_dir=str(base),
        )


class SetupWindow(QWidget):
    ready = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYPIR Upscaler — First-time setup")
        self.resize(620, 330)
        self.setObjectName("setup")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 34, 30)
        layout.setSpacing(14)

        title = QLabel("Preparing HYPIR")
        title.setObjectName("title")
        layout.addWidget(title)

        text = QLabel(
            "The application is downloading the AI models it needs. "
            "This happens only once and can take a while because the models are large."
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        self.status = QLabel("Checking required files…")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)

        self.details = QLabel(
            "Your images are processed locally. Model downloads come from Hugging Face."
        )
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        layout.addStretch()

        self.thread = QThread()
        self.worker = SetupWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status.setText)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.complete)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def complete(self):
        self.status.setText("Setup complete. Starting HYPIR…")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.ready.emit()

    def failed(self, details):
        self.progress.hide()
        self.status.setText("Setup failed.")
        self.details.setText(
            "Please check your internet connection and try launching the app again.\n\n"
            + details[-4000:]
        )


class SetupWorker(QObject):
    finished = Signal()
    failed = Signal(str)
    status = Signal(str)

    def run(self):
        try:
            if models_ready():
                self.finished.emit()
                return
            self.status.emit("Downloading HYPIR weights (~1 GB)…")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            weight = MODEL_DIR / "HYPIR_sd2.pth"
            if not weight.exists():
                hf_hub_download(
                    repo_id="lxq007/HYPIR",
                    filename="HYPIR_sd2.pth",
                    local_dir=str(MODEL_DIR),
                )
            self.status.emit("Downloading Stable Diffusion 2.1 base model (several GB)…")
            base = MODEL_DIR / "stable-diffusion-2-1-base"
            if not (base / "model_index.json").exists():
                snapshot_download(
                    repo_id="stabilityai/stable-diffusion-2-1-base",
                    local_dir=str(base),
                )
            self.finished.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYPIR Upscaler")
        self.resize(1240, 820)
        self.setMinimumSize(1050, 700)
        self.input_path = None
        self.result = None
        self.thread = None
        self.worker = None
        self._build_ui()
        self._apply_style()

        if not torch.cuda.is_available():
            self.status_label.setText(
                "No CUDA-compatible NVIDIA GPU was detected. HYPIR's official implementation requires CUDA."
            )
            self.run_btn.setEnabled(False)
        else:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.status_label.setText(f"Ready • {name} • {vram:.1f} GB VRAM")

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("HYPIR Upscaler")
        title.setObjectName("title")
        subtitle = QLabel("Local AI image restoration & upscaling — simple by design, powered by HYPIR.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        top = QHBoxLayout()
        self.drop = DropLabel()
        self.drop.setMinimumHeight(220)
        self.drop.file_dropped.connect(self.load_input)
        top.addWidget(self.drop, 1)

        controls = QFrame()
        controls.setObjectName("controls")
        c = QVBoxLayout(controls)
        c.setContentsMargins(20, 18, 20, 18)

        self.browse = QPushButton("Browse for image…")
        self.browse.clicked.connect(lambda: self.load_input("__browse__"))
        c.addWidget(self.browse)

        c.addWidget(QLabel("Upscale"))
        self.scale = QComboBox()
        self.scale.addItems(["2×", "4×"])
        self.scale.setCurrentIndex(1)
        c.addWidget(self.scale)

        c.addWidget(QLabel("Restoration prompt (optional)"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "e.g. detailed realistic photograph, natural texture, faithful colours"
        )
        self.prompt.setMaximumHeight(90)
        c.addWidget(self.prompt)

        c.addWidget(QLabel("Quality / memory"))
        self.quality = QComboBox()
        self.quality.addItems([
            "Automatic — recommended",
            "Lower VRAM — 512 / 384",
            "Balanced — 512 / 256",
            "Higher quality — 768 / 512",
        ])
        c.addWidget(self.quality)

        self.retry = QCheckBox("Automatically retry with smaller tiles if VRAM runs out")
        self.retry.setChecked(True)
        c.addWidget(self.retry)

        row = QHBoxLayout()
        row.addWidget(QLabel("Seed"))
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2147483647)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("Random")
        row.addWidget(self.seed)
        c.addLayout(row)

        self.run_btn = QPushButton("Upscale image")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self.run_upscale)
        c.addWidget(self.run_btn)

        self.save_btn = QPushButton("Save result…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        c.addWidget(self.save_btn)
        c.addStretch()
        top.addWidget(controls, 0)
        layout.addLayout(top)

        previews = QHBoxLayout()
        self.before = QLabel("Original")
        self.after = QLabel("Upscaled result")
        for label in (self.before, self.after):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(480, 330)
            label.setObjectName("preview")
        previews.addWidget(self.before)
        previews.addWidget(self.after)
        layout.addLayout(previews, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background: #111318; color: #e9ecf1; font-size: 14px; }
            QLabel#title { font-size: 30px; font-weight: 700; }
            QLabel#subtitle { color: #9aa2b1; font-size: 15px; }
            QFrame#controls { background: #1a1e26; border: 1px solid #2c3340; border-radius: 12px; }
            QLabel#dropArea { background: #181c24; border: 2px dashed #454e5d; border-radius: 14px; color: #aeb7c5; font-size: 17px; }
            QLabel#preview { background: #181c24; border: 1px solid #2c3340; border-radius: 10px; color: #727b89; }
            QPushButton { background: #292f3a; border: 0; border-radius: 8px; padding: 11px 14px; }
            QPushButton:hover { background: #343b49; }
            QPushButton#runButton { background: #6d5dfc; font-weight: 700; padding: 13px; }
            QPushButton#runButton:hover { background: #806fff; }
            QComboBox, QSpinBox, QPlainTextEdit { background: #11151c; border: 1px solid #303846; border-radius: 7px; padding: 8px; }
            QProgressBar { border: 0; background: #242a34; height: 7px; border-radius: 3px; }
            QProgressBar::chunk { background: #6d5dfc; border-radius: 3px; }
            QLabel#status { color: #9aa2b1; }
        """)

    def load_input(self, path: str):
        if path == "__browse__":
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose image", "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
            )
            if not path:
                return

        try:
            image = Image.open(path).convert("RGB")
            self.input_path = path
            self.result = None
            self.save_btn.setEnabled(False)
            self.after.setText("Upscaled result")
            self.before.setPixmap(pil_to_pixmap(image, self.before.size() - QSize(20, 20)))
            self.status_label.setText(f"Loaded • {image.width} × {image.height}")
        except Exception as exc:
            QMessageBox.critical(self, "Could not open image", str(exc))

    def _select_tiles(self):
        idx = self.quality.currentIndex()
        if idx == 1:
            return 512, 384
        if idx == 2:
            return 512, 256
        if idx == 3:
            return 768, 512

        # Automatic: choose a conservative starting point from detected VRAM.
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 0
        if vram_gb < 8:
            return 384, 192
        if vram_gb < 12:
            return 512, 256
        return 768, 512

    def run_upscale(self):
        if not self.input_path:
            QMessageBox.information(self, "Choose an image", "Drop an image into the window first.")
            return
        if not torch.cuda.is_available():
            return

        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.show()

        patch, stride = self._select_tiles()
        prompt = self.prompt.toPlainText().strip()
        scale = 2 if self.scale.currentIndex() == 0 else 4

        self.thread = QThread()
        self.worker = UpscaleWorker(
            self.input_path, prompt, scale, patch, stride, self.seed.value(), self.retry.isChecked()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_finished(self, image, message):
        self.result = image
        self.after.setPixmap(pil_to_pixmap(image, self.after.size() - QSize(20, 20)))
        self.status_label.setText(message)
        self.progress.hide()
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def on_failed(self, details):
        self.progress.hide()
        self.run_btn.setEnabled(True)
        QMessageBox.critical(
            self, "HYPIR failed",
            "HYPIR could not process this image.\n\n"
            "Try the Lower VRAM preset or a smaller input image.\n\n"
            + details[-5000:]
        )
        self.status_label.setText("Processing failed — see error dialog.")

    def save_result(self):
        if self.result is None:
            return
        default = str(Path(self.input_path).with_name(
            Path(self.input_path).stem + "_HYPIR.png"
        ))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save upscaled image", default,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)"
        )
        if not path:
            return
        try:
            self.result.save(path)
            self.status_label.setText(f"Saved • {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save image", str(exc))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HYPIR Upscaler")
    app.setFont(QFont("Segoe UI", 10))

    if not torch.cuda.is_available():
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    if not models_ready():
        setup = SetupWindow()
        setup.show()

        holder = {"window": None}

        def start_main():
            setup.close()
            holder["window"] = MainWindow()
            holder["window"].show()

        setup.ready.connect(start_main)
        sys.exit(app.exec())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
