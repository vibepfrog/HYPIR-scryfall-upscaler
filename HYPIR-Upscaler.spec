# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPEC).parent

hiddenimports = []
binaries = []
datas = []

# Packages with dynamic imports/data that need explicit collection.
for pkg in [
    'torch', 'torchvision', 'diffusers', 'transformers', 'peft',
    'accelerate', 'huggingface_hub', 'safetensors', 'PySide6'
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

for pkg in ['HYPIR', 'einops', 'PIL', 'tqdm']:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# Keep the pinned upstream HYPIR Python source available at runtime.
datas.append((str(root / 'vendor' / 'HYPIR'), 'vendor/HYPIR'))

a = Analysis(
    [str(root / 'app' / 'main.py')],
    pathex=[str(root), str(root / 'vendor' / 'HYPIR')],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'gradio', 'tensorboard'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HYPIR Upscaler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='HYPIR Upscaler',
)
