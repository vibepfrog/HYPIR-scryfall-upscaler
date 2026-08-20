from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
exe = root / 'dist' / 'HYPIR Upscaler' / 'HYPIR Upscaler.exe'
installer = root / 'installer-output' / 'HYPIR-Upscaler-Setup-0.4.1.exe'

missing = [str(p) for p in (exe, installer) if not p.exists()]
if missing:
    print('Missing build outputs:')
    for p in missing:
        print(' -', p)
    sys.exit(1)

print('Build outputs verified:')
print(' -', exe)
print(' -', installer)
