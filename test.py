import sys
import os
from pathlib import Path

print("=" * 40)
print("🔍 DIAGNOSA STRUKTUR FOLDER")
print("=" * 40)

root_dir = Path(__file__).parent
print(f"📁 Root Folder: {root_dir}")

src_path = root_dir / 'src'
print(f"📁 Target SRC : {src_path}")

if not src_path.exists():
    print("❌ ERROR: Folder 'src' tidak ditemukan!")
    sys.exit()
else:
    print("✅ Folder 'src' ditemukan.")

utils_path = src_path / 'utils'
if not utils_path.exists():
    print("❌ ERROR: Folder 'src/utils' tidak ditemukan!")
    if (src_path / 'utils.py').exists():
        print("⚠️ PERINGATAN: Ditemukan file 'src/utils.py'. Ini bisa bikin konflik! Hapus/Rename file ini.")
else:
    print("✅ Folder 'src/utils' ditemukan.")

    print("\n📄 Isi folder 'src/utils':")
    files = [f.name for f in utils_path.iterdir() if f.is_file()]
    for f in files:
        print(f"   - {f}")

    if 'visualizer.py' in files:
        print("\n✅ File 'visualizer.py' ADA.")
    else:
        print("\n❌ Gawat! File 'visualizer.py' HILANG.")

print("=" * 40)

print("🧪 Test Import...")
sys.path.insert(0, str(src_path))

try:
    from utils import visualizer

    print("✅ SUCCESS: Import utils.visualizer berhasil!")
except ImportError as e:
    print(f"❌ FAILED: Masih error -> {e}")