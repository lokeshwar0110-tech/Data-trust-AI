import os
import shutil

src_root = r"C:\Users\Lenovo\.gemini\antigravity\scratch\datatrust-ai"
dst_root = r"C:\DataTrust-AI-v3"

folders_to_sync = ["datatrust", "tests", "benchmarks", "docs", "backend"]

print(f"Syncing from {src_root} to {dst_root}...")

for folder in folders_to_sync:
    src_dir = os.path.join(src_root, folder)
    dst_dir = os.path.join(dst_root, folder)
    if os.path.exists(src_dir):
        print(f"Syncing folder: {folder}")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

print("Sync completed successfully.")
