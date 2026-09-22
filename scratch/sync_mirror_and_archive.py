import os
import sys
import shutil
import zipfile

SOURCE = r"C:\Users\Lenovo\.gemini\antigravity\scratch\datatrust-ai"
MIRROR = r"C:\DataTrust-AI-v3"
FROZEN_DIR = r"C:\DataTrust-AI-v3-Frozen"
FROZEN_ZIP = os.path.join(FROZEN_DIR, "DataTrust-AI-v3-Research-Hardened.zip")
USER_ZIP = r"C:\Users\Lenovo\DataTrust-AI-v3-Research-Hardened.zip"

print("=" * 80)
print("DATA TRUST AI v3 — PERMANENT MIRROR AND FROZEN ARCHIVE SYNCHRONIZATION")
print("=" * 80)

# 1. Sync to C:\DataTrust-AI-v3
print(f"Step 1: Synchronizing mirror at {MIRROR}...")
os.makedirs(MIRROR, exist_ok=True)

# Walk source and copy files excluding ephemeral
ephemeral_dirs = {"__pycache__", ".pytest_cache", ".git"}
ephemeral_exts = {".pyc", ".pyo"}

copied_files = 0
for root, dirs, files in os.walk(SOURCE):
    # Filter directories in-place
    dirs[:] = [d for d in dirs if d not in ephemeral_dirs]
    
    rel_path = os.path.relpath(root, SOURCE)
    dest_dir = os.path.join(MIRROR, rel_path) if rel_path != "." else MIRROR
    os.makedirs(dest_dir, exist_ok=True)
    
    for f in files:
        _, ext = os.path.splitext(f)
        if ext in ephemeral_exts:
            continue
        src_file = os.path.join(root, f)
        dst_file = os.path.join(dest_dir, f)
        shutil.copy2(src_file, dst_file)
        copied_files += 1

print(f"  Successfully mirrored {copied_files} files to {MIRROR}")

# 2. Create Frozen ZIP archive
print(f"\nStep 2: Generating frozen ZIP archive at {FROZEN_ZIP}...")
os.makedirs(FROZEN_DIR, exist_ok=True)

zip_file_count = 0
with zipfile.ZipFile(FROZEN_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(MIRROR):
        dirs[:] = [d for d in dirs if d not in ephemeral_dirs]
        for f in files:
            _, ext = os.path.splitext(f)
            if ext in ephemeral_exts:
                continue
            abs_file = os.path.join(root, f)
            arc_name = os.path.relpath(abs_file, MIRROR)
            zf.write(abs_file, arcname=arc_name)
            zip_file_count += 1

zip_size = os.path.getsize(FROZEN_ZIP)
print(f"  Successfully packaged {zip_file_count} files into {FROZEN_ZIP} ({zip_size:,} bytes)")

# 3. Copy ZIP to User root
print(f"\nStep 3: Copying archive to {USER_ZIP}...")
shutil.copy2(FROZEN_ZIP, USER_ZIP)
user_zip_size = os.path.getsize(USER_ZIP)
print(f"  Successfully copied to {USER_ZIP} ({user_zip_size:,} bytes)")

print("\n" + "=" * 80)
print("PERMANENT MIRROR & FROZEN ARCHIVES SYNCHRONIZED SUCCESSFULLY")
print("=" * 80)
