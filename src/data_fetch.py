from pathlib import Path
from urllib.parse import urlparse
import tarfile, zipfile
import requests

CHUNK = 1024 * 1024

def download(url: str, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    resume = out_file.exists()
    start = out_file.stat().st_size if resume else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        # If server doesn't support resume, restart
        if start > 0 and r.status_code == 200:
            start = 0
        r.raise_for_status()
        mode = "ab" if start > 0 else "wb"

        total = r.headers.get("Content-Length")
        total = int(total) + start if total else None
        done = start

        with open(out_file, mode) as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\rDownloading {out_file.name}: {done*100/total:6.2f}%", end="")
                else:
                    print(f"\rDownloading {out_file.name}: {done/1024/1024:,.1f} MB", end="")
    print("\nDownload complete.")

def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive, "r:*") as t:
            t.extractall(dest)

def ensure_raw_midi(dataset_url: str, project_root: Path) -> Path:
    raw_dir = Path(project_root) / "data" / "raw_midi"
    if any(raw_dir.rglob("*.mid")) or any(raw_dir.rglob("*.midi")):
        return raw_dir

    if not dataset_url or "PASTE_" in dataset_url:
        raise RuntimeError("DATASET_URL not set. Provide a direct download URL (zip/tar.gz).")

    downloads = Path(project_root) / "data" / "_downloads"
    filename = Path(urlparse(dataset_url).path).name or "dataset.zip"
    archive_path = downloads / filename

    print("[DATA] raw_midi missing → downloading dataset…")
    download(dataset_url, archive_path)
    print("[DATA] extracting…")
    extract(archive_path, raw_dir)

    if not (any(raw_dir.rglob("*.mid")) or any(raw_dir.rglob("*.midi"))):
        raise RuntimeError(f"No MIDI found after extract in {raw_dir}. Check archive structure.")
    return raw_dir
