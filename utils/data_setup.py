from pathlib import Path


def ensure_musdbhq(root: str | Path, gdrive_id: str = "1ieGcVPPfgWg__BTDlIGi1TpntdOWwwdn") -> Path:
    """
    Ensure MUSDB18-HQ dataset exists at root. If missing, download a .7z archive via gdown and extract with py7zr.
    The archive is expected to contain track folders with mixture.wav and stems (drums/bass/vocals/other).
    """
    try:
        from gdown import download as gdown_download
    except Exception:
        raise RuntimeError("gdown is required. Please install with `pip install gdown`." )

    try:
        import py7zr  # noqa: F401
    except Exception:
        raise RuntimeError("py7zr is required to extract .7z archives. Please install with `pip install py7zr`.")

    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # Heuristic: dataset present if there is at least one mixture.wav under root
    has_any = any(root.glob("**/*/mixture.wav"))
    if has_any:
        return root

    archive_path = root / "MUSDB18-HQ.7z"
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    # Download
    if not archive_path.exists():
        print(f"Downloading MUSDB18-HQ (.7z) to {archive_path}...")
        gdown_download(url, str(archive_path), quiet=False)

    # Extract using py7zr
    print(f"Extracting {archive_path} ...")
    import py7zr
    with py7zr.SevenZipFile(str(archive_path), mode='r') as z:
        z.extractall(path=str(root))
    print("Extraction completed.")

    return root
