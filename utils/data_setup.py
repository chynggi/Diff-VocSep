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
    # Try to show per-file progress using tqdm; gracefully degrade if unavailable.
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None  # type: ignore

    with py7zr.SevenZipFile(str(archive_path), mode='r') as z:
        # Collect file names in archive (exclude directories)
        try:
            names = z.getnames()
        except Exception:
            # Fallback to using list() entries if getnames is unavailable
            try:
                names = [getattr(info, "filename", getattr(info, "name", "")) for info in z.list()]
            except Exception:
                names = []

        file_names = [n for n in names if n and not n.endswith("/") and not n.endswith("\\")]  # exclude dirs

        if tqdm and file_names:
            bar = tqdm(total=len(file_names), desc="Extracting MUSDB18-HQ", unit="file")
        else:
            bar = None

        def _update_bar():
            if bar is not None:
                bar.update(1)

        try:
            # Extract per-file to enable progress; skip files that already exist
            for name in file_names or []:
                dest = root / name
                if dest.exists():
                    _update_bar()
                    continue
                # Some py7zr versions use (path, targets), others support (targets, path)
                try:
                    z.extract(path=str(root), targets=[name])  # type: ignore[arg-type]
                except TypeError:
                    z.extract(targets=[name], path=str(root))  # type: ignore[arg-type]
                _update_bar()
        finally:
            if bar is not None:
                bar.close()

        # If we couldn't enumerate names (rare), fall back to single-shot extract
        if not file_names:
            z.extractall(path=str(root))

    print("Extraction completed.")

    return root
