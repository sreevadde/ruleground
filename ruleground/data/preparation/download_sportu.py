"""
SPORTU Video Downloader

Downloads videos from the SPORTU benchmark (Google Drive) for the
3 target sports: basketball, soccer, American football.

Videos are organized as zip files per sport on Google Drive:
    https://drive.google.com/drive/folders/1nvA8gqF32lrhqzhbJ2r39-TwwW5tEvsu

Usage:
    python -m ruleground.data.preparation.download_sportu \\
        --output-dir data/sportu_raw/videos

Prerequisites:
    pip install gdown
"""

from __future__ import annotations

import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Google Drive folder: https://drive.google.com/drive/folders/1nvA8gqF32lrhqzhbJ2r39-TwwW5tEvsu
# Individual sport zip file IDs (extracted from the Drive folder).
# These are placeholders — update with actual file IDs after listing the Drive folder.
DRIVE_FOLDER_ID = "1nvA8gqF32lrhqzhbJ2r39-TwwW5tEvsu"


def download_drive_folder(
    output_dir: str | Path,
    sports: Optional[list] = None,
) -> Dict[str, Path]:
    """Download SPORTU videos from Google Drive.

    Uses gdown to download from the shared Drive folder.
    Downloads entire folder, then extracts relevant sport zips.

    Args:
        output_dir:  Directory to save videos.
        sports:      List of sports to download (default: all 3).

    Returns:
        Dict mapping sport → directory of extracted videos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sports = sports or ["basketball", "soccer", "american_football"]

    # Ensure gdown is available
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    # Download the entire shared folder
    folder_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
    download_dir = output_dir / "_downloads"
    download_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading SPORTU videos from Google Drive folder...")
    logger.info(f"Folder URL: {folder_url}")
    logger.info(f"Download directory: {download_dir}")

    try:
        gdown.download_folder(
            url=folder_url,
            output=str(download_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        logger.error(f"gdown folder download failed: {e}")
        logger.info(
            "Alternative: manually download zip files from:\n"
            f"  {folder_url}\n"
            f"Place them in: {download_dir}"
        )
        raise

    # Find and extract zip files
    sport_dirs = {}
    zip_files = list(download_dir.rglob("*.zip"))
    logger.info(f"Found {len(zip_files)} zip files")

    # Map zip filename patterns to our sport names
    sport_patterns = {
        "basketball": ["basketball", "Basketball"],
        "soccer": ["soccer", "Soccer"],
        "american_football": ["american_football", "American_Football", "football"],
    }

    for zf in zip_files:
        fname = zf.stem.lower()
        for sport, patterns in sport_patterns.items():
            if sport not in sports:
                continue
            if any(p.lower() in fname for p in patterns):
                extract_dir = output_dir / sport
                extract_dir.mkdir(exist_ok=True)
                logger.info(f"Extracting {zf.name} → {extract_dir}")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(extract_dir)
                sport_dirs[sport] = extract_dir
                break

    logger.info(f"Downloaded and extracted: {list(sport_dirs.keys())}")
    return sport_dirs


def list_available_videos(video_dir: str | Path) -> Dict[str, int]:
    """Count available video files per sport.

    Args:
        video_dir: Root directory containing sport subdirectories.

    Returns:
        Dict mapping sport → video count.
    """
    video_dir = Path(video_dir)
    counts = {}
    for sport_dir in video_dir.iterdir():
        if sport_dir.is_dir() and sport_dir.name != "_downloads":
            videos = list(sport_dir.glob("*.mp4")) + list(sport_dir.glob("**/*.mp4"))
            counts[sport_dir.name] = len(videos)
    return counts


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SPORTU videos from Google Drive"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sportu_raw/videos",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--sports", nargs="+",
        default=["basketball", "soccer", "american_football"],
        help="Sports to download",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list available videos (no download)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    if args.list_only:
        counts = list_available_videos(args.output_dir)
        print("\nAvailable videos:")
        for sport, cnt in sorted(counts.items()):
            print(f"  {sport}: {cnt} videos")
        return

    sport_dirs = download_drive_folder(args.output_dir, args.sports)
    counts = list_available_videos(args.output_dir)
    print("\nDownloaded videos:")
    for sport, cnt in sorted(counts.items()):
        print(f"  {sport}: {cnt} videos")


if __name__ == "__main__":
    main()
