import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Any


ARCHIVE_EXTENSIONS = ('.zip', '.tar.gz', '.tgz', '.tar', '.gz')


def is_supported_archive(path: Path) -> bool:
    return any(path.name.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def find_archives(store_path: Path) -> List[Path]:
    return sorted(
        path for path in store_path.iterdir()
        if path.is_file() and is_supported_archive(path)
    )


def unique_destination(target_dir: Path, filename: str, overwrite: bool = False) -> Path:
    destination = target_dir / filename
    if overwrite or not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while destination.exists():
        destination = target_dir / f'{stem}_{counter}{suffix}'
        counter += 1
    return destination


def _ensure_within_directory(base_dir: Path, target_path: Path) -> None:
    base_resolved = base_dir.resolve()
    target_resolved = target_path.resolve()
    if base_resolved != target_resolved and base_resolved not in target_resolved.parents:
        raise ValueError(f'Archive member escapes extraction directory: {target_path}')


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            member_path = target_dir / member.filename
            _ensure_within_directory(target_dir, member_path)
        zip_ref.extractall(target_dir)


def _safe_extract_tar(archive_path: Path, target_dir: Path, mode: str) -> None:
    with tarfile.open(archive_path, mode) as tar_ref:
        for member in tar_ref.getmembers():
            member_path = target_dir / member.name
            _ensure_within_directory(target_dir, member_path)
        tar_ref.extractall(target_dir)


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == '.zip':
        _safe_extract_zip(archive_path, target_dir)
    elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
        _safe_extract_tar(archive_path, target_dir, 'r:gz')
    elif archive_path.suffix == '.tar':
        _safe_extract_tar(archive_path, target_dir, 'r')
    elif archive_path.suffix == '.gz':
        output_file = target_dir / archive_path.stem
        with gzip.open(archive_path, 'rb') as gz_ref:
            with output_file.open('wb') as out_ref:
                shutil.copyfileobj(gz_ref, out_ref)
    else:
        raise ValueError(f'Unsupported archive format: {archive_path}')


def move_csvs_from_directory(
    source_dir: Path,
    target_dir: Path,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> List[Path]:
    moved_files = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in source_dir.rglob('*.csv'):
        destination = unique_destination(target_dir, csv_file.name, overwrite=overwrite)
        if overwrite and destination.exists():
            destination.unlink()
            if logger:
                logger.info(f'Replacing existing file: {destination.name}')
        shutil.move(str(csv_file), str(destination))
        moved_files.append(destination)
        if logger:
            logger.info(f'Moved CSV file: {csv_file.name} -> {destination.name}')

    return moved_files


def extract_csvs_from_archive(
    archive_path: Path,
    target_dir: Path,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> List[Path]:
    temp_dir = target_dir / f'_temp_{archive_path.stem}'
    try:
        extract_archive(archive_path, temp_dir)
        csv_files = list(temp_dir.rglob('*.csv'))
        if not csv_files:
            if logger:
                logger.warning(f'No CSV files found in archive: {archive_path.name}')
            return []
        return move_csvs_from_directory(
            temp_dir,
            target_dir,
            overwrite=overwrite,
            logger=logger,
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
