import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core import get_logger, settings
from core.archive_utils import extract_csvs_from_archive, find_archives

from ..common import emit_result

logger = get_logger(__name__)


def run_feature_migration(store_path: Path, raw_path: Path, config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    migrated_files = []
    processed_features = []

    for feature_name, mappings in config.get("features", {}).items():
        csv_file = store_path / f"{feature_name}.csv"
        if not csv_file.exists():
            logger.warning(f"CSV file not found for feature {feature_name}: {csv_file}")
            continue

        logger.info(f"Processing feature: {feature_name}")

        try:
            sep = "\t" if feature_name == "free_hosting" else ","
            df = pd.read_csv(csv_file, sep=sep, comment="#")
            logger.info(f"Loaded {len(df)} rows from {csv_file.name}")

            value_cols = mappings.get("value", [])
            if not value_cols:
                logger.warning(f"No value columns defined for feature {feature_name}")
                continue

            desc_cols = mappings.get("description", [])
            available_cols = df.columns.tolist()
            selected_value_cols = [col for col in value_cols if col in available_cols]
            selected_desc_cols = [col for col in desc_cols if col in available_cols]

            if not selected_value_cols:
                logger.warning(f"No valid value columns found for feature {feature_name}")
                continue

            new_df = pd.DataFrame()

            if len(selected_value_cols) == 1:
                new_df["value"] = df[selected_value_cols[0]]
            else:
                new_df["value"] = df[selected_value_cols].bfill(axis=1).iloc[:, 0]

            if selected_desc_cols:
                if len(selected_desc_cols) == 1:
                    new_df["description"] = df[selected_desc_cols[0]]
                else:
                    new_df["description"] = df[selected_desc_cols].bfill(axis=1).iloc[:, 0]
            else:
                new_df["description"] = ""

            new_df = new_df.dropna(subset=["value"])
            new_df["value"] = new_df["value"].astype(str).str.strip()
            new_df = new_df[new_df["value"] != ""]
            new_df = new_df.drop_duplicates(subset=["value"])
            new_df = new_df.reset_index(drop=True)
            new_df.insert(0, "id", range(len(new_df)))
            new_df["description"] = new_df["description"].fillna("").astype(str)

            output_file = raw_path / f"{feature_name}.csv"
            new_df.to_csv(output_file, index=False)
            migrated_files.append(output_file.name)
            processed_features.append(feature_name)

            logger.info(f"Migrated {len(new_df)} cleaned rows to {output_file.name}")

        except Exception as e:
            logger.error(f"Error processing feature {feature_name}: {str(e)}")
            continue

    return {
        "status": "success" if migrated_files else "no_files_migrated",
        "processed_features": processed_features,
        "migrated_files": migrated_files,
        "total_features": len(processed_features),
        "total_files": len(migrated_files),
    }


def cmd_data_migrate(args: Any) -> int:
    logger.info("Starting data migration from CLI...")

    store_path = Path(args.store_path) if args.store_path else Path(settings.dataset_path).parent / "store"
    raw_path = Path(args.raw_path) if args.raw_path else Path(settings.dataset_path)

    if not store_path.exists():
        logger.error(f"Store path does not exist: {store_path}")
        return 1

    os.makedirs(raw_path, exist_ok=True)
    logger.info(f"Extracting datasets from {store_path} to {raw_path}")

    extracted_files = []
    processed_archives = []

    archive_files = find_archives(store_path)

    if not archive_files:
        logger.warning(f"No archive files found in {store_path}")
        return 1

    logger.info(f"Found {len(archive_files)} archive file(s) to process")

    for archive_file in archive_files:
        logger.info(f"Processing archive: {archive_file.name}")

        try:
            moved_files = extract_csvs_from_archive(
                archive_file,
                raw_path,
                overwrite=False,
                logger=logger,
            )

            if not moved_files:
                continue

            logger.info(f"Found {len(moved_files)} CSV file(s) in archive")
            extracted_files.extend(path.name for path in moved_files)
            processed_archives.append(archive_file.name)

        except Exception as e:
            logger.error(f"Error processing archive {archive_file.name}: {str(e)}")
            continue

    migration_report = {
        "status": "success" if extracted_files else "no_files_extracted",
        "store_path": str(store_path),
        "raw_path": str(raw_path),
        "processed_archives": processed_archives,
        "extracted_files": extracted_files,
        "total_archives": len(processed_archives),
        "total_csv_files": len(extracted_files),
    }

    logger.info(
        f"Data migration complete: {len(extracted_files)} CSV file(s) extracted "
        f"from {len(processed_archives)} archive(s)"
    )

    emit_result(migration_report, args.output)

    return 0 if extracted_files else 1


def cmd_data_migrate_feature(args: Any) -> int:
    logger.info("Starting feature data migration from CLI...")

    store_path = Path(args.store_path) if args.store_path else Path(settings.dataset_path).parent / "feature" / "store"
    raw_path = Path(args.raw_path) if args.raw_path else Path(settings.dataset_path).parent / "feature" / "raw"
    config_path = (
        Path(args.config) if args.config else Path(settings.dataset_path).parent / "feature" / "dataset_feature.yaml"
    )

    if not store_path.exists():
        logger.error(f"Store path does not exist: {store_path}")
        return 1

    if not config_path.exists():
        logger.error(f"Config file does not exist: {config_path}")
        return 1

    os.makedirs(raw_path, exist_ok=True)
    logger.info(f"Migrating feature datasets from {store_path} to {raw_path} using config {config_path}")

    result = run_feature_migration(store_path, raw_path, config_path)

    migration_report = {
        **result,
        "store_path": str(store_path),
        "raw_path": str(raw_path),
        "config_path": str(config_path),
    }

    logger.info(
        f"Feature data migration complete: {result['total_files']} file(s) migrated "
        f"from {result['total_features']} feature(s)"
    )

    emit_result(migration_report, args.output)

    return 0 if result["migrated_files"] else 1
