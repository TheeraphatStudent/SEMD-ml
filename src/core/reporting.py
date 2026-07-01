import json
from pathlib import Path
from typing import Any, Optional


def normalize_json_filename(filename: str) -> str:
    return filename if filename.endswith('.json') else f'{filename}.json'


def write_json_result(
    data: Any,
    output_filename: str,
    reports_dir: str,
    logger: Optional[Any] = None,
) -> Path:
    output_path = Path(reports_dir) / normalize_json_filename(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    if logger:
        logger.info(f'Results saved to {output_path}')
    return output_path


def emit_json_result(
    data: Any,
    output_filename: Optional[str],
    reports_dir: str,
    logger: Optional[Any] = None,
) -> None:
    if output_filename:
        write_json_result(data, output_filename, reports_dir, logger)
    else:
        print(json.dumps(data, indent=2))
