from pathlib import Path
from typing import Any

import pandas as pd

from core import get_logger, settings
from ml import prediction_service

from ..common import emit_result

logger = get_logger(__name__)


def cmd_predict(args: Any) -> int:
    if getattr(args, "model_id", None):
        prediction_service.load_model(args.model_id)

    if getattr(args, "url", None):
        urls = [args.url]
    elif getattr(args, "urls", None):
        urls = args.urls
    elif getattr(args, "url_file", None):
        with open(args.url_file, "r", encoding="utf-8") as handle:
            urls = [line.strip() for line in handle if line.strip()]
    else:
        logger.error("Must provide a URL to predict")
        return 1

    predictions = [prediction_service.execute_prediction({"url": url}) for url in urls]
    payload = predictions[0] if len(predictions) == 1 else {"predictions": predictions}
    emit_result(payload, getattr(args, "output", None))
    return 0


def cmd_predict_test(args: Any) -> int:
    urls = []
    if getattr(args, "url", None):
        urls.append(args.url)
    if getattr(args, "urls", None):
        urls.extend(args.urls)
    if getattr(args, "csv", None):
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = Path(settings.dataset_path).parent / "test" / args.csv
        frame = pd.read_csv(csv_path)
        first_column = frame.columns[0]
        urls.extend(frame[first_column].dropna().astype(str).tolist())

    if getattr(args, "model_id", None):
        prediction_service.load_model(args.model_id)

    payload = prediction_service.batch_predict({"urls": urls, "model_id": args.model_id})
    emit_result(payload, getattr(args, "output", None))
    return 0
