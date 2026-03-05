from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ml.ml_pipeline import ml_pipeline
from infra import db_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:

    def __init__(self):
        self.current_model_id = None

    def load_model(self, run_id: str) -> bool:
        try:
            success = ml_pipeline.load_artifacts(run_id)
            if success:
                self.current_model_id = run_id
                logger.info(f"Model loaded successfully: {run_id}")
            return success
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def execute_prediction(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting prediction job: {job_data}")

        try:
            url = job_data.get('url')
            user_id = job_data.get('user_id')
            model_id = job_data.get('model_id')
            compare = job_data.get('compare', False)

            if not url:
                raise ValueError('No URL provided for prediction')

            if model_id and model_id != self.current_model_id:
                logger.info(f"Loading model: {model_id}")
                if not self.load_model(model_id):
                    raise ValueError(f"Failed to load model: {model_id}")

            if ml_pipeline.best_model is None:
                raise ValueError(
                    'No model loaded. Please train or load a model first.')

            logger.info(f"Predicting URL: {url}")
            prediction_result = ml_pipeline.predict(url)

            print(f"Prediction result: {prediction_result}\n")

            predicted_class = prediction_result['predicted_class']
            confidence = prediction_result['confidence']
            class_probabilities = prediction_result['class_probabilities']
            features = prediction_result.get('features', {}) if compare else None

            is_malicious = predicted_class != 'benign'

            suggested_desc = self._generate_suggestion(
                predicted_class, confidence, class_probabilities)

            result = {
                'status': 'success',
                'url': url,
                'prediction': {
                    'class': predicted_class,
                    'is_malicious': is_malicious,
                    'confidence': confidence,
                    'probabilities': class_probabilities
                },
                'suggested_desc': suggested_desc,
                'timestamp': datetime.now().isoformat(),
                'model_id': self.current_model_id
            }

            if compare and features:
                result['features'] = features

            if user_id:
                try:
                    model_registry = db_client.get_model_registry(
                        int(self.current_model_id.split('_')[-1])) if self.current_model_id else None

                    prediction_id = db_client.create_prediction(
                        user_id=user_id,
                        url=url,
                        accuracy_score=model_registry['accuracy_score'] if model_registry else confidence,
                        recall_score=model_registry['recall_score'] if model_registry else 0,
                        precision_score=model_registry['precision_score'] if model_registry else 0,
                        f1_score=model_registry['f1_score'] if model_registry else 0,
                        suggested_desc=suggested_desc
                    )

                    result['prediction_id'] = prediction_id
                    logger.info(
                        f"Saved prediction to database: {prediction_id}")

                except Exception as e:
                    logger.warning(
                        f"Could not save prediction to database: {str(e)}")

            logger.info(
                f"Prediction completed: {predicted_class} (confidence: {confidence:.4f})")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)

            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def batch_predict(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting batch prediction job")

        try:
            urls = job_data.get('urls', [])
            user_id = job_data.get('user_id')
            model_id = job_data.get('model_id')

            if not urls:
                raise ValueError('No URLs provided for batch prediction')

            if model_id and model_id != self.current_model_id:
                if not self.load_model(model_id):
                    raise ValueError(f"Failed to load model: {model_id}")

            results = []
            for url in urls:
                prediction_job = {
                    'url': url,
                    'user_id': user_id,
                    'model_id': model_id
                }
                result = self.execute_prediction(prediction_job)
                results.append(result)

            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful

            return {
                'status': 'success',
                'total': len(urls),
                'successful': successful,
                'failed': failed,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)

            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_suggestion(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float]
    ) -> str:
        if predicted_class == 'benign':
            if confidence > 0.9:
                return 'This URL appears to be safe with high confidence.'
            elif confidence > 0.7:
                return 'This URL appears to be safe, but exercise caution.'
            else:
                return 'This URL may be safe, but confidence is low. Verify before accessing.'

        elif predicted_class == 'phishing':
            return f"Warning: This URL is likely a phishing attempt (confidence: {confidence:.2%}). Do not enter personal information or credentials."

        elif predicted_class == 'malware':
            return f"Danger: This URL may contain malware (confidence: {confidence:.2%}). Avoid accessing this link."

        elif predicted_class == 'redirect':
            return f"Caution: This URL appears to be a redirect (confidence: {confidence:.2%}). It may lead to malicious content."

        elif predicted_class == 'spam':
            return f"This URL is likely spam (confidence: {confidence:.2%}). It may be unwanted or misleading content."

        else:
            return f"This URL has been classified as {predicted_class} with {confidence:.2%} confidence."


prediction_service = PredictionService()
