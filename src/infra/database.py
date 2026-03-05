from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import json
from datetime import datetime

from core import settings


class DatabaseClient:

    def __init__(self):
        self.connection_params = {
            'host': settings.postgres_host,
            'port': settings.postgres_port,
            'user': settings.postgres_user,
            'password': settings.postgres_password,
            'database': settings.postgres_db
        }

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_service_config(self, service_conf_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM service_conf 
                    WHERE service_conf_id = %s
                """, (service_conf_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_model_registry(self, model_registry_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM model_registry 
                    WHERE model_registry_id = %s
                """, (model_registry_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_model_by_service_conf(self, service_conf_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM model_registry 
                    WHERE service_conf_id = %s
                """, (service_conf_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def update_model_registry(
        self,
        model_registry_id: int,
        name: str,
        algorithm: str,
        mlflow_id: str,
        model_uri: str,
        scaler_uri: str,
        label_uri: str,
        accuracy_score: float,
        recall_score: float,
        precision_score: float,
        f1_score: float,
        config_json: Dict[str, Any]
    ) -> bool:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE model_registry 
                    SET name = %s,
                        algorithm = %s,
                        mlflow_id = %s,
                        model_uri = %s,
                        scaler_uri = %s,
                        label_uri = %s,
                        accuracy_score = %s,
                        recall_score = %s,
                        precision_score = %s,
                        f1_score = %s,
                        config_json = %s,
                        updated_at = NOW()
                    WHERE model_registry_id = %s
                """, (
                    name, algorithm, mlflow_id, model_uri, scaler_uri, label_uri,
                    accuracy_score, recall_score, precision_score, f1_score,
                    json.dumps(config_json), model_registry_id
                ))
                return cur.rowcount > 0

    def create_model_registry(
        self,
        service_conf_id: int,
        name: str,
        algorithm: str,
        mlflow_id: str,
        model_uri: str,
        scaler_uri: str,
        label_uri: str,
        accuracy_score: float,
        recall_score: float,
        precision_score: float,
        f1_score: float,
        config_json: Dict[str, Any]
    ) -> int:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_registry (
                        service_conf_id, name, algorithm, mlflow_id,
                        model_uri, scaler_uri, label_uri,
                        accuracy_score, recall_score, precision_score, f1_score,
                        config_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING model_registry_id
                """, (
                    service_conf_id, name, algorithm, mlflow_id,
                    model_uri, scaler_uri, label_uri,
                    accuracy_score, recall_score, precision_score, f1_score,
                    json.dumps(config_json)
                ))
                result = cur.fetchone()
                return result[0] if result else None

    def create_prediction(
        self,
        user_id: int,
        url: str,
        accuracy_score: float,
        recall_score: float,
        precision_score: float,
        f1_score: float,
        suggested_desc: str
    ) -> int:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prediction (
                        user_id, url, accuracy_score, recall_score,
                        precision_score, f1_score, suggested_desc
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                """, (
                    user_id, url, accuracy_score, recall_score,
                    precision_score, f1_score, suggested_desc
                ))
                result = cur.fetchone()
                return result[0] if result else None


db_client = DatabaseClient()
