from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from core import features_config, settings
from features.schema import FeatureSchema, build_feature_schema
from tracking.registry import ModelReference, ModelRegistryError, Registry


class ModelValidationError(ModelRegistryError):
    pass


@dataclass(frozen=True)
class GateResult:
    metric: str
    operator: str
    threshold: float
    actual: Optional[float]
    passed: bool
    source: str


class Promotion:
    """Promotion policy on top of a Registry: gate evaluation, champion
    comparison, smoke tests, and the promote/rollback-adjacent state
    transitions. Gate thresholds and whether champion comparison is required
    come from settings (`MODEL_PROMOTION_GATES`, `PROMOTION_REQUIRE_CHAMPION_COMPARISON`),
    so promotion behavior is policy-driven and configurable without code changes.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def validate_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        alias = self.registry.candidate_alias if model_version is None else None
        reference = self.registry.get_reference(alias=alias, version=model_version)
        validation = self._validate_reference(reference)
        validation["model_alias"] = reference.alias
        validation["model_version"] = reference.version
        validation["model_name"] = reference.name
        return validation

    def promote_candidate(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        alias = self.registry.candidate_alias if model_version is None else None
        reference = self.registry.get_reference(alias=alias, version=model_version)
        validation = self._validate_reference(reference)
        if not validation["passed"]:
            raise ModelValidationError("Candidate validation failed")

        previous = self.registry.safe_get_reference(alias=self.registry.champion_alias)
        if previous is not None:
            self.registry.client.set_registered_model_alias(
                self.registry.model_name, self.registry.previous_champion_alias, previous.version
            )
            self.registry.client.set_model_version_tag(
                self.registry.model_name,
                previous.version,
                "replaced_at",
                self.registry.utc_now(),
            )

        self.registry.client.set_registered_model_alias(
            self.registry.model_name, self.registry.champion_alias, reference.version
        )
        metadata = {
            "promoted_at": self.registry.utc_now(),
            "promotion_status": "approved",
            "promoted_from_alias": reference.alias or "explicit-version",
            "previous_champion_version": previous.version if previous is not None else "",
            "validation_summary": json.dumps(
                {
                    "gates_passed": validation["gates_passed"],
                    "champion_comparison_passed": validation["champion_comparison_passed"],
                    "smoke_tests_passed": validation["smoke_tests_passed"],
                },
                sort_keys=True,
            ),
        }
        for key, value in metadata.items():
            self.registry.client.set_model_version_tag(self.registry.model_name, reference.version, key, value)

        return {
            "model_name": self.registry.model_name,
            "model_version": reference.version,
            "model_alias": self.registry.champion_alias,
            "previous_champion_version": previous.version if previous is not None else None,
            "validation": validation,
        }

    def _validate_reference(self, reference: ModelReference) -> Dict[str, Any]:
        run = self.registry.client.get_run(reference.run_id)
        feature_schema = self.registry.load_json_artifact(reference.run_id, "artifacts/feature_schema.json")
        dataset_metadata = self.registry.load_json_artifact(reference.run_id, "artifacts/dataset_metadata.json")
        sample_predictions = self.registry.load_json_artifact(reference.run_id, "artifacts/sample_predictions.json")

        self._validate_feature_schema(feature_schema)
        self._validate_dataset_metadata(dataset_metadata)

        gate_results = self._evaluate_gates(run.data.metrics, source="candidate")
        champion = self.registry.safe_get_reference(alias=self.registry.champion_alias)
        champion_results = self._compare_to_champion(run.data.metrics, champion)
        smoke_test_results = self._run_smoke_tests(reference, sample_predictions)

        passed = all(result.passed for result in gate_results)
        comparison_passed = all(result.passed for result in champion_results)
        smoke_passed = all(item["passed"] for item in smoke_test_results)
        return {
            "passed": passed and comparison_passed and smoke_passed,
            "feature_schema_version": feature_schema["schema_version"],
            "dataset_version": dataset_metadata["dataset_version"],
            "dataset_hash": dataset_metadata["dataset_hash"],
            "gate_results": [result.__dict__ for result in gate_results],
            "gates_passed": passed,
            "champion_comparison": [result.__dict__ for result in champion_results],
            "champion_comparison_passed": comparison_passed,
            "smoke_tests": smoke_test_results,
            "smoke_tests_passed": smoke_passed,
            "current_champion_version": champion.version if champion is not None else None,
        }

    def _compare_to_champion(
        self, candidate_metrics: Mapping[str, Any], champion: Optional[ModelReference]
    ) -> List[GateResult]:
        if champion is None or not settings.promotion_require_champion_comparison:
            return []

        run = self.registry.client.get_run(champion.run_id)
        champion_metrics = run.data.metrics
        results: List[GateResult] = []
        for metric, rule in settings.parsed_model_promotion_gates.items():
            candidate_value = self.registry.coerce_metric(candidate_metrics.get(metric))
            champion_value = self.registry.coerce_metric(champion_metrics.get(metric))
            if candidate_value is None or champion_value is None:
                results.append(
                    GateResult(
                        metric=metric,
                        operator="compare",
                        threshold=champion_value or 0.0,
                        actual=candidate_value,
                        passed=False,
                        source="champion",
                    )
                )
                continue

            operator = str(rule["operator"])
            passed = candidate_value >= champion_value if operator == ">=" else candidate_value <= champion_value
            results.append(
                GateResult(
                    metric=metric,
                    operator=operator,
                    threshold=champion_value,
                    actual=candidate_value,
                    passed=passed,
                    source="champion",
                )
            )
        return results

    def _run_smoke_tests(
        self, reference: ModelReference, sample_predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        loader = self.registry.load_reference(version=reference.version)
        pipeline = loader["pipeline"]
        urls = settings.parsed_promotion_smoke_test_urls
        if sample_predictions:
            urls = [item["url"] for item in sample_predictions[:2] if item.get("url")] + urls

        results = []
        for url in urls[:4]:
            prediction = pipeline.predict(url)
            results.append(
                {
                    "url": url,
                    "passed": prediction["prediction"] in {"benign", "malicious"},
                    "model_version": reference.version,
                    "prediction": prediction["prediction"],
                }
            )
        return results

    def _validate_feature_schema(self, payload: Dict[str, Any]) -> None:
        if "schema_version" not in payload or "features" not in payload:
            raise ModelValidationError("Candidate artifact is missing feature schema metadata")
        runtime_schema = build_feature_schema(features_config)
        artifact_schema = FeatureSchema.from_dict(payload)
        if artifact_schema.schema_version != runtime_schema.schema_version:
            raise ModelValidationError(
                "Candidate feature schema mismatch: "
                f"runtime={runtime_schema.schema_version}, candidate={artifact_schema.schema_version}"
            )

    def _validate_dataset_metadata(self, payload: Dict[str, Any]) -> None:
        required = {"dataset_version", "dataset_hash", "total_records", "source_references"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ModelValidationError(f"Candidate dataset metadata is missing required keys: {', '.join(missing)}")

    def _evaluate_gates(self, metrics: Mapping[str, Any], source: str) -> List[GateResult]:
        results: List[GateResult] = []
        for metric, rule in settings.parsed_model_promotion_gates.items():
            actual = self.registry.coerce_metric(metrics.get(metric))
            operator = str(rule["operator"])
            threshold = float(rule["threshold"])
            passed = False
            if actual is not None:
                passed = actual >= threshold if operator == ">=" else actual <= threshold
            results.append(
                GateResult(
                    metric=metric,
                    operator=operator,
                    threshold=threshold,
                    actual=actual,
                    passed=passed,
                    source=source,
                )
            )
        return results
