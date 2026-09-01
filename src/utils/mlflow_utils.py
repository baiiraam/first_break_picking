"""
MLflow integration for experiment tracking, model registry, and checkpoint management.
Supports:
- Autologging for PyTorch
- System metrics logging (GPU/CPU)
- Model registry with versioning and aliases
- Checkpoint tracking with step parameter
- Search and comparison of logged models
"""

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.pytorch
from loguru import logger


class MLflowManager:
    """
    Advanced MLflow manager with full feature support.
    """

    def __init__(
        self,
        experiment_name: str = "seismic-fbp",
        tracking_uri: str | None = None,
        enable_system_metrics: bool = True,
        enable_autolog: bool = True,
        autolog_config: dict[str, Any] | None = None,
    ):
        """
        Initialize MLflow manager with all features.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: Tracking server URI (None = local SQLite)
            enable_system_metrics: Enable GPU/CPU metrics logging
            enable_autolog: Enable PyTorch autologging
            autolog_config: Custom autolog configuration
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to SQLite for local development
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            # Allow file store (for backward compatibility)
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")

        # Enable system metrics logging
        if enable_system_metrics:
            try:
                mlflow.enable_system_metrics_logging()
                logger.info("System metrics logging enabled (GPU/CPU monitoring)")
            except (mlflow.MlflowException, OSError) as e:
                logger.warning(f"System metrics logging failed: {e}")

        # Enable PyTorch autologging
        if enable_autolog:
            default_config = {
                "log_models": True,
                "log_every_n_epoch": 1,
                "log_every_n_step": 10,
            }
            if autolog_config:
                default_config.update(autolog_config)
            try:
                mlflow.pytorch.autolog(**default_config)
                logger.info(
                    f"PyTorch autologging enabled with config: {default_config}"
                )
            except (mlflow.MlflowException, OSError) as e:
                logger.warning(f"PyTorch autologging failed: {e}")

        self.current_run = None
        self.run_id = None
        self.client = mlflow.MlflowClient()

    def start_run(
        self,
        config_dict: dict[str, Any],
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new MLflow run with configuration tracking.

        Args:
            config_dict: Configuration dictionary (will be logged as params)
            run_name: Optional run name (auto-generated from config hash)
            tags: Optional tags for the run (e.g., dataset, model type)

        Returns:
            run_id: The ID of the started run
        """
        # Generate unique run name from config
        if run_name is None:
            config_hash = hashlib.md5(
                json.dumps(config_dict, sort_keys=True).encode()
            ).hexdigest()[:8]
            run_name = f"seismic_{config_hash}"

        self.current_run = mlflow.start_run(run_name=run_name)
        self.run_id = self.current_run.info.run_id

        # Log all configuration parameters
        mlflow.log_params(config_dict)

        # Log tags (including dataset, model type, etc.)
        if tags:
            mlflow.set_tags(tags)

        # Log git info if available
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL  # Suppress git errors
            ).decode().strip()
            mlflow.set_tag("git_commit", git_commit)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Could not get git commit: {e}")
            # Not critical, continue

        # Add timestamp tag
        mlflow.set_tag("start_time", datetime.now(timezone.utc).isoformat())

        logger.info(f"MLflow run started: {run_name} (ID: {self.run_id})")
        return self.run_id

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log an artifact file."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_model_with_registry(
        self,
        model,
        model_name: str,
        dataset_name: str,
        step: int | None = None,
        registered_model_name: str | None = None,
        input_example: Any | None = None,
        signature: Any | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Log a PyTorch model with registry support and checkpoint tracking."""

        # Build the model name
        if step is not None:
            full_model_name = f"{model_name}_epoch_{step}"
        else:
            full_model_name = model_name

        # Remove invalid characters
        full_model_name = full_model_name.replace("/", "_").replace(":", "_")

        logger.info("📦 Attempting to log model to MLflow:")
        logger.info(f"   Model: {full_model_name}")
        logger.info(f"   Dataset: {dataset_name}")
        logger.info(f"   Step: {step}")
        logger.info(f"   Registered model: {registered_model_name}")

        # Ensure input_example is on CPU as numpy array
        if input_example is not None and hasattr(input_example, "cpu"):
            input_example = input_example.cpu().numpy()

        # Log the model with pickle serialization
        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=full_model_name,
                registered_model_name=registered_model_name,
                input_example=input_example,
                signature=signature,
                serialization_format="pickle",  # ← CRITICAL FIX
            )
            logger.info("✅ Model logged successfully!")
            logger.info(f"   Model ID: {model_info.model_id}")
            logger.info(f"   Model URI: {model_info.model_uri}")
        except (mlflow.MlflowException, OSError) as e:
            logger.error(f"❌ MLflow logging failed: {e}")
            logger.info("Attempting fallback without registry...")
            try:
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name=full_model_name,
                    serialization_format="pickle",
                )
                logger.info(f"✅ Fallback successful: {model_info.model_id}")
            except Exception as e2:
                logger.error(f"❌ Fallback also failed: {e2}")
                raise

        # Add tags to the model version if registered
        if registered_model_name and model_info.registered_model_version:
            version = model_info.registered_model_version
            model_version_tags = {
                "dataset": dataset_name,
                "model_type": model_name.split("_")[0]
                if "_" in model_name
                else model_name,
                "step": str(step) if step is not None else "final",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if tags:
                model_version_tags.update(tags)

            try:
                self.client.set_model_version_tags(
                    name=registered_model_name,
                    version=version,
                    tags=model_version_tags,
                )
                logger.info(
                    f"Added tags to model version {version} of {registered_model_name}"
                )
            except (mlflow.MlflowException, OSError) as e:
                logger.warning(f"Failed to add tags to model version: {e}")

        result = {
            "model_id": model_info.model_id,
            "model_uri": model_info.model_uri,
        }
        if hasattr(model_info, "registered_model_version"):
            result["registered_model_version"] = model_info.registered_model_version

        return result

    def set_model_alias(
        self,
        registered_model_name: str,
        alias: str,
        version: int | str,
    ):
        """
        Set an alias for a registered model version.

        Args:
            registered_model_name: Name of the registered model
            alias: Alias name (e.g., "champion", "challenger", "staging")
            version: Model version number or "latest"
        """
        try:
            self.client.set_registered_model_alias(
                name=registered_model_name,
                alias=alias,
                version=version,
            )
            logger.info(
                f"Set alias '{alias}' = version {version} for {registered_model_name}"
            )
        except (mlflow.MlflowException, OSError) as e:
            logger.warning(f"Failed to set alias: {e}")

    def get_model_by_alias(
        self,
        registered_model_name: str,
        alias: str,
    ):
        """
        Get a model version by alias.

        Args:
            registered_model_name: Name of the registered model
            alias: Alias name (e.g., "champion")

        Returns:
            Model version info
        """
        try:
            model_info = self.client.get_model_version_by_alias(
                name=registered_model_name,
                alias=alias,
            )
            logger.info(
                f"Found model {registered_model_name} with alias '{alias}': version {model_info.version}"
            )
            return model_info
        except (mlflow.MlflowException, OSError) as e:
            logger.warning(f"Failed to get model by alias: {e}")
            return None

    def search_models(
        self,
        filter_string: str | None = None,
        order_by: list[dict[str, str]] | None = None,
        max_results: int = 10,
        output_format: str = "list",
    ) -> list[Any]:
        """
        Search and compare logged models.

        Args:
            filter_string: SQL-like filter (e.g., "metrics.val_iou > 0.65")
            order_by: Order criteria (e.g., [{"field_name": "metrics.val_iou", "ascending": False}])
            max_results: Maximum number of results
            output_format: "list" or "pandas"

        Returns:
            List of model objects or pandas DataFrame

        Examples:
            # Find best models for Halfmile dataset
            best_models = mlflow_manager.search_models(
                filter_string="params.dataset = 'Halfmile'",
                order_by=[{"field_name": "metrics.val_iou", "ascending": False}],
                max_results=5,
            )

            # Find models with high class 2 IoU
            best_strip_models = mlflow_manager.search_models(
                filter_string="metrics.class_2_iou > 0.2",
                order_by=[{"field_name": "metrics.class_2_iou", "ascending": False}],
            )
        """
        try:
            results = mlflow.search_logged_models(
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
                output_format=output_format,
            )
            logger.info(f"Search returned {len(results)} models")
            return results
        except (mlflow.MlflowException, OSError) as e:
            logger.warning(f"Search failed: {e}")
            return []

    def load_model_from_uri(self, model_uri: str):
        """
        Load a model from a URI.

        Args:
            model_uri: URI in format "models:/{model_id}" or "models:/{name}/{version}"

        Returns:
            Loaded model
        """
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
        except (mlflow.MlflowException, OSError) as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def load_pytorch_model(self, model_uri: str):
        """Load a PyTorch model from a URI."""
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded PyTorch model from {model_uri}")
            return model
        except (mlflow.MlflowException, OSError) as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return None

    def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        """Get all metrics from a specific run."""
        try:
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags
            return {
                "metrics": metrics,
                "params": params,
                "tags": tags,
                "info": run.info,
            }
        except (mlflow.MlflowException, OSError) as e:
            logger.warning(f"Failed to get run metrics: {e}")
            return {}

    def compare_runs(
        self,
        run_ids: list[str],
        metric_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metric_names: List of metric names to compare

        Returns:
            Dict mapping run_id to {metric_name: value}
        """
        results = {}
        for run_id in run_ids:
            run_data = self.get_run_metrics(run_id)
            if run_data:
                results[run_id] = {
                    name: float(run_data["metrics"].get(name, float("nan")))
                    for name in metric_names
                }
        return results

    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.set_tag("end_time", datetime.now(timezone.utc).isoformat())
            mlflow.end_run()
            logger.info("MLflow run ended")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


# ============================================================
# Utility Functions
# ============================================================


def get_mlflow_manager(
    experiment_name: str = "seismic-fbp",
    tracking_uri: str | None = None,
    enable_system_metrics: bool = True,
    enable_autolog: bool = True,
) -> MLflowManager:
    """Get or create an MLflowManager instance."""
    return MLflowManager(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        enable_system_metrics=enable_system_metrics,
        enable_autolog=enable_autolog,
    )


def format_model_name(
    model_type: str,
    dataset_name: str,
    suffix: str | None = None,
) -> str:
    """Format a model name consistently."""
    parts = [model_type, dataset_name]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def format_registered_model_name(dataset_name: str) -> str:
    """Format a registered model name."""
    return dataset_name.lower()
