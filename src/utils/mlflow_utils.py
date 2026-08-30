"""
MLflow integration for experiment tracking.
"""

import os
import mlflow
import mlflow.pytorch
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class MLflowManager:
    """Manages MLflow experiments and runs."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
        
        self.current_run = None
        self.run_id = None
    
    def start_run(self, config_dict: Dict[str, Any], run_name: Optional[str] = None) -> str:
        """Start a new MLflow run with configuration tracking."""
        
        # Generate unique run name from config
        if run_name is None:
            config_hash = hashlib.md5(
                json.dumps(config_dict, sort_keys=True).encode()
            ).hexdigest()[:8]
            run_name = f"seismic_{config_hash}"
        
        self.current_run = mlflow.start_run(run_name=run_name)
        self.run_id = self.current_run.info.run_id
        
        # Log all parameters
        mlflow.log_params(config_dict)
        
        # Log git info if available
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            mlflow.set_tag("git_commit", git_commit)
        except:
            pass
        
        logger.info(f"MLflow run started: {run_name} (ID: {self.run_id})")
        return self.run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model, artifact_path: str = "model", registered_name: Optional[str] = None):
        """Log PyTorch model to MLflow."""
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_name
        )
        logger.info(f"Model logged to MLflow: {artifact_path}")
    
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            logger.info("MLflow run ended")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()