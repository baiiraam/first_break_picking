#!/usr/bin/env python3
"""
MLflow Sweep Script
Runs grid search over datasets, models, and loss functions with MLflow tracking.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger
from src.utils.mlflow_utils import get_mlflow_manager

logger = setup_logger(task_name="sweep_mlflow")


class SweepExperiment:
    """Manage sweep experiments with MLflow tracking."""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.global_config = self.config.get("global", {})
        self.sweep_config = self.config.get("sweep", {})
        self.tracking_config = self.config.get("tracking", {})
        
        # Initialize MLflow
        if self.tracking_config.get("enabled", True):
            self.mlflow_manager = get_mlflow_manager(
                experiment_name=self.tracking_config.get("experiment_name", "model_loss_sweep"),
                enable_system_metrics=True,
                enable_autolog=self.tracking_config.get("autolog", {}).get("enabled", True),
            )
            logger.info("✅ MLflow tracking enabled")
        else:
            self.mlflow_manager = None
            logger.info("ℹ️ MLflow tracking disabled")
    
    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_experiment(
        self,
        dataset: str,
        model: str,
        loss: str,
        loss_params: dict,
        experiment_id: int,
        total_experiments: int,
    ) -> dict:
        """Run a single experiment with MLflow tracking."""
        
        experiment_name = f"{dataset}_{model}_{loss}"
        logger.info(f"\n[{experiment_id}/{total_experiments}] 🔬 {experiment_name}")
        
        # Build command
        cmd = self.build_command(dataset, model, loss, loss_params)
        
        start_time = time.time()
        
        try:
            # Start MLflow run
            if self.mlflow_manager:
                run_id = self.mlflow_manager.start_run(
                    config_dict={
                        "dataset": dataset,
                        "model": model,
                        "loss": loss,
                        **loss_params,
                        **self.global_config,
                    },
                    run_name=experiment_name,
                    tags={
                        "dataset": dataset,
                        "model": model,
                        "loss": loss,
                        "experiment_type": "sweep",
                        "sweep_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                        **self.tracking_config.get("tags", {}),
                    },
                )
                logger.info(f"   MLflow Run ID: {run_id}")
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            duration = time.time() - start_time
            
            # Parse metrics from output
            metrics = self.parse_metrics(result.stdout + result.stderr)
            
            # Log metrics to MLflow
            if self.mlflow_manager and metrics:
                self.mlflow_manager.log_metrics(metrics, step=0)
                logger.info(f"   📊 Metrics: {metrics}")
            
            # Log artifacts
            if self.mlflow_manager:
                self.mlflow_manager.log_artifact("configs/batch_config.yaml", artifact_path="configs")
                
                model_path = Path(f"models/registry/*{model}_{dataset}*.pt")
                if list(model_path.parent.glob(f"*{model}_{dataset}*.pt")):
                    self.mlflow_manager.log_artifact(str(model_path), artifact_path="models")
            
            success = result.returncode == 0
            
            if success:
                logger.info(f"   ✅ SUCCESS! Duration: {duration:.1f}s")
            else:
                logger.warning(f"   ❌ FAILED! Duration: {duration:.1f}s")
                if result.stderr:
                    logger.warning(f"   Error: {result.stderr[:200]}")
            
            return {
                "success": success,
                "dataset": dataset,
                "model": model,
                "loss": loss,
                "run_id": self.mlflow_manager.run_id if self.mlflow_manager else None,
                "duration": duration,
                "return_code": result.returncode,
                "metrics": metrics,
                "error": result.stderr if not success else None,
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"   ❌ TIMEOUT after {self.global_config.get('timeout_seconds', 3600)}s")
            return {
                "success": False,
                "dataset": dataset,
                "model": model,
                "loss": loss,
                "duration": time.time() - start_time,
                "error": "Timeout",
                "metrics": {},
            }
        except Exception as e:
            logger.error(f"   ❌ ERROR: {e}")
            return {
                "success": False,
                "dataset": dataset,
                "model": model,
                "loss": loss,
                "duration": time.time() - start_time,
                "error": str(e),
                "metrics": {},
            }
        finally:
            # End MLflow run
            if self.mlflow_manager:
                self.mlflow_manager.end_run()
    
    def build_command(self, dataset: str, model: str, loss: str, loss_params: dict) -> list:
        """Build the training command."""
        cmd = [
            "python3.12",
            "scripts/train.py",
            "--config", f"configs/{dataset.lower()}.yaml",
            "--model", model,
            "--epochs", str(self.global_config.get("epochs", 2)),
            "--loss", loss,
            "--device", self.global_config.get("device", "mps"),
        ]
        
        # Add class weights
        if "class_weights" in loss_params:
            weights = [str(w) for w in loss_params["class_weights"]]
            cmd.append("--class-weights")
            cmd.extend(weights)
        
        # Add loss-specific params
        if loss == "combo":
            cmd.extend(["--dice-weight", str(loss_params.get("dice_weight", 0.5))])
            cmd.extend(["--focal-gamma", str(loss_params.get("focal_gamma", 2.0))])
        elif loss == "focal":
            cmd.extend(["--focal-gamma", str(loss_params.get("focal_gamma", 2.0))])
        
        # Verbose
        if self.global_config.get("verbose"):
            cmd.append("--verbose")
        if self.global_config.get("log_memory"):
            cmd.append("--log-memory")
        
        return cmd

    # ✅ Fixed: Added @staticmethod decorator
    @staticmethod
    def parse_metrics(output: str) -> dict:
        """Parse metrics from training output."""
        metrics = {}
        
        for line in output.split('\n'):
            if "Train Loss:" in line:
                try:
                    metrics["train_loss"] = float(line.split("Train Loss:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            
            if "Val Loss:" in line:
                try:
                    metrics["val_loss"] = float(line.split("Val Loss:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            
            if "Train IoU:" in line:
                try:
                    metrics["train_iou"] = float(line.split("Train IoU:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            
            if "Val IoU:" in line:
                try:
                    metrics["val_iou"] = float(line.split("Val IoU:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            
            if "Train Acc:" in line:
                try:
                    metrics["train_acc"] = float(line.split("Train Acc:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            
            if "Val Acc:" in line:
                try:
                    metrics["val_acc"] = float(line.split("Val Acc:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        
        return metrics
    
    def run_sweep(self):
        """Run the full sweep."""
        
        datasets = self.sweep_config.get("datasets", [])
        models = self.sweep_config.get("models", [])
        losses = self.sweep_config.get("losses", [])
        loss_params = self.sweep_config.get("loss_params", {})
        
        total_experiments = len(datasets) * len(models) * len(losses)
        experiment_id = 0
        
        results = []
        failed = []
        
        logger.info("=" * 80)
        logger.info("🧪 MLflow SWEEP")
        logger.info("=" * 80)
        logger.info(f"Datasets: {len(datasets)}")
        logger.info(f"Models: {len(models)}")
        logger.info(f"Losses: {len(losses)}")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("=" * 80)
        
        for dataset in datasets:
            for model in models:
                for loss in losses:
                    experiment_id += 1
                    
                    result = self.run_experiment(
                        dataset=dataset,
                        model=model,
                        loss=loss,
                        loss_params=loss_params.get(loss, {}),
                        experiment_id=experiment_id,
                        total_experiments=total_experiments,
                    )
                    
                    results.append(result)
                    
                    if not result["success"]:
                        failed.append({
                            "dataset": dataset,
                            "model": model,
                            "loss": loss,
                            "error": result.get("error"),
                        })
                    
                    # Save checkpoint every 10 experiments
                    if experiment_id % 10 == 0:
                        self.save_checkpoint(results, failed)
        
        # Final summary
        self.print_summary(results, failed, total_experiments)
        
        return results
    
    def save_checkpoint(self, results: list, failed: list):
        """Save checkpoint to resume later."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_file = Path("logs/sweep") / f"sweep_checkpoint_{timestamp}.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_file, "w") as f:
            json.dump({
                "results": results,
                "failed": failed,
                "timestamp": timestamp,
                "total": len(results) + len(failed),
            }, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def print_summary(self, results: list, failed: list, total: int):
        """Print experiment summary."""
        
        successful = [r for r in results if r["success"]]
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 SWEEP SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {total}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            best = max(successful, key=lambda x: x.get("metrics", {}).get("val_iou", 0))
            logger.info("\n🏆 Best Experiment:")
            logger.info(f"  Dataset: {best['dataset']}")
            logger.info(f"  Model: {best['model']}")
            logger.info(f"  Loss: {best['loss']}")
            logger.info(f"  MLflow Run: {best.get('run_id', 'N/A')}")
            logger.info(f"  Val IoU: {best.get('metrics', {}).get('val_iou', 'N/A')}")
            logger.info(f"  Val Loss: {best.get('metrics', {}).get('val_loss', 'N/A')}")
        
        if failed:
            logger.info(f"\n❌ Failed Experiments ({len(failed)}):")
            for f in failed[:10]:
                logger.info(f"  • {f['dataset']} | {f['model']} | {f['loss']}: {f.get('error', 'Unknown')[:100]}")
            if len(failed) > 10:
                logger.info(f"  ... and {len(failed) - 10} more")
        
        logger.info("\n📊 MLflow UI:")
        logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow.db")


@click.command()
@click.option("--config", "-c", default="configs/sweep_config.yaml", help="Sweep config file")
def main(config: str):
    """Run MLflow sweep."""
    sweep = SweepExperiment(config)
    sweep.run_sweep()


if __name__ == "__main__":
    main()
    