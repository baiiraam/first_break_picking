"""
Training callbacks for Seismic FBP.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from loguru import logger
import torch
from pathlib import Path


class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        pass
    
    @abstractmethod
    def on_batch_start(self, batch: int, **kwargs):
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback that monitors validation loss.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        current_value = metrics.get('val_loss', metrics.get('loss'))
        if current_value is None:
            return
        
        if self.best_value is None:
            self.best_value = current_value
            if self.verbose:
                logger.info(f"Early stopping: initial best {self.best_value:.4f}")
            return
        
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                logger.info(f"Early stopping: improved to {self.best_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping: {self.counter}/{self.patience} no improvement")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
    
    def on_batch_start(self, batch: int, **kwargs):
        pass
    
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    """
    Saves model checkpoints during training.
    """
    
    def __init__(
        self,
        save_dir: Path,
        save_best: bool = True,
        save_every: int = 5,
        mode: str = 'min',
        monitor: str = 'val_loss',
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        self.mode = mode
        self.monitor = monitor
        self.verbose = verbose
        
        self.best_value = None
        self.best_path = None
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model, optimizer, scheduler, **kwargs):
        # Save checkpoint every N epochs
        if (epoch + 1) % self.save_every == 0:
            ckpt_path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                **metrics
            }, ckpt_path)
            if self.verbose:
                logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Save best model
        if self.save_best:
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                if self.best_value is None:
                    self.best_value = current_value
                    self._save_best(model, epoch + 1, metrics)
                elif self.mode == 'min' and current_value < self.best_value:
                    self.best_value = current_value
                    self._save_best(model, epoch + 1, metrics)
                elif self.mode == 'max' and current_value > self.best_value:
                    self.best_value = current_value
                    self._save_best(model, epoch + 1, metrics)
    
    def _save_best(self, model, epoch, metrics):
        """Save the best model."""
        self.best_path = self.save_dir / "best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            **metrics
        }, self.best_path)
        if self.verbose:
            logger.info(f"Best model saved: {self.best_path} ({self.monitor}: {self.best_value:.4f})")
    
    def on_batch_start(self, batch: int, **kwargs):
        pass
    
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass


class LoggingCallback(Callback):
    """
    Logs metrics during training.
    """
    
    def __init__(
        self,
        writer=None,
        mlflow_manager=None,
        log_every: int = 10,
        verbose: bool = True
    ):
        self.writer = writer
        self.mlflow_manager = mlflow_manager
        self.log_every = log_every
        self.verbose = verbose
        self.batch_losses = []
    
    def on_epoch_start(self, epoch: int, **kwargs):
        self.batch_losses = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
        
        # Log to MLflow
        if self.mlflow_manager:
            self.mlflow_manager.log_metrics(metrics, step=epoch)
        
        if self.verbose:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"Epoch {epoch + 1}: {metrics_str}")
    
    def on_batch_start(self, batch: int, **kwargs):
        pass
    
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        self.batch_losses.append(loss)
        
        if (batch + 1) % self.log_every == 0:
            avg_loss = sum(self.batch_losses[-self.log_every:]) / self.log_every
            if self.verbose:
                logger.debug(f"Batch {batch + 1}: avg loss={avg_loss:.4f}")