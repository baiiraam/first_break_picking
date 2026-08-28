"""
Training loop with MLflow, TensorBoard, and Loguru integration.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLflowManager
from src.config import SeismicConfig

logger = get_logger()


class SeismicTrainer:
    """
    Integrated trainer with MLflow, TensorBoard, and Loguru.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SeismicConfig
    ):
        self.config = config
        self.device = self._setup_device(config.device)
        
        # Setup model
        if config.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model, device_ids=config.gpu_ids)
            logger.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
        else:
            self.model = model
            logger.info(f"Single device: {self.device}")
        
        self.model = self.model.to(self.device)
        self.dataloaders = dataloaders
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = self._create_scheduler()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.dataset_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tb_dir = Path(config.tensorboard_log_dir) / config.dataset_name / "experiment"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        
        # MLflow
        self.mlflow_manager = MLflowManager(config.mlflow_experiment_name)
        
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"TensorBoard directory: {self.tb_dir}")
    
    def _setup_device(self, requested_device: str) -> torch.device:
        """Setup device with fallback."""
        if requested_device == "mps":
            if torch.backends.mps.is_available():
                try:
                    test = torch.ones(1, device="mps")
                    del test
                    logger.info("MPS device initialized successfully")
                    return torch.device("mps")
                except Exception as e:
                    logger.warning(f"MPS initialization failed: {e}, falling back to CPU")
                    return torch.device("cpu")
            else:
                logger.warning("MPS not available, falling back to CPU")
                return torch.device("cpu")
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                logger.info(f"CUDA device initialized: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.lr_patience,
                factor=self.config.lr_factor
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_T_max
            )
        return None
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint with full state restoration."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        try:
            target_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded successfully")
        except RuntimeError as e:
            logger.warning(f"Model state mismatch: {e}, loading with strict=False")
            target_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except ValueError as e:
            logger.warning(f"Optimizer state mismatch: {e}, using fresh optimizer state")
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded successfully")
            except ValueError as e:
                logger.warning(f"Scheduler state mismatch: {e}, using fresh scheduler state")
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']} with val_loss={checkpoint.get('val_loss', 'N/A')}")
        return checkpoint['epoch']
    
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        pbar = tqdm(self.dataloaders['train'], desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            # Ensure tensors are contiguous
            x = x.contiguous()
            y = y.contiguous()
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Loss computation on MPS (fast)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
            
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.dataloaders['train'])
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        pbar = tqdm(self.dataloaders['val'], desc="Validation")
        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            x = x.contiguous()
            y = y.contiguous()
            
            outputs = self.model(x)
            
            loss = self.criterion(outputs, y)
            
            total_loss += loss.item()
        
        return total_loss / len(self.dataloaders['val'])
    
    def _warmup_mps(self):
        """Warm up MPS shaders to avoid JIT compilation delay during training."""
        if self.device.type != "mps":
            return
        
        logger.info("🔥 Warming up MPS shaders (first pass can take 2-10 minutes)...")
        
        # Create dummy data with production shape
        dummy_x = torch.randn(4, 1, 1578, 751, device=self.device)
        dummy_y = torch.randint(0, 3, (4, 1578, 751), device=self.device)
        
        # Forward pass
        dummy_out = self.model(dummy_x)
        
        # Loss
        dummy_loss = self.criterion(dummy_out, dummy_y)
        
        # Backward pass (this triggers shader compilation)
        dummy_loss.backward()
        
        # Synchronize to ensure compilation completes
        torch.mps.synchronize()
        
        # Clear gradients and memory
        self.model.zero_grad()
        torch.mps.empty_cache()
        
        logger.info("✅ MPS warmup complete!")
    
    def fit(self, resume_from: Optional[str] = None):
        """Main training loop with integrated logging."""
        
        # Setup
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        config_dict = self.config.to_dict()
        
        # Start MLflow run
        self.mlflow_manager.start_run(config_dict)
        run_id = self.mlflow_manager.run_id
        
        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"TensorBoard: {self.tb_dir}")
        
        # Log model graph once
        if start_epoch == 0:
            sample_x = next(iter(self.dataloaders['train']))[0][:1].to(self.device)
            self.writer.add_graph(self.model, sample_x)
        
        # --- WARMUP MPS SHADERS ---
        self._warmup_mps()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config.n_epochs):
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} - Val Loss: {val_loss:.4f}")
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metrics/lr", current_lr, epoch)
            
            self.mlflow_manager.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1} - LR: {current_lr:.6f}")
            
            # Early stopping
            if self.config.early_stopping_patience is not None:
                if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    logger.info(f"New best val_loss: {best_val_loss:.4f}")
                    # Save best model
                    self._save_best_model(best_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, train_loss, val_loss)
        
        # Finalize
        self.writer.close()
        self.mlflow_manager.end_run()
        
        logger.info(f"Training completed! Best val_loss: {best_val_loss:.4f}")
        logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        logger.info(f"TensorBoard: {self.tb_dir}")
        logger.info(f"MLflow Run: {run_id}")
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save a checkpoint with full state."""
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        self.mlflow_manager.log_artifact(str(ckpt_path), artifact_path="checkpoints")
    
    def _save_best_model(self, best_val_loss: float):
        """Save the best model."""
        best_path = self.checkpoint_dir / "best_model.pt"
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'val_loss': best_val_loss,
            'config': self.config.to_dict()
        }, best_path)
        logger.info(f"Best model saved: {best_path} (val_loss: {best_val_loss:.4f})")