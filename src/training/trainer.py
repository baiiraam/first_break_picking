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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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
        config: SeismicConfig,
        model_name: str = "unet"
    ):
        self.config = config
        self.model_name = model_name
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
        
        # Model Registry Directory (centralized)
        self.registry_dir = Path(config.model_registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tb_dir = Path(config.tensorboard_log_dir) / config.dataset_name / model_name
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        
        # MLflow
        self.mlflow_manager = MLflowManager(config.mlflow_experiment_name)
        
        logger.info(f"Model registry: {self.registry_dir}")
        logger.info(f"TensorBoard: {self.tb_dir}")
    
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
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage without sudo."""
        memory = {}
        if self.device.type == "mps":
            memory['allocated'] = torch.mps.current_allocated_memory() / 1e9
            memory['max_allocated'] = torch.mps.driver_allocated_memory() / 1e9
        elif self.device.type == "cuda":
            memory['allocated'] = torch.cuda.memory_allocated() / 1e9
            memory['reserved'] = torch.cuda.memory_reserved() / 1e9
            memory['max_allocated'] = torch.cuda.max_memory_allocated() / 1e9
        return memory
    
    def train_epoch(self, verbose: bool = False) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        pbar = tqdm(self.dataloaders['train'], desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            x = x.contiguous()
            y = y.contiguous()
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            loss = self.criterion(outputs, y)
            loss.backward()
            
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if verbose and batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(self.dataloaders['train'])} - Loss: {loss.item():.4f}")
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.dataloaders['train'])
    
    @torch.no_grad()
    def validate(self, verbose: bool = False) -> float:
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
        dummy_x = torch.randn(1, 1, 1578, 751, device=self.device)
        dummy_y = torch.randint(0, 3, (1, 1578, 751), device=self.device)
        
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
    
    def _log_sample_predictions(self, epoch: int):
        """Log sample predictions to TensorBoard and MLflow."""
        if self.config.log_predictions_every <= 0 or epoch % self.config.log_predictions_every != 0:
            return
        
        logger.info(f"📸 Logging sample predictions (epoch {epoch})...")
        
        self.model.eval()
        num_samples = min(4, len(self.dataloaders['val'].dataset))
        
        with torch.no_grad():
            for idx in range(num_samples):
                data, mask = self.dataloaders['val'].dataset[idx]
                x = data.unsqueeze(0).to(self.device)
                output = self.model(x)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                data_np = data.numpy()[0]
                mask_np = mask.numpy()
                shot_id = self.dataloaders['val'].dataset.get_shot_id(idx)
                
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original seismogram
                axes[0].imshow(data_np.T, cmap='seismic', aspect='auto',
                               vmin=-np.percentile(np.abs(data_np), 95),
                               vmax=np.percentile(np.abs(data_np), 95))
                axes[0].set_title(f'Seismogram (Shot {shot_id})')
                axes[0].set_xlabel('Trace')
                axes[0].set_ylabel('Sample')
                
                # Ground truth mask
                axes[1].imshow(mask_np.T, cmap='tab10', aspect='auto', vmin=0, vmax=2)
                axes[1].set_title('Ground Truth')
                axes[1].set_xlabel('Trace')
                axes[1].set_ylabel('Sample')
                
                # Prediction
                axes[2].imshow(pred.T, cmap='tab10', aspect='auto', vmin=0, vmax=2)
                axes[2].set_title('Prediction')
                axes[2].set_xlabel('Trace')
                axes[2].set_ylabel('Sample')
                
                plt.tight_layout()
                
                # Log to TensorBoard
                self.writer.add_figure(f'Seismogram/Shot_{shot_id}', fig, epoch)
                
                # Log to MLflow
                temp_path = f"temp_prediction_{shot_id}_{epoch}.png"
                fig.savefig(temp_path, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(temp_path, artifact_path=f"predictions/epoch_{epoch}")
                os.remove(temp_path)
                
                plt.close(fig)
    
    def fit(self, resume_from: Optional[str] = None, verbose: bool = False):
        """Main training loop with integrated logging."""
        
        # Setup
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        config_dict = self.config.to_dict()
        config_dict['model_name'] = self.model_name
        
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
        epoch_times = []
        
        for epoch in range(start_epoch, self.config.n_epochs):
            epoch_start = datetime.now()
            
            # Train
            train_loss = self.train_epoch(verbose=verbose)
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(verbose=verbose)
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} - Val Loss: {val_loss:.4f}")
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # === LOGGING ===
            
            # Loss (TensorBoard)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metrics/lr", current_lr, epoch)
            
            # Loss (MLflow)
            self.mlflow_manager.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr
            }, step=epoch)
            
            # Memory (if enabled)
            if self.config.log_memory:
                memory = self.get_memory_usage()
                if memory:
                    self.writer.add_scalar("Memory/allocated_GB", memory['allocated'], epoch)
                    self.mlflow_manager.log_metrics({
                        'memory_allocated_gb': memory['allocated'],
                        'memory_max_gb': memory.get('max_allocated', memory['allocated'])
                    }, step=epoch)
                    logger.info(f"📊 Memory: allocated={memory['allocated']:.2f} GB")
            
            # Training time
            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            epoch_times.append(epoch_duration)
            self.mlflow_manager.log_metrics({
                'epoch_duration_seconds': epoch_duration
            }, step=epoch)
            logger.info(f"⏱️ Epoch duration: {epoch_duration:.1f}s")
            
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
            
            # Checkpoint (save to registry)
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, train_loss, val_loss)
            
            # Sample predictions
            self._log_sample_predictions(epoch)
        
        # Finalize
        total_time = sum(epoch_times)
        self.mlflow_manager.log_metrics({
            'total_training_time_seconds': total_time,
            'average_epoch_time_seconds': total_time / len(epoch_times) if epoch_times else 0
        }, step=epoch)
        
        self.writer.close()
        self.mlflow_manager.end_run()
        
        logger.info(f"Training completed! Best val_loss: {best_val_loss:.4f}")
        logger.info(f"Model registry: {self.registry_dir}")
        logger.info(f"TensorBoard: {self.tb_dir}")
        logger.info(f"MLflow Run: {run_id}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Epochs: {len(epoch_times)}")
        logger.info(f"Best val_loss: {best_val_loss:.4f}")
        if self.config.log_memory:
            logger.info(f"Peak memory: {self.get_memory_usage().get('max_allocated', 0):.2f} GB")
        logger.info(f"Total training time: {total_time:.1f}s")
        logger.info("=" * 60)
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save checkpoint to model registry."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{self.config.dataset_name}_epoch_{epoch}_{timestamp}.pt"
        save_path = self.registry_dir / filename
        
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'model_name': self.model_name,
            'dataset_name': self.config.dataset_name,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
        self.mlflow_manager.log_artifact(str(save_path), artifact_path="checkpoints")
    
    def _save_best_model(self, best_val_loss: float):
        """Save the best model to registry."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{self.config.dataset_name}_best_{timestamp}.pt"
        save_path = self.registry_dir / filename
        
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'val_loss': best_val_loss,
            'config': self.config.to_dict(),
            'model_name': self.model_name,
            'dataset_name': self.config.dataset_name,
            'timestamp': timestamp
        }, save_path)
        logger.info(f"Best model saved: {save_path} (val_loss: {best_val_loss:.4f})")
        
        # Also save as 'best' without timestamp for easy access
        best_path = self.registry_dir / f"{self.model_name}_{self.config.dataset_name}_best.pt"
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'val_loss': best_val_loss,
            'config': self.config.to_dict(),
            'model_name': self.model_name,
            'dataset_name': self.config.dataset_name,
            'timestamp': timestamp
        }, best_path)
        self.mlflow_manager.log_artifact(str(best_path), artifact_path="checkpoints")