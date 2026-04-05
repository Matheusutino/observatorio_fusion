"""
Trainer específico para AutoencoderFusionInside com dual loss.

Similar ao AutoencoderTrainer, mas trabalha com dois inputs e duas reconstruções.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import compute_metrics
from ..config.config import DEVICE


class AutoencoderFusionInsideTrainer:
    """
    Trainer para AutoencoderFusionInside com dual loss: reconstruction + classification.

    Recebe duas entradas separadas (x1, x2) e reconstrói ambas.
    Monitora F1-macro no conjunto de validação e restaura o melhor modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion_class: nn.Module,
        criterion_recon: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_class: float = 1.0,
        lambda_recon: float = 1.0,
        device: torch.device = DEVICE,
    ):
        """
        Parâmetros
        ----------
        model : nn.Module
            Modelo AutoencoderFusionInside a ser treinado.
        criterion_class : nn.Module
            Função de perda de classificação (ex: BCEWithLogitsLoss).
        criterion_recon : nn.Module
            Função de perda de reconstrução (ex: MSELoss).
        optimizer : torch.optim.Optimizer
            Otimizador.
        lambda_class : float
            Peso da loss de classificação.
        lambda_recon : float
            Peso da loss de reconstrução.
            loss_total = lambda_class * loss_class + lambda_recon * loss_recon
        device : torch.device
            Device (CPU/GPU).
        """
        self.model = model
        self.criterion_class = criterion_class
        self.criterion_recon = criterion_recon
        self.optimizer = optimizer
        self.lambda_class = lambda_class
        self.lambda_recon = lambda_recon
        self.device = device

    def train_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        """
        Executa uma época de treino.

        Parâmetros
        ----------
        loader : DataLoader
            DataLoader de treino (retorna (X1, X2, y)).

        Retorna
        -------
        tuple[float, float, float]
            (loss_total, loss_class, loss_recon)
        """
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0

        for X1_batch, X2_batch, y_batch in loader:
            X1_batch = X1_batch.to(self.device)
            X2_batch = X2_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            self.optimizer.zero_grad()

            # Forward pass
            logits, recon1, recon2 = self.model(X1_batch, X2_batch)

            # Compute losses
            loss_class = self.criterion_class(logits, y_batch)
            loss_recon1 = self.criterion_recon(recon1, X1_batch)
            loss_recon2 = self.criterion_recon(recon2, X2_batch)
            loss_recon = loss_recon1 + loss_recon2
            loss = self.lambda_class * loss_class + self.lambda_recon * loss_recon

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_recon_loss += loss_recon.item()

        n_batches = len(loader)
        return (
            total_loss / n_batches,
            total_class_loss / n_batches,
            total_recon_loss / n_batches,
        )

    @torch.no_grad()
    def eval_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        Executa uma época de avaliação.

        Parâmetros
        ----------
        loader : DataLoader
            DataLoader de validação/teste (retorna (X1, X2, y)).

        Retorna
        -------
        tuple[float, float, float, float, np.ndarray, np.ndarray]
            (loss_total, loss_class, loss_recon, f1_macro, y_true, y_pred)
        """
        self.model.eval()
        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        all_true, all_pred = [], []

        for X1_batch, X2_batch, y_batch in loader:
            X1_batch = X1_batch.to(self.device)
            X2_batch = X2_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            # Forward pass
            logits, recon1, recon2 = self.model(X1_batch, X2_batch)

            # Compute losses
            loss_class = self.criterion_class(logits, y_batch)
            loss_recon1 = self.criterion_recon(recon1, X1_batch)
            loss_recon2 = self.criterion_recon(recon2, X2_batch)
            loss_recon = loss_recon1 + loss_recon2
            loss = self.lambda_class * loss_class + self.lambda_recon * loss_recon

            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_recon_loss += loss_recon.item()

            # Predictions
            preds = (logits > 0).long().cpu().numpy()
            all_true.extend(y_batch.long().cpu().numpy())
            all_pred.extend(preds)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        metrics = compute_metrics(y_true, y_pred)

        n_batches = len(loader)
        return (
            total_loss / n_batches,
            total_class_loss / n_batches,
            total_recon_loss / n_batches,
            metrics["f1_macro"],
            y_true,
            y_pred,
        )

    def train(
        self,
        loader_train: DataLoader,
        loader_val: DataLoader,
        epochs: int,
        patience: int,
        verbose: bool = True,
    ) -> dict:
        """
        Loop de treino completo com early stopping no F1-macro de validação.

        Parâmetros
        ----------
        loader_train : DataLoader
            DataLoader de treino.
        loader_val : DataLoader
            DataLoader de validação.
        epochs : int
            Número máximo de épocas.
        patience : int
            Épocas sem melhora antes de parar.
        verbose : bool
            Se True, imprime progresso.

        Retorna
        -------
        dict
            Histórico: {"train_loss", "train_class_loss", "train_recon_loss",
                        "val_loss", "val_class_loss", "val_recon_loss", "val_f1"}
        """
        history = {
            "train_loss": [],
            "train_class_loss": [],
            "train_recon_loss": [],
            "val_loss": [],
            "val_class_loss": [],
            "val_recon_loss": [],
            "val_f1": [],
        }
        best_val_f1 = -1.0
        best_state = None
        epochs_no_impr = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_class, tr_recon = self.train_epoch(loader_train)
            val_loss, val_class, val_recon, val_f1, _, _ = self.eval_epoch(loader_val)

            history["train_loss"].append(tr_loss)
            history["train_class_loss"].append(tr_class)
            history["train_recon_loss"].append(tr_recon)
            history["val_loss"].append(val_loss)
            history["val_class_loss"].append(val_class)
            history["val_recon_loss"].append(val_recon)
            history["val_f1"].append(val_f1)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={tr_loss:.4f} (class={tr_class:.4f}, recon={tr_recon:.4f}) | "
                    f"val_loss={val_loss:.4f} (class={val_class:.4f}, recon={val_recon:.4f}) val_f1={val_f1:.4f}"
                )

            # Early stopping — monitora F1-macro val
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_impr = 0
            else:
                epochs_no_impr += 1
                if epochs_no_impr >= patience:
                    if verbose:
                        print(
                            f"\n  Early stop at epoch {epoch} "
                            f"(patience={patience}). "
                            f"Best val F1: {best_val_f1:.4f}"
                        )
                    break

        # Restaura melhor modelo
        self.model.load_state_dict(best_state)
        if verbose:
            print(f"  Best model restored (val F1={best_val_f1:.4f})")

        return history
