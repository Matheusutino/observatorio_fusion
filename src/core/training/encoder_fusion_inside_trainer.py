"""
Trainer específico para EncoderFusionInside (apenas classificação).

Similar ao EncoderTrainer, mas trabalha com dois inputs separados.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import compute_metrics
from ..config.config import DEVICE


class EncoderFusionInsideTrainer:
    """
    Trainer para EncoderFusionInside com classificação apenas.

    Recebe duas entradas separadas (x1, x2) e realiza classificação.
    Monitora F1-macro no conjunto de validação e restaura o melhor modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = DEVICE,
    ):
        """
        Parâmetros
        ----------
        model : nn.Module
            Modelo EncoderFusionInside a ser treinado.
        criterion : nn.Module
            Função de perda de classificação (ex: BCEWithLogitsLoss).
        optimizer : torch.optim.Optimizer
            Otimizador.
        device : torch.device
            Device (CPU/GPU).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Executa uma época de treino.

        Parâmetros
        ----------
        loader : DataLoader
            DataLoader de treino (retorna (X1, X2, y)).

        Retorna
        -------
        float
            Loss média da época.
        """
        self.model.train()
        total_loss = 0.0

        for X1_batch, X2_batch, y_batch in loader:
            X1_batch = X1_batch.to(self.device)
            X2_batch = X2_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(X1_batch, X2_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Executa uma época de avaliação.

        Parâmetros
        ----------
        loader : DataLoader
            DataLoader de validação/teste (retorna (X1, X2, y)).

        Retorna
        -------
        tuple[float, float, np.ndarray, np.ndarray]
            (loss, f1_macro, y_true, y_pred)
        """
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []

        for X1_batch, X2_batch, y_batch in loader:
            X1_batch = X1_batch.to(self.device)
            X2_batch = X2_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            # Forward pass
            logits = self.model(X1_batch, X2_batch)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item()

            # Predictions
            preds = (logits > 0).long().cpu().numpy()
            all_true.extend(y_batch.long().cpu().numpy())
            all_pred.extend(preds)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        metrics = compute_metrics(y_true, y_pred)

        return (
            total_loss / len(loader),
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
            Histórico: {"train_loss", "val_loss", "val_f1"}
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }
        best_val_f1 = -1.0
        best_state = None
        epochs_no_impr = 0

        for epoch in range(1, epochs + 1):
            tr_loss = self.train_epoch(loader_train)
            val_loss, val_f1, _, _ = self.eval_epoch(loader_val)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={tr_loss:.4f} | "
                    f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
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
