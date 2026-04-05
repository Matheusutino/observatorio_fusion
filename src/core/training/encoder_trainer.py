"""
Classe EncoderTrainer genérica para treino com early stopping.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import compute_metrics
from ..config.config import DEVICE


class EncoderTrainer:
    """
    EncoderTrainer genérico para modelos de classificação binária com early stopping.

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
            Modelo a ser treinado.
        criterion : nn.Module
            Função de perda (ex: BCEWithLogitsLoss).
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
            DataLoader de treino.

        Retorna
        -------
        float
            Loss média da época.
        """
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            self.optimizer.zero_grad()
            logit = self.model(X_batch)
            loss = self.criterion(logit, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Executa uma época de avaliação.

        Parâmetros
        ----------
        loader : DataLoader
            DataLoader de validação/teste.

        Retorna
        -------
        tuple[float, float, np.ndarray, np.ndarray]
            (loss, f1_macro, y_true, y_pred)
        """
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()

            logit = self.model(X_batch)
            loss = self.criterion(logit, y_batch)
            total_loss += loss.item()

            preds = (logit > 0).long().cpu().numpy()
            all_true.extend(y_batch.long().cpu().numpy())
            all_pred.extend(preds)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        metrics = compute_metrics(y_true, y_pred)

        return total_loss / len(loader), metrics["f1_macro"], y_true, y_pred

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
            Histórico de treino: {"train_loss", "val_loss", "val_f1"}
        """
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
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
                            f"\n  Early stop na época {epoch} "
                            f"(patience={patience}). "
                            f"Melhor val F1: {best_val_f1:.4f}"
                        )
                    break

        # Restaura melhor modelo
        self.model.load_state_dict(best_state)
        if verbose:
            print(f"  Melhor modelo restaurado (val F1={best_val_f1:.4f})")

        return history
