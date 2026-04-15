"""
PyTorch Autoencoder for Financial Transaction Anomaly Detection

Implements a deep autoencoder neural network trained on normal transactions
to detect anomalies via reconstruction error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TransactionAutoencoder(nn.Module):
    """Deep autoencoder for transaction anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2
    ):
        """
        Initialize autoencoder architecture.

        Args:
            input_dim: Number of input features
            encoding_dims: List of hidden layer dimensions for encoder
            dropout_rate: Dropout rate for regularization
        """
        super(TransactionAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dims = encoding_dims

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        decoder_dims = list(reversed(encoding_dims[:-1])) + [input_dim]

        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Sigmoid(),
            ])
            if dim != input_dim:
                decoder_layers.extend([
                    nn.BatchNorm1d(dim),
                    nn.Dropout(dropout_rate)
                ])
            prev_dim = dim

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded representation."""
        return self.encoder(x)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode from encoded representation."""
        return self.decoder(encoded)


class TransactionDataProcessor:
    """Preprocesses transaction data for autoencoder training."""

    def __init__(self):
        self.feature_columns: List[str] = []
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance: Dict[str, float] = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for model training."""

        df = df.copy()

        # Convert timestamp to features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Log transform amount features (handle zeros)
        df['log_amount'] = np.log1p(df['amount'])
        df['log_spend_last_hour'] = np.log1p(df['spend_last_hour'])
        df['log_spend_last_day'] = np.log1p(df['spend_last_day'])

        # Categorical features to encode
        categorical_cols = ['merchant_category', 'risk_profile']
        for col in categorical_cols:
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.categorical_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.categorical_encoders[col].transform(df[col])

        # Select final feature set
        self.feature_columns = [
            'log_amount', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'distance_from_home', 'is_preferred_category', 'amount_zscore',
            'transactions_last_hour', 'transactions_last_day',
            'log_spend_last_hour', 'log_spend_last_day',
            'merchant_category_encoded', 'risk_profile_encoded',
            'is_weekend'
        ]

        # Handle missing values
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)

        return df[self.feature_columns + ['is_anomaly', 'customer_id', 'transaction_id']]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessors and transform data."""
        df_features = self.prepare_features(df)
        X = df_features[self.feature_columns].values

        # Fit and transform scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessors."""
        df_features = self.prepare_features(df)
        X = df_features[self.feature_columns].values
        return self.scaler.transform(X)

    def save(self, path: str) -> None:
        """Save preprocessor state."""
        joblib.dump({
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, path)

    def load(self, path: str) -> None:
        """Load preprocessor state."""
        state = joblib.load(path)
        self.feature_columns = state['feature_columns']
        self.categorical_encoders = state['categorical_encoders']
        self.scaler = state['scaler']
        self.feature_importance = state.get('feature_importance', {})


class AnomalyDetector:
    """Complete anomaly detection system with autoencoder."""

    def __init__(
        self,
        encoding_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate

        self.processor = TransactionDataProcessor()
        self.model: Optional[TransactionAutoencoder] = None
        self.threshold: float = 0.0
        self.training_history: Dict[str, List[float]] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, np.ndarray]:
        """Prepare data for training."""

        # Use only normal transactions for training
        normal_df = df[~df['is_anomaly']].copy()
        logger.info(f"Training on {len(normal_df)} normal transactions")

        # Preprocess features
        X_normal = self.processor.fit_transform(normal_df)

        # Split training data
        X_train, X_val = train_test_split(
            X_normal, test_size=validation_split, random_state=42
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)  # Autoencoder targets = inputs
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Also return full processed data for threshold calibration
        X_full = self.processor.transform(df)

        return train_loader, val_loader, X_full

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10
    ) -> None:
        """Train the autoencoder model."""

        # Initialize model
        input_dim = next(iter(train_loader))[0].shape[1]
        self.model = TransactionAutoencoder(
            input_dim=input_dim,
            encoding_dims=self.encoding_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.training_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting training...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                reconstructed, encoded = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    reconstructed, encoded = self.model(batch_x)
                    loss = criterion(reconstructed, batch_y)
                    val_losses.append(loss.item())

            # Record losses
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter >= patience:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        logger.info("Training completed")

    def calibrate_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_fpr: float = 0.01
    ) -> Dict[str, Any]:
        """Calibrate anomaly threshold using precision-recall analysis."""

        # Get reconstruction errors
        reconstruction_errors = self.get_reconstruction_errors(X)

        # Precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y, reconstruction_errors)

        # Find threshold for target FPR
        fpr_scores = 1 - precisions
        valid_indices = fpr_scores <= target_fpr

        if not np.any(valid_indices):
            # Fallback to 95th percentile of normal transactions
            normal_errors = reconstruction_errors[y == 0]
            self.threshold = np.percentile(normal_errors, 95)
            logger.warning(f"Could not find threshold for FPR {target_fpr}, using 95th percentile: {self.threshold:.6f}")
        else:
            # Choose threshold with highest recall at acceptable FPR
            best_idx = np.where(valid_indices)[0][np.argmax(recalls[valid_indices])]
            self.threshold = thresholds[best_idx]

        # Calculate metrics at chosen threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

        metrics = {
            'threshold': self.threshold,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'auc_roc': roc_auc_score(y, reconstruction_errors),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }

        logger.info(f"Threshold calibrated: {self.threshold:.6f}")
        logger.info(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        logger.info(f"FPR: {metrics['fpr']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}")

        return metrics

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for input data."""
        self.model.eval()

        errors = []
        with torch.no_grad():
            # Process in batches
            batch_size = 1000
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                reconstructed, _ = self.model(batch)
                error = torch.mean((batch - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())

        return np.array(errors)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores and binary predictions."""
        scores = self.get_reconstruction_errors(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores

    def save(self, model_dir: str) -> None:
        """Save complete model state."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': self.model.input_dim,
                    'encoding_dims': self.encoding_dims,
                    'dropout_rate': self.dropout_rate
                },
                'threshold': self.threshold,
                'training_history': self.training_history
            }, f"{model_dir}/autoencoder.pth")

        # Save preprocessor
        self.processor.save(f"{model_dir}/preprocessor.pkl")

        logger.info(f"Model saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        """Load complete model state."""
        # Load preprocessor
        self.processor.load(f"{model_dir}/preprocessor.pkl")

        # Load model
        checkpoint = torch.load(f"{model_dir}/autoencoder.pth", map_location=self.device)

        config = checkpoint['model_config']
        self.model = TransactionAutoencoder(
            input_dim=config['input_dim'],
            encoding_dims=config['encoding_dims'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.training_history = checkpoint.get('training_history', {})

        logger.info(f"Model loaded from {model_dir}")

    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training and validation loss curves."""
        if not self.training_history:
            logger.warning("No training history available")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Train and evaluate the anomaly detection model."""

    logging.basicConfig(level=logging.INFO)

    # Load data
    data_path = "data/generated/transactions.parquet"
    if not Path(data_path).exists():
        logger.error(f"Dataset not found at {data_path}. Run synthetic_generator.py first.")
        return

    logger.info("Loading transaction data...")
    df = pd.read_parquet(data_path)

    # Initialize detector
    detector = AnomalyDetector(
        encoding_dims=[64, 32, 16],
        dropout_rate=0.2
    )

    # Prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, X_full = detector.prepare_data(df, validation_split=0.2)

    # Train model
    logger.info("Training autoencoder...")
    detector.train(train_loader, val_loader, epochs=100, learning_rate=0.001)

    # Calibrate threshold
    logger.info("Calibrating threshold...")
    y_full = df['is_anomaly'].astype(int).values
    metrics = detector.calibrate_threshold(X_full, y_full, target_fpr=0.01)

    # Save model
    model_dir = "model/artifacts"
    detector.save(model_dir)

    # Save evaluation metrics
    with open(f"{model_dir}/evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot training history
    detector.plot_training_history(f"{model_dir}/training_history.png")

    logger.info("Model training and evaluation complete!")


if __name__ == "__main__":
    main()