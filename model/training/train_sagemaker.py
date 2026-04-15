"""
SageMaker Training Script for Financial Transaction Autoencoder

This script handles training the autoencoder model in SageMaker environment
with proper data loading, model checkpointing, and metric logging.
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path

import torch
import boto3
import pandas as pd
from sagemaker.session import Session

# Add parent directory to path for imports
sys.path.append('/opt/ml/code')

from autoencoder import AnomalyDetector

logger = logging.getLogger(__name__)


class SageMakerTrainer:
    """Handles SageMaker-specific training workflow."""

    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Configure logging for SageMaker environment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/opt/ml/output/training.log')
            ]
        )

    def setup_directories(self):
        """Setup SageMaker directory structure."""
        self.data_dir = Path('/opt/ml/input/data')
        self.model_dir = Path('/opt/ml/model')
        self.output_dir = Path('/opt/ml/output')
        self.checkpoints_dir = Path('/opt/ml/checkpoints')

        # Create directories
        for dir_path in [self.model_dir, self.output_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load training data from SageMaker input channels."""

        # SageMaker channels
        training_channel = self.data_dir / 'training'

        if not training_channel.exists():
            raise ValueError(f"Training data channel not found at {training_channel}")

        # Look for parquet or CSV files
        data_files = list(training_channel.glob('*.parquet'))
        if not data_files:
            data_files = list(training_channel.glob('*.csv'))

        if not data_files:
            raise ValueError("No training data files found")

        logger.info(f"Loading data from {data_files[0]}")

        if data_files[0].suffix == '.parquet':
            df = pd.read_parquet(data_files[0])
        else:
            df = pd.read_csv(data_files[0])

        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Anomaly rate: {df['is_anomaly'].mean():.3%}")

        return df

    def train(self):
        """Execute the training workflow."""

        logger.info("Starting SageMaker training job")
        logger.info(f"Arguments: {self.args}")

        # Load data
        df = self.load_data()

        # Initialize detector with hyperparameters
        detector = AnomalyDetector(
            encoding_dims=self.args.encoding_dims,
            dropout_rate=self.args.dropout_rate,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info(f"Using device: {detector.device}")

        # Prepare data
        train_loader, val_loader, X_full = detector.prepare_data(
            df, validation_split=self.args.validation_split
        )

        # Train model
        detector.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.args.epochs,
            learning_rate=self.args.learning_rate,
            patience=self.args.patience
        )

        # Calibrate threshold
        y_full = df['is_anomaly'].astype(int).values
        metrics = detector.calibrate_threshold(
            X_full, y_full, target_fpr=self.args.target_fpr
        )

        # Save model artifacts
        detector.save(str(self.model_dir))

        # Save training metrics
        training_results = {
            'training_args': vars(self.args),
            'model_config': {
                'encoding_dims': detector.encoding_dims,
                'dropout_rate': detector.dropout_rate,
                'input_dim': detector.model.input_dim if detector.model else None
            },
            'evaluation_metrics': metrics,
            'training_history': detector.training_history,
            'dataset_stats': {
                'total_samples': len(df),
                'normal_samples': len(df[~df['is_anomaly']]),
                'anomaly_samples': len(df[df['is_anomaly']]),
                'anomaly_rate': df['is_anomaly'].mean()
            }
        }

        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)

        # Generate training plots
        if detector.training_history:
            detector.plot_training_history(
                str(self.output_dir / 'training_history.png')
            )

        # Log final metrics to CloudWatch (via SageMaker)
        self.log_metrics(metrics)

        logger.info("Training completed successfully")
        logger.info(f"Final metrics: Precision={metrics['precision']:.3f}, "
                   f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    def log_metrics(self, metrics):
        """Log metrics for SageMaker experiment tracking."""

        # These will be automatically captured by SageMaker
        metric_definitions = [
            ('precision', metrics['precision']),
            ('recall', metrics['recall']),
            ('f1_score', metrics['f1_score']),
            ('fpr', metrics['fpr']),
            ('auc_roc', metrics['auc_roc']),
            ('threshold', metrics['threshold'])
        ]

        for metric_name, value in metric_definitions:
            print(f"{metric_name}: {value}")


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='Train financial transaction autoencoder')

    # Model hyperparameters
    parser.add_argument('--encoding_dims', type=int, nargs='+', default=[64, 32, 16],
                       help='Hidden layer dimensions for encoder')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate for regularization')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation data split ratio')

    # Threshold calibration
    parser.add_argument('--target_fpr', type=float, default=0.01,
                       help='Target false positive rate for threshold calibration')

    # SageMaker parameters
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data'))

    return parser.parse_args()


def main():
    """Main training entry point for SageMaker."""

    args = parse_args()

    try:
        trainer = SageMakerTrainer(args)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()