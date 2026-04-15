"""
Unit Tests for Autoencoder Model

Tests for transaction preprocessing, model training,
anomaly scoring, and threshold calibration.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch

from model.autoencoder import (
    TransactionAutoencoder,
    TransactionDataProcessor,
    AnomalyDetector
)


class TestTransactionAutoencoder:
    """Test the autoencoder neural network."""

    def test_model_initialization(self):
        """Test model creates with correct architecture."""
        model = TransactionAutoencoder(
            input_dim=15,
            encoding_dims=[64, 32, 16],
            dropout_rate=0.2
        )

        assert model.input_dim == 15
        assert model.encoding_dims == [64, 32, 16]

        # Test forward pass
        x = torch.randn(10, 15)
        reconstructed, encoded = model(x)

        assert reconstructed.shape == (10, 15)
        assert encoded.shape == (10, 16)  # Last encoding dim

    def test_model_encode_decode(self):
        """Test separate encode and decode operations."""
        model = TransactionAutoencoder(input_dim=15, encoding_dims=[32, 16])

        x = torch.randn(5, 15)
        encoded = model.encode(x)
        decoded = model.decode(encoded)

        assert encoded.shape == (5, 16)
        assert decoded.shape == (5, 15)

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        np.random.seed(42)

        data = {
            'transaction_id': [f'TXN_{i:06d}' for i in range(1000)],
            'customer_id': [f'CUST_{i%100:04d}' for i in range(1000)],
            'amount': np.random.lognormal(mean=3.0, sigma=1.0, size=1000),
            'merchant_category': np.random.choice(
                ['grocery', 'gas_station', 'restaurant', 'retail'],
                size=1000
            ),
            'timestamp': pd.date_range(
                start='2024-01-01',
                periods=1000,
                freq='1H'
            ),
            'is_anomaly': np.random.choice([True, False], size=1000, p=[0.02, 0.98])
        }

        df = pd.DataFrame(data)

        # Add engineered features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['distance_from_home'] = np.random.exponential(scale=5.0, size=1000)
        df['is_preferred_category'] = np.random.choice([True, False], size=1000)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        df['transactions_last_hour'] = np.random.poisson(lam=0.5, size=1000)
        df['transactions_last_day'] = np.random.poisson(lam=5.0, size=1000)
        df['spend_last_hour'] = np.random.exponential(scale=50.0, size=1000)
        df['spend_last_day'] = np.random.exponential(scale=200.0, size=1000)
        df['risk_profile'] = np.random.choice(
            ['low_risk', 'medium_risk', 'high_risk'],
            size=1000,
            p=[0.7, 0.25, 0.05]
        )

        return df


class TestTransactionDataProcessor:
    """Test data preprocessing pipeline."""

    def test_feature_preparation(self, sample_transaction_data):
        """Test feature engineering and preparation."""
        processor = TransactionDataProcessor()
        processed_df = processor.prepare_features(sample_transaction_data)

        # Check required columns exist
        required_features = [
            'log_amount', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'distance_from_home', 'is_preferred_category', 'amount_zscore',
            'transactions_last_hour', 'transactions_last_day',
            'log_spend_last_hour', 'log_spend_last_day',
            'merchant_category_encoded', 'risk_profile_encoded',
            'is_weekend'
        ]

        for feature in required_features:
            assert feature in processed_df.columns

        # Check no missing values in features
        feature_data = processed_df[processor.feature_columns]
        assert not feature_data.isnull().any().any()

    def test_fit_transform(self, sample_transaction_data):
        """Test fitting and transforming data."""
        processor = TransactionDataProcessor()
        X_scaled = processor.fit_transform(sample_transaction_data)

        # Check output shape
        assert X_scaled.shape[0] == len(sample_transaction_data)
        assert X_scaled.shape[1] == len(processor.feature_columns)

        # Check scaling (should have roughly zero mean, unit variance)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1

    def test_transform_only(self, sample_transaction_data):
        """Test transform without fitting."""
        processor = TransactionDataProcessor()

        # First fit
        processor.fit_transform(sample_transaction_data[:800])

        # Then transform new data
        X_new = processor.transform(sample_transaction_data[800:])

        assert X_new.shape[0] == 200
        assert X_new.shape[1] == len(processor.feature_columns)

    def test_save_load_processor(self, sample_transaction_data, tmp_path):
        """Test saving and loading processor state."""
        processor = TransactionDataProcessor()
        processor.fit_transform(sample_transaction_data)

        # Save processor
        save_path = tmp_path / "processor.pkl"
        processor.save(str(save_path))

        # Load new processor
        new_processor = TransactionDataProcessor()
        new_processor.load(str(save_path))

        # Test they produce same output
        X1 = processor.transform(sample_transaction_data[:100])
        X2 = new_processor.transform(sample_transaction_data[:100])

        np.testing.assert_array_almost_equal(X1, X2)


class TestAnomalyDetector:
    """Test complete anomaly detection system."""

    @pytest.fixture
    def detector(self):
        """Create anomaly detector instance."""
        return AnomalyDetector(
            encoding_dims=[32, 16, 8],
            dropout_rate=0.1
        )

    def test_data_preparation(self, detector, sample_transaction_data):
        """Test data preparation for training."""
        train_loader, val_loader, X_full = detector.prepare_data(
            sample_transaction_data,
            validation_split=0.2
        )

        # Check data loaders
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(train_loader.dataset) > len(val_loader.dataset)

        # Check full dataset
        assert X_full.shape[0] == len(sample_transaction_data)

        # Test batch from loader
        for batch_x, batch_y in train_loader:
            assert batch_x.shape == batch_y.shape
            assert len(batch_x.shape) == 2
            break

    @patch('model.autoencoder.torch.save')
    @patch('model.autoencoder.joblib.dump')
    def test_training_workflow(
        self,
        mock_joblib_dump,
        mock_torch_save,
        detector,
        sample_transaction_data,
        tmp_path
    ):
        """Test training workflow (mocked to avoid long training)."""
        # Prepare data
        train_loader, val_loader, X_full = detector.prepare_data(
            sample_transaction_data,
            validation_split=0.2
        )

        # Mock training (normally would call detector.train)
        detector.model = TransactionAutoencoder(
            input_dim=len(detector.processor.feature_columns),
            encoding_dims=detector.encoding_dims
        )
        detector.threshold = 0.5
        detector.training_history = {'train_loss': [1.0, 0.8], 'val_loss': [1.1, 0.9]}

        # Test saving
        model_dir = str(tmp_path / "model")
        detector.save(model_dir)

        # Verify save was called
        mock_torch_save.assert_called_once()
        mock_joblib_dump.assert_called_once()

    def test_reconstruction_errors(self, detector, sample_transaction_data):
        """Test reconstruction error calculation."""
        # Prepare minimal model
        detector.processor.fit_transform(sample_transaction_data)

        detector.model = TransactionAutoencoder(
            input_dim=len(detector.processor.feature_columns),
            encoding_dims=[16, 8]
        )

        # Get some data
        X = detector.processor.transform(sample_transaction_data[:10])

        # Calculate errors
        errors = detector.get_reconstruction_errors(X)

        assert len(errors) == 10
        assert all(error >= 0 for error in errors)

    def test_threshold_calibration(self, detector, sample_transaction_data):
        """Test threshold calibration logic."""
        # Create mock data with known labels
        X = np.random.randn(100, 15)
        y = np.random.choice([0, 1], size=100, p=[0.95, 0.05])

        # Mock model
        detector.model = Mock()
        detector.model.eval.return_value = None

        # Mock reconstruction errors
        with patch.object(detector, 'get_reconstruction_errors') as mock_errors:
            # Errors higher for anomalies
            mock_errors.return_value = np.where(y == 1, 0.8, 0.2) + np.random.normal(0, 0.1, 100)

            metrics = detector.calibrate_threshold(X, y, target_fpr=0.05)

            assert 'threshold' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert metrics['threshold'] > 0

    def test_prediction(self, detector, sample_transaction_data):
        """Test anomaly prediction."""
        # Minimal setup
        detector.processor.fit_transform(sample_transaction_data)
        detector.model = TransactionAutoencoder(
            input_dim=len(detector.processor.feature_columns),
            encoding_dims=[16, 8]
        )
        detector.threshold = 0.5

        # Test prediction
        X = detector.processor.transform(sample_transaction_data[:10])
        predictions, scores = detector.predict(X)

        assert len(predictions) == 10
        assert len(scores) == 10
        assert all(pred in [0, 1] for pred in predictions)
        assert all(score >= 0 for score in scores)


if __name__ == '__main__':
    pytest.main([__file__])