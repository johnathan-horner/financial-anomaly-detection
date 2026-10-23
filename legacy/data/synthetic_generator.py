"""
Synthetic Financial Transaction Dataset Generator

Generates realistic financial transaction data with injected anomalies
for training and testing the anomaly detection system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
from faker import Faker
import json
import logging

logger = logging.getLogger(__name__)
fake = Faker()


class TransactionGenerator:
    """Generates synthetic financial transaction data."""

    MERCHANT_CATEGORIES = [
        'grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy',
        'entertainment', 'travel', 'online', 'utility', 'insurance',
        'healthcare', 'education', 'automotive', 'home_improvement'
    ]

    RISK_PROFILES = {
        'low_risk': {'amount_mean': 45, 'amount_std': 25, 'frequency': 0.8},
        'medium_risk': {'amount_mean': 120, 'amount_std': 60, 'frequency': 0.6},
        'high_risk': {'amount_mean': 800, 'amount_std': 400, 'frequency': 0.3}
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        np.random.seed(seed)
        random.seed(seed)
        Faker.seed(seed)

    def generate_customer_profiles(self, num_customers: int = 1000) -> List[Dict]:
        """Generate customer profiles with behavioral patterns."""
        profiles = []

        for customer_id in range(num_customers):
            # Assign risk profile
            risk_profile = np.random.choice(
                list(self.RISK_PROFILES.keys()),
                p=[0.7, 0.25, 0.05]  # Most customers are low risk
            )

            profile = {
                'customer_id': f"CUST_{customer_id:06d}",
                'risk_profile': risk_profile,
                'home_location': {
                    'lat': fake.latitude(),
                    'lon': fake.longitude(),
                    'city': fake.city(),
                    'state': fake.state_abbr()
                },
                'preferred_categories': np.random.choice(
                    self.MERCHANT_CATEGORIES,
                    size=random.randint(3, 6),
                    replace=False
                ).tolist(),
                'typical_amount_range': self.RISK_PROFILES[risk_profile],
                'transaction_frequency': self.RISK_PROFILES[risk_profile]['frequency'],
                'active_hours': {
                    'start': random.randint(6, 10),
                    'end': random.randint(20, 23)
                }
            }

            profiles.append(profile)

        return profiles

    def generate_normal_transactions(
        self,
        customer_profiles: List[Dict],
        num_transactions: int = 100000,
        days_range: int = 90
    ) -> pd.DataFrame:
        """Generate normal transaction patterns."""

        transactions = []
        start_date = datetime.now() - timedelta(days=days_range)

        for _ in range(num_transactions):
            # Select random customer
            customer = random.choice(customer_profiles)

            # Generate timestamp within active hours
            random_day = random.randint(0, days_range)
            hour = random.randint(
                customer['active_hours']['start'],
                customer['active_hours']['end']
            )
            timestamp = start_date + timedelta(
                days=random_day,
                hours=hour,
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            # Generate amount based on risk profile
            amount_params = customer['typical_amount_range']
            amount = max(1.0, np.random.normal(
                amount_params['amount_mean'],
                amount_params['amount_std']
            ))

            # Select merchant category from preferences
            category = random.choice(customer['preferred_categories'])

            # Generate location near home (within 50 miles)
            home_lat = float(customer['home_location']['lat'])
            home_lon = float(customer['home_location']['lon'])

            # Small random offset for location
            lat_offset = np.random.normal(0, 0.1)  # ~7 miles std dev
            lon_offset = np.random.normal(0, 0.1)

            transaction = {
                'transaction_id': f"TXN_{len(transactions):08d}",
                'customer_id': customer['customer_id'],
                'timestamp': timestamp.isoformat(),
                'amount': round(amount, 2),
                'merchant_category': category,
                'merchant_id': f"MERCH_{category}_{random.randint(1, 1000):04d}",
                'location_lat': home_lat + lat_offset,
                'location_lon': home_lon + lon_offset,
                'is_anomaly': False,
                'anomaly_type': None
            }

            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Add engineered features
        df = self._add_engineered_features(df, customer_profiles)

        return df.sort_values('timestamp').reset_index(drop=True)

    def inject_anomalies(
        self,
        normal_df: pd.DataFrame,
        customer_profiles: List[Dict],
        num_anomalies: int = 2000
    ) -> pd.DataFrame:
        """Inject various types of anomalies into the normal transaction data."""

        anomalies = []

        anomaly_types = [
            ('unusual_amount', 0.3),
            ('geographic_impossible', 0.25),
            ('velocity_spike', 0.2),
            ('unusual_time', 0.15),
            ('new_merchant_category', 0.1)
        ]

        for anomaly_type, proportion in anomaly_types:
            count = int(num_anomalies * proportion)

            for _ in range(count):
                # Base transaction from normal data
                base_txn = normal_df.sample(1).iloc[0].copy()
                customer = next(
                    c for c in customer_profiles
                    if c['customer_id'] == base_txn['customer_id']
                )

                # Modify based on anomaly type
                if anomaly_type == 'unusual_amount':
                    # 10x typical amount
                    base_txn['amount'] = base_txn['amount'] * random.uniform(8, 15)

                elif anomaly_type == 'geographic_impossible':
                    # Transaction far from home
                    base_txn['location_lat'] = fake.latitude()
                    base_txn['location_lon'] = fake.longitude()

                elif anomaly_type == 'velocity_spike':
                    # Multiple transactions in short timeframe
                    for i in range(random.randint(3, 6)):
                        velocity_txn = base_txn.copy()
                        velocity_txn['transaction_id'] = f"ANOM_{len(anomalies):06d}_{i}"
                        velocity_txn['timestamp'] = (
                            pd.to_datetime(base_txn['timestamp']) +
                            timedelta(minutes=random.randint(1, 10))
                        ).isoformat()
                        velocity_txn['amount'] = random.uniform(50, 500)
                        velocity_txn['is_anomaly'] = True
                        velocity_txn['anomaly_type'] = anomaly_type
                        anomalies.append(velocity_txn)
                    continue

                elif anomaly_type == 'unusual_time':
                    # Transaction at 3 AM
                    dt = pd.to_datetime(base_txn['timestamp'])
                    dt = dt.replace(hour=random.randint(1, 5))
                    base_txn['timestamp'] = dt.isoformat()

                elif anomaly_type == 'new_merchant_category':
                    # Category never seen before for this customer
                    new_categories = set(self.MERCHANT_CATEGORIES) - set(customer['preferred_categories'])
                    if new_categories:
                        base_txn['merchant_category'] = random.choice(list(new_categories))

                base_txn['transaction_id'] = f"ANOM_{len(anomalies):06d}"
                base_txn['is_anomaly'] = True
                base_txn['anomaly_type'] = anomaly_type
                anomalies.append(base_txn)

        anomaly_df = pd.DataFrame(anomalies)

        # Combine and re-engineer features
        combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        combined_df = self._add_engineered_features(combined_df, customer_profiles)

        return combined_df.sort_values('timestamp').reset_index(drop=True)

    def _add_engineered_features(
        self,
        df: pd.DataFrame,
        customer_profiles: List[Dict]
    ) -> pd.DataFrame:
        """Add engineered features for anomaly detection."""

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # Customer-specific features
        customer_map = {c['customer_id']: c for c in customer_profiles}

        def calculate_customer_features(row):
            customer = customer_map[row['customer_id']]

            # Distance from home
            home_lat = float(customer['home_location']['lat'])
            home_lon = float(customer['home_location']['lon'])

            # Simple distance calculation (not geodesic, but sufficient for demo)
            lat_diff = abs(row['location_lat'] - home_lat)
            lon_diff = abs(row['location_lon'] - home_lon)
            distance_from_home = np.sqrt(lat_diff**2 + lon_diff**2) * 69  # Rough miles

            return pd.Series({
                'distance_from_home': distance_from_home,
                'is_preferred_category': row['merchant_category'] in customer['preferred_categories'],
                'risk_profile': customer['risk_profile']
            })

        customer_features = df.apply(calculate_customer_features, axis=1)
        df = pd.concat([df, customer_features], axis=1)

        # Rolling statistics (90-day window)
        df = df.sort_values(['customer_id', 'timestamp'])

        # Amount-based features
        df['amount_zscore'] = df.groupby('customer_id')['amount'].transform(
            lambda x: (x - x.rolling(30, min_periods=1).mean()) /
                     (x.rolling(30, min_periods=1).std() + 1e-6)
        )

        # Frequency features
        df['transactions_last_hour'] = df.groupby('customer_id')['timestamp'].transform(
            lambda x: x.rolling('1H').count() - 1
        )

        df['transactions_last_day'] = df.groupby('customer_id')['timestamp'].transform(
            lambda x: x.rolling('1D').count() - 1
        )

        # Velocity features (spending in time windows)
        df['spend_last_hour'] = df.groupby('customer_id')['amount'].transform(
            lambda x: x.rolling('1H').sum() - x
        )

        df['spend_last_day'] = df.groupby('customer_id')['amount'].transform(
            lambda x: x.rolling('1D').sum() - x
        )

        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        customer_profiles: List[Dict],
        output_dir: str = "data/generated"
    ) -> None:
        """Save generated dataset and metadata."""

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save main dataset
        df.to_csv(f"{output_dir}/transactions.csv", index=False)
        df.to_parquet(f"{output_dir}/transactions.parquet", index=False)

        # Save customer profiles
        with open(f"{output_dir}/customer_profiles.json", 'w') as f:
            json.dump(customer_profiles, f, indent=2, default=str)

        # Save dataset statistics
        stats = {
            'total_transactions': len(df),
            'normal_transactions': len(df[~df['is_anomaly']]),
            'anomalous_transactions': len(df[df['is_anomaly']]),
            'anomaly_rate': df['is_anomaly'].mean(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'anomaly_types': df[df['is_anomaly']]['anomaly_type'].value_counts().to_dict(),
            'feature_columns': df.columns.tolist()
        }

        with open(f"{output_dir}/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"Total transactions: {stats['total_transactions']}")
        logger.info(f"Anomaly rate: {stats['anomaly_rate']:.3%}")


def main():
    """Generate the complete synthetic dataset."""

    logging.basicConfig(level=logging.INFO)

    generator = TransactionGenerator(seed=42)

    # Generate customer profiles
    logger.info("Generating customer profiles...")
    customers = generator.generate_customer_profiles(num_customers=1000)

    # Generate normal transactions
    logger.info("Generating normal transactions...")
    normal_df = generator.generate_normal_transactions(
        customers,
        num_transactions=100000,
        days_range=90
    )

    # Inject anomalies
    logger.info("Injecting anomalies...")
    final_df = generator.inject_anomalies(
        normal_df,
        customers,
        num_anomalies=2000
    )

    # Save dataset
    logger.info("Saving dataset...")
    generator.save_dataset(final_df, customers)

    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()