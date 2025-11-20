"""
Data loading and preprocessing module for movie plots dataset.
"""
import os
import pandas as pd
from typing import Optional
import kagglehub


class DataLoader:
    """Handles loading and preprocessing of the Wikipedia Movie Plots dataset."""

    def __init__(self, max_rows: int = 500):
        """
        Initialize the data loader.

        Args:
            max_rows: Maximum number of rows to load from the dataset
        """
        self.max_rows = max_rows
        self.data_path = None

    def download_dataset(self) -> str:
        """
        Download the Wikipedia Movie Plots dataset using kagglehub.

        Returns:
            Path to the downloaded dataset
        """
        print("Downloading Wikipedia Movie Plots dataset...")
        path = kagglehub.dataset_download("jrobischon/wikipedia-movie-plots")
        print(f"Dataset downloaded to: {path}")
        self.data_path = path
        return path

    def load_data(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the movie plots dataset.

        Args:
            dataset_path: Path to the dataset. If None, will download it.

        Returns:
            DataFrame with Title and Plot columns
        """
        if dataset_path is None:
            dataset_path = self.download_dataset()
        else:
            self.data_path = dataset_path

        # Find the CSV file in the dataset directory
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {dataset_path}")

        csv_path = os.path.join(dataset_path, csv_files[0])
        print(f"Loading data from: {csv_path}")

        # Load the dataset
        df = pd.read_csv(csv_path, nrows=self.max_rows)

        # Check required columns
        required_columns = ['Title', 'Plot']

        # Handle different column name variations
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower and 'Title' not in df.columns:
                column_mapping[col] = 'Title'
            elif 'plot' in col_lower and 'Plot' not in df.columns:
                column_mapping[col] = 'Plot'

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Verify we have the required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")

        # Select only the required columns and drop rows with missing values
        df = df[required_columns].dropna()

        # Basic preprocessing
        df['Title'] = df['Title'].astype(str).str.strip()
        df['Plot'] = df['Plot'].astype(str).str.strip()

        # Remove very short plots (likely errors)
        df = df[df['Plot'].str.len() > 50]

        print(f"Loaded {len(df)} movie plots")

        return df

    def get_sample_data(self) -> pd.DataFrame:
        """
        Get a small sample of data for testing.

        Returns:
            DataFrame with sample movie plots
        """
        # Create sample data for testing without downloading
        sample_data = {
            'Title': [
                '2001: A Space Odyssey',
                'The Matrix',
                'Inception',
                'The Godfather',
                'Pulp Fiction'
            ],
            'Plot': [
                'A mysterious black monolith appears on Earth and in space. Dr. Dave Bowman and other astronauts are sent on a mysterious mission to Jupiter. The HAL 9000 computer becomes antagonistic and kills most of the crew. Bowman disconnects HAL and travels beyond Jupiter where he encounters another monolith and undergoes a transformation.',
                'Thomas Anderson is a computer programmer who leads a double life as a hacker named Neo. He discovers that reality is actually a simulation called the Matrix, created by sentient machines. Neo joins a rebellion led by Morpheus and Trinity to free humanity from the Matrix.',
                'Dom Cobb is a skilled thief who steals secrets from people\'s subconscious during their dreams. He is offered a chance at redemption by performing inception - planting an idea in someone\'s mind. The team must navigate multiple dream levels while facing dangers from Cobb\'s past.',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son Michael. Michael transforms from a war hero who wants nothing to do with the family business into a ruthless mafia boss.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption. The non-linear narrative follows various criminals in Los Angeles as their stories intersect in unexpected ways.'
            ]
        }
        return pd.DataFrame(sample_data)
