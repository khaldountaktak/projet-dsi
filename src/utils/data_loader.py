"""
Data Loader for ISO Standards CSV files
Handles loading and processing of labeled ISO questionnaires
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISODataLoader:
    """Loader for ISO standards questionnaire data"""
    
    def __init__(self, data_dir: str = "/home/khaldoun/prjt_vap/data"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Root directory containing the data folders
        """
        self.data_dir = Path(data_dir)
        self.method1_dir = self.data_dir / "method 1"
        self.method2_dir = self.data_dir / "method 2"
        
    def load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_method_data(self, method: int = 1) -> pd.DataFrame:
        """
        Load all CSV files from a specific method directory
        
        Args:
            method: Method number (1 or 2)
            
        Returns:
            Combined DataFrame with all ISO standards
        """
        method_dir = self.method1_dir if method == 1 else self.method2_dir
        
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")
        
        # Get all CSV files
        csv_files = list(method_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {method_dir}")
        
        # Load and combine all files
        dataframes = []
        for file_path in sorted(csv_files):
            df = self.load_single_file(file_path)
            # Extract ISO standard name from filename
            iso_standard = file_path.stem.replace("labeled_our_", "").replace("labeled_", "")
            df['iso_standard'] = iso_standard
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded total of {len(combined_df)} records from method {method}")
        
        return combined_df
    
    def load_specific_standard(self, standard: str, method: int = 1) -> pd.DataFrame:
        """
        Load data for a specific ISO standard
        
        Args:
            standard: ISO standard name (e.g., 'iso_27001')
            method: Method number (1 or 2)
            
        Returns:
            DataFrame with the specified standard data
        """
        method_dir = self.method1_dir if method == 1 else self.method2_dir
        
        # Try different filename patterns
        possible_names = [
            f"labeled_{standard}.csv",
            f"labeled_our_{standard}.csv"
        ]
        
        for name in possible_names:
            file_path = method_dir / name
            if file_path.exists():
                df = self.load_single_file(file_path)
                df['iso_standard'] = standard
                return df
        
        raise FileNotFoundError(f"Standard {standard} not found in method {method}")
    
    def get_documents_for_rag(self, method: int = 1) -> List[Dict]:
        """
        Prepare documents in RAG-friendly format
        
        Args:
            method: Method number (1 or 2)
            
        Returns:
            List of dictionaries with document content and metadata
        """
        df = self.load_method_data(method)
        
        documents = []
        for _, row in df.iterrows():
            doc = {
                'id': row['id'],
                'content': row['text'],
                'metadata': {
                    'title': row['title'],
                    'labels': row['labels'],
                    'iso_standard': row['iso_standard']
                }
            }
            documents.append(doc)
        
        logger.info(f"Prepared {len(documents)} documents for RAG")
        return documents
    
    def get_statistics(self, method: int = 1) -> Dict:
        """
        Get statistics about the loaded data
        
        Args:
            method: Method number (1 or 2)
            
        Returns:
            Dictionary with statistics
        """
        df = self.load_method_data(method)
        
        stats = {
            'total_questions': len(df),
            'standards': df['iso_standard'].unique().tolist(),
            'questions_per_standard': df['iso_standard'].value_counts().to_dict(),
            'unique_titles': df['title'].nunique(),
            'titles': df['title'].unique().tolist()
        }
        
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = ISODataLoader()
    
    # Load method 1 data
    print("\n=== Loading Method 1 Data ===")
    docs = loader.get_documents_for_rag(method=1)
    print(f"Loaded {len(docs)} documents")
    print(f"\nFirst document example:")
    print(docs[0])
    
    # Get statistics
    print("\n=== Statistics ===")
    stats = loader.get_statistics(method=1)
    for key, value in stats.items():
        print(f"{key}: {value}")
