import os
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from autogluon.tabular import TabularPredictor
import parameters
from tqdm import tqdm
import concurrent.futures
import numpy as np
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./inference_logging.txt', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Optimized parallel predictor with efficient batch processing"""

    def __init__(self, model_path: str, label: str, batch_size: int = 2000):
        self.model_path = Path(model_path)
        self.label = label
        self.batch_size = batch_size
        self.predictor = None
        self._model_loaded = False

    def load_model(self) -> None:
        """Load model once and reuse across batches"""
        if not self._model_loaded:
            logger.info(f"Loading model from {self.model_path}")
            self.predictor = TabularPredictor.load(str(self.model_path))
            self._model_loaded = True

    @staticmethod
    def _process_sequence_batch(batch: List[str]) -> Optional[pd.DataFrame]:
        """Process a batch of sequences to features"""
        try:
            features = [parameters.cal_pep(seq) if '_' not in seq else None for seq in batch]
            valid_features = [f for f in features if f is not None]
            return pd.DataFrame(valid_features) if valid_features else None
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return None

    def _predict_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Predict a batch of features"""
        try:
            return self.predictor.predict(features)
        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            return np.array([])

    def process_and_predict(self, sequences: List[str]) -> pd.DataFrame:
        """
        Full processing pipeline with parallel batch processing
        Returns DataFrame with sequences and predictions
        """
        self.load_model()

        # Process sequences in parallel batches
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Process sequence batches to features
            feature_batches = []
            seq_batches = [sequences[i:i + self.batch_size] for i in range(0, len(sequences), self.batch_size)]

            with tqdm(total=len(seq_batches), desc="Processing sequences") as pbar:
                for feature_df in executor.map(self._process_sequence_batch, seq_batches):
                    if feature_df is not None:
                        feature_batches.append(feature_df)
                    pbar.update(1)

            if not feature_batches:
                raise ValueError("No valid features generated from sequences")

            features_df = pd.concat(feature_batches, ignore_index=True)

            # Predict in parallel batches
            prediction_batches = []
            predict_batches = [features_df.iloc[i:i + self.batch_size]
                               for i in range(0, len(features_df), self.batch_size)]

            with tqdm(total=len(predict_batches), desc="Making predictions") as pbar:
                for preds in executor.map(self._predict_batch, predict_batches):
                    if len(preds) > 0:
                        prediction_batches.append(preds)
                    pbar.update(1)

            predictions = np.concatenate(prediction_batches) if prediction_batches else np.array([])

        # Align results with original sequences
        results = pd.DataFrame({
            'AA_sequence': sequences[:len(predictions)],
            f'predicted_{self.label}': predictions
        })

        return results

    def predict_from_file(self, input_path: str, output_path: str) -> None:
        """Full pipeline from input file to output file"""
        try:
            # Efficient file reading with chunks if needed
            if os.path.getsize(input_path) > 100 * 1024 * 1024:  # >100MB
                chunks = pd.read_csv(input_path, chunksize=100000)
                sequences = []
                for chunk in chunks:
                    sequences.extend(chunk['AA_sequence'].tolist())
            else:
                df = pd.read_csv(input_path)
                sequences = df['AA_sequence'].tolist()

            logger.info(f"Processing {len(sequences)} sequences")

            results = self.process_and_predict(sequences)

            # Efficient file writing
            if len(results) > 1000000:  # >1M rows
                results.to_csv(output_path, index=False, chunksize=100000)
            else:
                results.to_csv(output_path, index=False)

            logger.info(f"Saved predictions to {output_path}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimized parallel AutoML predictor")
    parser.add_argument('--input', required=True, help="Input CSV file with AA_sequence column")
    parser.add_argument('--output', default='./predictions.csv', help="Output CSV path")
    parser.add_argument('--task', default='hTfR1_avi_mean__V8L5_8_mean_log2_enr',
                        help="Target task label")
    parser.add_argument('--model_dir', default='./model', help="Directory with trained model")
    parser.add_argument('--batch_size', type=int, default=2000, help="Processing batch size")

    args = parser.parse_args()

    try:
        predictor = ModelPredictor(
            model_path=args.model_dir,
            label=args.task,
            batch_size=args.batch_size
        )

        predictor.predict_from_file(args.input, args.output)

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()