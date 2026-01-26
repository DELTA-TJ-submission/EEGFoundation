"""
Stress/EEG Mental Arithmetic Dataset Preprocessing Script
Convert EEG EDF files to standardized LMDB format
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional

import lmdb
import mne
import numpy as np


class StressDatasetPreprocessor:
    """Preprocessor for Stress/EEG Mental Arithmetic Dataset
    
    This class handles the conversion of stress/mental arithmetic EEG dataset
    from EDF format to a structured LMDB database with standardized preprocessing.
    """
    
    # Standard 20-channel EEG montage
    STANDARD_CHANNELS = [
        'EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 
        'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 
        'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 
        'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1'
    ]
    
    # Preprocessing parameters
    DEFAULT_PARAMS = {
        'target_fs': 200,          # Target sampling frequency (Hz)
        'segment_duration': 5.0,   # Segment duration in seconds
        'overlap': 0.0,           # Overlap between segments (0-1)
        'units': 'uV',             # Units for EEG data
        'random_seed': 42          # Random seed for reproducibility
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Stress Dataset preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self.DEFAULT_PARAMS.copy()
        
        # Calculate samples per segment
        self.samples_per_segment = int(
            self.config['segment_duration'] * self.config['target_fs']
        )
    
    def validate_channels(self, raw: mne.io.Raw) -> None:
        """Validate that all required channels are present
        
        Args:
            raw: MNE Raw object
            
        Raises:
            ValueError: If required channels are missing
        """
        missing_channels = [
            ch for ch in self.STANDARD_CHANNELS 
            if ch not in raw.ch_names
        ]
        
        if missing_channels:
            raise ValueError(
                f"Missing required channels: {missing_channels}. "
                f"Available channels: {raw.ch_names}"
            )
    
    def preprocess_raw_eeg(self, raw: mne.io.Raw) -> np.ndarray:
        """Apply preprocessing pipeline to raw EEG data
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Preprocessed EEG data (n_channels, n_samples)
        """
        # Validate channels
        self.validate_channels(raw)
        
        # Select and reorder channels
        raw.pick(self.STANDARD_CHANNELS)
        raw.reorder_channels(self.STANDARD_CHANNELS)
        
        # Resample to target frequency
        raw.resample(self.config['target_fs'])
        
        # Extract data
        eeg_data = raw.get_data(units=self.config['units'])
        
        return eeg_data
    
    def segment_eeg_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """Segment continuous EEG data into fixed-duration segments
        
        Args:
            eeg_data: EEG data (n_channels, n_samples)
            
        Returns:
            Segmented EEG data (n_segments, n_channels, 5, 200)
        """
        n_channels, n_samples = eeg_data.shape
        
        # Calculate samples per segment
        segment_samples = int(self.config['segment_duration'] * self.config['target_fs'])
        
        # Trim to multiple of segment duration
        n_segments = n_samples // segment_samples
        n_samples_trimmed = n_segments * segment_samples
        eeg_trimmed = eeg_data[:, :n_samples_trimmed]
        
        if n_samples != n_samples_trimmed:
            print(f"  Trimmed {n_samples - n_samples_trimmed} samples "
                  f"({(n_samples - n_samples_trimmed)/self.config['target_fs']:.1f}s)")
        
        # Reshape to segments
        # First reshape to (n_channels, n_segments, 5, 200)
        eeg_reshaped = eeg_trimmed.reshape(
            n_channels, n_segments, 5, segment_samples // 5
        )
        
        # Transpose to (n_segments, n_channels, 5, 200)
        eeg_segmented = eeg_reshaped.transpose(1, 0, 2, 3)
        
        return eeg_segmented
    
    def extract_label_from_filename(self, filename: str) -> int:
        """Extract label from filename
        
        Args:
            filename: EEG file name
            
        Returns:
            Integer label (0-based)
            
        Raises:
            ValueError: If label cannot be extracted
        """
        try:
            # Assuming label is the 5th character from the end
            # Adjust this based on actual filename pattern
            label_char = filename[-5]
            label = int(label_char) - 1  # Convert to 0-based
            
            if label not in [0, 1, 2]:  # Assuming 3 stress levels
                raise ValueError(f"Invalid label {label} from filename {filename}")
            
            return label
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot extract label from filename {filename}: {e}")
    
    def process_file(self, file_path: str) -> Tuple[np.ndarray, int, str]:
        """Process a single EEG file
        
        Args:
            file_path: Path to EEG file
            
        Returns:
            Tuple of (segmented_data, label, file_name)
        """
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        
        # Load and preprocess EEG data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        eeg_data = self.preprocess_raw_eeg(raw)
        
        # Extract label from filename
        label = self.extract_label_from_filename(file_name)
        
        # Segment the data
        segmented_data = self.segment_eeg_data(eeg_data)
        
        print(f"  Extracted {segmented_data.shape[0]} segments, label: {label}")
        
        return segmented_data, label, file_name
    
    def save_to_lmdb(
        self, 
        data: List[Tuple[np.ndarray, int, str]], 
        split: str, 
        lmdb_path: str,
        map_size: int = 1000000000
    ) -> List[str]:
        """Save processed data to LMDB database
        
        Args:
            data: List of (segmented_data, label, file_name) tuples
            split: Dataset split name
            lmdb_path: Path to LMDB database
            map_size: LMDB map size in bytes
            
        Returns:
            List of sample keys
        """
        sample_keys = []
        
        # Open LMDB environment
        env = lmdb.open(lmdb_path, map_size=map_size, readonly=False)
        
        try:
            for segmented_data, label, file_name in data:
                base_name = os.path.splitext(file_name)[0]
                
                for seg_idx, segment in enumerate(segmented_data):
                    # Create unique sample key
                    sample_key = f"{split}_{base_name}_seg{seg_idx:04d}"
                    
                    # Prepare sample data
                    sample_data = {
                        'signal': segment.astype(np.float32),
                        'label': int(label),
                        'metadata': {
                            'file': file_name,
                            'split': split,
                            'segment_idx': seg_idx,
                            'n_segments': len(segmented_data),
                            'n_channels': segment.shape[0],
                            'segment_shape': segment.shape,
                            'original_label': label + 1  # Store 1-based label
                        }
                    }
                    
                    # Write to LMDB
                    with env.begin(write=True) as txn:
                        txn.put(
                            key=sample_key.encode(),
                            value=pickle.dumps(sample_data)
                        )
                    
                    sample_keys.append(sample_key)
        
        finally:
            env.close()
        
        return sample_keys
    
    def create_data_splits(
        self, 
        file_list: List[str], 
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    ) -> Dict[str, List[str]]:
        """Split files into train/validation/test sets
        
        Args:
            file_list: List of all file names
            split_ratios: Tuple of (train_ratio, val_ratio, test_ratio)
            
        Returns:
            Dictionary with file splits
        """
        # Validate split ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        n_files = len(file_list)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])
        
        splits = {
            'train': file_list[:train_end],
            'val': file_list[train_end:val_end],
            'test': file_list[val_end:]
        }
        
        return splits
    
    def analyze_labels(self, data_dir: str, file_list: List[str]) -> Dict[int, int]:
        """Analyze label distribution in dataset
        
        Args:
            data_dir: Directory containing data files
            file_list: List of file names to analyze
            
        Returns:
            Dictionary mapping label to count
        """
        label_counts = {}
        
        for file_name in file_list:
            try:
                label = self.extract_label_from_filename(file_name)
                label_counts[label] = label_counts.get(label, 0) + 1
            except ValueError as e:
                print(f"Warning: {e}")
                continue
        
        return label_counts
    
    def process_dataset(
        self, 
        data_dir: str, 
        output_path: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        map_size: int = 1000000000
    ) -> None:
        """Process entire stress dataset
        
        Args:
            data_dir: Directory containing raw EDF files
            output_path: Path for LMDB output
            split_ratios: Train/validation/test split ratios
            map_size: LMDB map size in bytes
        """
        # Get all EDF files
        all_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.endswith('.edf')
        ])
        
        if not all_files:
            raise FileNotFoundError(f"No EDF files found in {data_dir}")
        
        print(f"Found {len(all_files)} EDF files")
        
        # Analyze label distribution
        print("\nLabel distribution in full dataset:")
        label_counts = self.analyze_labels(data_dir, all_files)
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} files")
        
        # Create data splits
        splits = self.create_data_splits(all_files, split_ratios)
        
        print(f"\nData splits:")
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} files")
            split_label_counts = self.analyze_labels(data_dir, files)
            for label, count in sorted(split_label_counts.items()):
                print(f"    Label {label}: {count} files")
        
        # Process each split
        all_sample_keys = {}
        split_statistics = {}
        
        for split_name, file_list in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            split_data = []
            for file_name in file_list:
                file_path = os.path.join(data_dir, file_name)
                
                try:
                    segmented_data, label, file_name = self.process_file(file_path)
                    split_data.append((segmented_data, label, file_name))
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue
            
            if not split_data:
                print(f"Warning: No data processed for {split_name} split")
                all_sample_keys[split_name] = []
                split_statistics[split_name] = {'n_samples': 0}
                continue
            
            # Save to LMDB
            lmdb_split_path = f"{output_path}_{split_name}"
            sample_keys = self.save_to_lmdb(
                split_data, split_name, lmdb_split_path, map_size
            )
            
            # Calculate statistics
            n_segments = sum(data[0].shape[0] for data in split_data)
            label_dist = {}
            for _, label, _ in split_data:
                label_dist[label] = label_dist.get(label, 0) + 1
            
            all_sample_keys[split_name] = sample_keys
            split_statistics[split_name] = {
                'n_files': len(split_data),
                'n_segments': n_segments,
                'label_distribution': label_dist,
                'lmdb_path': lmdb_split_path
            }
            
            print(f"  Saved {n_segments} segments to {lmdb_split_path}")
        
        # Save metadata and keys
        self.save_metadata(
            output_path, all_sample_keys, split_statistics, split_ratios
        )
        
        # Print summary
        self.print_summary(split_statistics)
    
    def save_metadata(
        self, 
        output_path: str, 
        sample_keys: Dict[str, List[str]], 
        statistics: Dict[str, Dict], 
        split_ratios: Tuple[float, float, float]
    ) -> None:
        """Save dataset metadata
        
        Args:
            output_path: Base path for metadata files
            sample_keys: Dictionary of sample keys per split
            statistics: Dataset statistics
            split_ratios: Split ratios used
        """
        # Save sample keys
        keys_path = f"{output_path}_keys.pkl"
        with open(keys_path, 'wb') as f:
            pickle.dump(sample_keys, f)
        
        # Save comprehensive metadata
        metadata = {
            'statistics': statistics,
            'split_ratios': split_ratios,
            'preprocessing_params': self.config,
            'channels': self.STANDARD_CHANNELS,
            'total_samples': {
                split: len(keys) for split, keys in sample_keys.items()
            }
        }
        
        metadata_path = f"{output_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nMetadata saved to:")
        print(f"  Sample keys: {keys_path}")
        print(f"  Metadata: {metadata_path}")
    
    def print_summary(self, statistics: Dict[str, Dict]) -> None:
        """Print dataset processing summary
        
        Args:
            statistics: Dataset statistics
        """
        print("\n" + "="*60)
        print("DATASET PROCESSING SUMMARY")
        print("="*60)
        
        total_files = 0
        total_segments = 0
        
        for split_name, stats in statistics.items():
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"  Files: {stats.get('n_files', 0)}")
            print(f"  Segments: {stats.get('n_segments', 0)}")
            print(f"  LMDB Path: {stats.get('lmdb_path', 'N/A')}")
            
            label_dist = stats.get('label_distribution', {})
            if label_dist:
                print("  Label Distribution:")
                for label, count in sorted(label_dist.items()):
                    print(f"    Label {label}: {count} files")
            
            total_files += stats.get('n_files', 0)
            total_segments += stats.get('n_segments', 0)
        
        print("\n" + "="*60)
        print(f"TOTAL: {total_files} files, {total_segments} segments")
        print("="*60)


def main():
    """Main function for Stress Dataset preprocessing"""
    # Configuration
    CONFIG = {
        'data_dir': '/path/to/stress/eeg/data',  # Replace with actual path
        'output_path': '/path/to/output/stress_dataset',  # Replace with actual path
        'split_ratios': (0.7, 0.1, 0.2),  # Train, validation, test
        'map_size': 1000000000,  # 1GB LMDB map size
        
        # Preprocessing parameters (optional override)
        'preprocessing': {
            'target_fs': 200,
            'segment_duration': 5.0,
            'units': 'uV'
        }
    }
    
    # Initialize preprocessor
    preprocessor = StressDatasetPreprocessor(CONFIG.get('preprocessing'))
    
    # Process dataset
    preprocessor.process_dataset(
        data_dir=CONFIG['data_dir'],
        output_path=CONFIG['output_path'],
        split_ratios=CONFIG['split_ratios'],
        map_size=CONFIG['map_size']
    )


if __name__ == "__main__":
    main()