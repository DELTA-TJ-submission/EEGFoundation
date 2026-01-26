"""
TUEG Dataset Preprocessing Script for Self-Supervised Pretraining
Convert TUEG EDF files to standardized LMDB format for foundation model pretraining
"""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple, Callable
from multiprocessing import Pool, Manager
from functools import partial

import lmdb
import mne
import numpy as np
from tqdm import tqdm


class TUEGPreprocessor:
    """Preprocessor for TUEG dataset for self-supervised pretraining
    
    This class handles the conversion of TUEG dataset from EDF format
    to a structured LMDB database for EEG foundation model pretraining.
    """
    
    # Channel configurations for different montages
    CHANNEL_CONFIGS = {
        '01_tcp_ar': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
        ],
        '02_tcp_le': [
            'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 
            'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 
            'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 
            'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 
            'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
        ],
        '03_tcp_ar_a': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
        ]
    }
    
    # Preprocessing parameters
    DEFAULT_PARAMS = {
        'target_fs': 200,          # Target sampling frequency (Hz)
        'filter_low': 0.3,         # High-pass filter cutoff (Hz)
        'filter_high': 75.0,       # Low-pass filter cutoff (Hz)
        'notch_freq': 60.0,        # Notch filter frequency (Hz)
        'segment_duration': 30.0,  # Segment duration in seconds
        'start_trim': 60.0,        # Trim from start (seconds)
        'end_trim': 60.0,          # Trim from end (seconds)
        'min_duration': 300.0,     # Minimum recording duration (seconds)
        'artifact_threshold': 100.0,  # Artifact rejection threshold (uV)
        'random_seed': 42,         # Random seed for reproducibility
        'n_jobs': 1,               # Number of parallel jobs
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize TUEG preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self.DEFAULT_PARAMS.copy()
        
        # Set random seed
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Calculate derived parameters
        self.segment_length = int(
            self.config['segment_duration'] * self.config['target_fs']
        )
        self.start_trim_samples = int(
            self.config['start_trim'] * self.config['target_fs']
        )
        self.end_trim_samples = int(
            self.config['end_trim'] * self.config['target_fs']
        )
        self.min_samples = int(
            self.config['min_duration'] * self.config['target_fs']
        )
        
    def _get_channel_config(self, file_path: str) -> Optional[List[str]]:
        """Determine channel configuration based on file path
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            Channel configuration or None if not recognized
        """
        file_path_lower = file_path.lower()
        
        for config_key, channels in self.CHANNEL_CONFIGS.items():
            if config_key in file_path_lower:
                return channels
        
        return None
    
    def _validate_channels(self, raw: mne.io.Raw, required_channels: List[str]) -> bool:
        """Validate that all required channels are present
        
        Args:
            raw: MNE Raw object
            required_channels: List of required channel names
            
        Returns:
            True if all channels are present, False otherwise
        """
        available_channels = set(raw.ch_names)
        required_set = set(required_channels)
        
        missing_channels = required_set - available_channels
        
        if missing_channels:
            print(f"  Warning: Missing channels {missing_channels}")
            return False
        
        return True
    
    def preprocess_eeg_file(
        self, 
        file_path: str, 
        verbose: bool = False
    ) -> Optional[Tuple[str, np.ndarray]]:
        """Preprocess a single EEG file
        
        Args:
            file_path: Path to EDF file
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (file_base_name, processed_data) or None if processing fails
        """
        try:
            file_name = os.path.basename(file_path)
            file_base = os.path.splitext(file_name)[0]
            
            if verbose:
                print(f"Processing: {file_name}")
            
            # Determine channel configuration
            channel_config = self._get_channel_config(file_path)
            if channel_config is None:
                if verbose:
                    print(f"  Skipping: Unknown channel configuration")
                return None
            
            # Load EDF file
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Validate channels
            if not self._validate_channels(raw, channel_config):
                if verbose:
                    print(f"  Skipping: Missing required channels")
                return None
            
            # Select and reorder channels
            raw.pick_channels(channel_config, ordered=True)
            
            # Apply preprocessing pipeline
            raw.resample(self.config['target_fs'])
            raw.filter(l_freq=self.config['filter_low'], 
                      h_freq=self.config['filter_high'])
            raw.notch_filter(self.config['notch_freq'])
            
            # Extract data
            eeg_data = raw.get_data(units='uV').T  # Convert to (n_samples, n_channels)
            
            # Check minimum duration
            if eeg_data.shape[0] < self.min_samples:
                if verbose:
                    print(f"  Skipping: Too short ({eeg_data.shape[0]/self.config['target_fs']:.1f}s)")
                return None
            
            # Trim start and end
            eeg_trimmed = eeg_data[
                self.start_trim_samples:-self.end_trim_samples
                if self.end_trim_samples > 0 else None, :
            ]
            
            # Segment into 30-second epochs
            n_segments = eeg_trimmed.shape[0] // self.segment_length
            eeg_trimmed = eeg_trimmed[:n_segments * self.segment_length, :]
            
            if n_segments == 0:
                if verbose:
                    print(f"  Skipping: No complete segments after trimming")
                return None
            
            # Reshape to (n_segments, n_channels, 30, 200)
            eeg_segmented = eeg_trimmed.reshape(
                n_segments, self.segment_length, -1
            )
            eeg_segmented = eeg_segmented.transpose(0, 2, 1)  # (n_segments, n_channels, segment_length)
            eeg_segmented = eeg_segmented.reshape(
                n_segments, len(channel_config), 30, 200
            )
            
            if verbose:
                print(f"  Created {n_segments} segments, shape: {eeg_segmented.shape}")
            
            return file_base, eeg_segmented
            
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            return None
    
    def artifact_rejection(
        self, 
        segment: np.ndarray, 
        threshold: float = 100.0
    ) -> bool:
        """Check if segment contains artifacts
        
        Args:
            segment: EEG segment (n_channels, 30, 200)
            threshold: Maximum absolute amplitude threshold
            
        Returns:
            True if segment passes artifact rejection, False otherwise
        """
        max_amplitude = np.max(np.abs(segment))
        return max_amplitude < threshold
    
    def process_and_store_file(
        self, 
        file_path: str, 
        db_env: lmdb.Environment,
        key_list: List[str],
        verbose: bool = False
    ) -> int:
        """Process a file and store segments in LMDB
        
        Args:
            file_path: Path to EDF file
            db_env: LMDB environment
            key_list: List to store sample keys
            verbose: Whether to print detailed information
            
        Returns:
            Number of segments stored
        """
        result = self.preprocess_eeg_file(file_path, verbose)
        
        if result is None:
            return 0
        
        file_base, segments = result
        n_segments = len(segments)
        stored_count = 0
        
        # Open transaction
        txn = db_env.begin(write=True)
        
        try:
            for seg_idx, segment in enumerate(segments):
                # Apply artifact rejection
                if not self.artifact_rejection(segment, self.config['artifact_threshold']):
                    continue
                
                # Create unique sample key
                sample_key = f"{file_base}_{seg_idx}"
                
                # Prepare data for storage
                sample_data = {
                    'signal': segment.astype(np.float32),
                    'metadata': {
                        'file': file_base,
                        'segment_idx': seg_idx,
                        'n_segments': n_segments,
                        'n_channels': segment.shape[0],
                        'shape': segment.shape,
                        'max_amplitude': float(np.max(np.abs(segment)))
                    }
                }
                
                # Store in LMDB
                txn.put(
                    key=sample_key.encode(),
                    value=pickle.dumps(sample_data)
                )
                
                # Add to key list
                key_list.append(sample_key)
                stored_count += 1
            
            # Commit transaction
            txn.commit()
            
        except Exception as e:
            txn.abort()
            if verbose:
                print(f"Error storing segments from {file_path}: {e}")
            return 0
        
        if verbose and stored_count > 0:
            print(f"  Stored {stored_count}/{n_segments} segments")
        
        return stored_count
    
    def _process_file_wrapper(
        self, 
        file_path: str, 
        db_path: str,
        key_queue
    ) -> int:
        """Wrapper function for parallel processing
        
        Args:
            file_path: Path to EDF file
            db_path: Path to LMDB database
            key_queue: Queue for collecting sample keys
            
        Returns:
            Number of segments stored
        """
        # Open LMDB environment
        env = lmdb.open(
            db_path, 
            map_size=1099511627776,  # 1TB
            max_dbs=0,
            lock=False,
            readonly=False
        )
        
        # Create temporary key list
        temp_keys = []
        
        # Process file
        stored = self.process_and_store_file(
            file_path, env, temp_keys, verbose=False
        )
        
        # Add keys to queue
        for key in temp_keys:
            key_queue.put(key)
        
        env.close()
        return stored
    
    def collect_edf_files(
        self, 
        root_dir: str,
        file_filter: Optional[Callable[[str], bool]] = None
    ) -> List[str]:
        """Collect all EDF files from directory
        
        Args:
            root_dir: Root directory to search
            file_filter: Optional filter function for file paths
            
        Returns:
            List of EDF file paths
        """
        edf_files = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.edf'):
                    file_path = os.path.join(root, file)
                    
                    # Apply filter if provided
                    if file_filter and not file_filter(file_path):
                        continue
                    
                    edf_files.append(file_path)
        
        return sorted(edf_files)
    
    def filter_file_paths(self, file_list: List[str]) -> List[str]:
        """Filter EDF file paths
        
        Args:
            file_list: List of file paths
            
        Returns:
            Filtered list of file paths
        """
        def is_valid_edf(path: str) -> bool:
            """Check if file is a valid EDF file"""
            # Check extension
            if not path.lower().endswith('.edf'):
                return False
            
            # Check for multiple extensions (excluding .edf)
            base_name = os.path.basename(path)
            name_parts = base_name.split('.')
            
            # Should have exactly 2 parts (name and .edf) or 
            # more if there are version numbers like .edf.gz
            if len(name_parts) > 2 and '.edf' not in base_name.lower():
                return False
            
            return True
        
        return list(filter(is_valid_edf, file_list))
    
    def process_dataset(
        self, 
        root_dir: str, 
        output_path: str,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        map_size: int = 1099511627776
    ) -> None:
        """Process entire TUEG dataset
        
        Args:
            root_dir: Root directory containing raw EDF files
            output_path: Directory for processed LMDB databases
            split_ratios: Train/validation/test split ratios
            map_size: LMDB map size in bytes
        """
        # Collect all EDF files
        print("Collecting EDF files...")
        all_files = self.collect_edf_files(root_dir)
        all_files = self.filter_file_paths(all_files)
        
        if not all_files:
            raise FileNotFoundError(f"No valid EDF files found in {root_dir}")
        
        print(f"Found {len(all_files)} EDF files")
        
        # Shuffle files
        random.shuffle(all_files)
        
        # Split files
        n_files = len(all_files)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])
        
        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }
        
        print(f"\nDataset splits:")
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} files")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process each split
        all_metadata = {}
        
        for split_name, file_list in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            # Create LMDB database for this split
            lmdb_path = os.path.join(output_path, f"{split_name}_database")
            
            # Use Manager for shared key list in multiprocessing
            with Manager() as manager:
                key_queue = manager.Queue()
                
                if self.config['n_jobs'] > 1:
                    # Parallel processing
                    process_func = partial(
                        self._process_file_wrapper,
                        db_path=lmdb_path,
                        key_queue=key_queue
                    )
                    
                    with Pool(processes=self.config['n_jobs']) as pool:
                        results = list(tqdm(
                            pool.imap(process_func, file_list),
                            total=len(file_list),
                            desc=f"Processing {split_name}"
                        ))
                    
                else:
                    # Serial processing
                    results = []
                    for file_path in tqdm(file_list, desc=f"Processing {split_name}"):
                        result = self._process_file_wrapper(
                            file_path, lmdb_path, key_queue
                        )
                        results.append(result)
                
                # Collect all keys
                all_keys = []
                while not key_queue.empty():
                    all_keys.append(key_queue.get())
                
                # Store keys in LMDB
                env = lmdb.open(lmdb_path, map_size=map_size, readonly=False)
                with env.begin(write=True) as txn:
                    txn.put(
                        key=b'__keys__',
                        value=pickle.dumps(all_keys)
                    )
                
                # Collect metadata
                total_segments = sum(results)
                all_metadata[split_name] = {
                    'n_files': len(file_list),
                    'n_segments': total_segments,
                    'lmdb_path': lmdb_path,
                    'segment_duration': self.config['segment_duration'],
                    'n_channels': len(self.CHANNEL_CONFIGS['01_tcp_ar']),
                    'sample_rate': self.config['target_fs']
                }
                
                print(f"  {split_name}: {total_segments} segments")
        
        # Save overall metadata
        metadata_path = os.path.join(output_path, "dataset_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        
        # Print summary
        self._print_summary(all_metadata)
    
    def _print_summary(self, metadata: Dict) -> None:
        """Print dataset processing summary
        
        Args:
            metadata: Dataset metadata dictionary
        """
        print("\n" + "="*60)
        print("TUEG DATASET PROCESSING SUMMARY")
        print("="*60)
        
        total_files = 0
        total_segments = 0
        
        for split_name, stats in metadata.items():
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"  Files: {stats.get('n_files', 0)}")
            print(f"  Segments: {stats.get('n_segments', 0)}")
            print(f"  Segment Duration: {stats.get('segment_duration', 0)}s")
            print(f"  Channels: {stats.get('n_channels', 0)}")
            print(f"  Sample Rate: {stats.get('sample_rate', 0)} Hz")
            print(f"  LMDB Path: {stats.get('lmdb_path', 'N/A')}")
            
            total_files += stats.get('n_files', 0)
            total_segments += stats.get('n_segments', 0)
        
        print("\n" + "="*60)
        print(f"TOTAL: {total_files} files, {total_segments} segments")
        print("="*60)
    
    def validate_dataset(self, lmdb_path: str) -> Dict:
        """Validate processed LMDB dataset
        
        Args:
            lmdb_path: Path to LMDB database
            
        Returns:
            Validation statistics
        """
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        stats = {
            'total_samples': 0,
            'shapes': [],
            'max_amplitudes': [],
            'key_count': 0
        }
        
        with env.begin() as txn:
            # Count all entries
            cursor = txn.cursor()
            stats['key_count'] = sum(1 for _ in cursor)
            
            # Check keys entry
            keys_data = txn.get(b'__keys__')
            if keys_data:
                keys = pickle.loads(keys_data)
                stats['total_samples'] = len(keys)
            
            # Sample a few entries for validation
            cursor.first()
            for _ in range(min(100, stats['key_count'])):
                key = cursor.key()
                value = cursor.value()
                
                if key != b'__keys__':
                    sample_data = pickle.loads(value)
                    stats['shapes'].append(sample_data['signal'].shape)
                    stats['max_amplitudes'].append(
                        sample_data['metadata']['max_amplitude']
                    )
                
                cursor.next()
        
        env.close()
        
        # Calculate statistics
        if stats['shapes']:
            stats['unique_shapes'] = set(stats['shapes'])
            stats['avg_max_amplitude'] = np.mean(stats['max_amplitudes'])
            stats['max_max_amplitude'] = np.max(stats['max_amplitudes'])
        
        return stats


def main():
    """Main function for TUEG dataset preprocessing"""
    # Configuration
    CONFIG = {
        'root_dir': '/path/to/tueg/raw/data',  # Replace with actual path
        'output_dir': '/path/to/tueg/processed',  # Replace with actual path
        'split_ratios': (0.8, 0.1, 0.1),  # Train, validation, test
        'map_size': 1099511627776,  # 1TB
        
        # Preprocessing parameters
        'preprocessing': {
            'target_fs': 200,
            'filter_low': 0.3,
            'filter_high': 75.0,
            'notch_freq': 60.0,
            'segment_duration': 30.0,
            'start_trim': 60.0,
            'end_trim': 60.0,
            'min_duration': 300.0,
            'artifact_threshold': 100.0,
            'random_seed': 42,
            'n_jobs': 4  # Number of parallel processes
        }
    }
    
    # Initialize preprocessor
    preprocessor = TUEGPreprocessor(CONFIG.get('preprocessing'))
    
    # Process dataset
    preprocessor.process_dataset(
        root_dir=CONFIG['root_dir'],
        output_path=CONFIG['output_dir'],
        split_ratios=CONFIG['split_ratios'],
        map_size=CONFIG['map_size']
    )
    
    # Validate the processed dataset
    print("\nValidating processed dataset...")
    for split in ['train', 'val', 'test']:
        lmdb_path = os.path.join(CONFIG['output_dir'], f"{split}_database")
        stats = preprocessor.validate_dataset(lmdb_path)
        print(f"\n{split.upper()} validation:")
        print(f"  Total samples: {stats.get('total_samples', 0)}")
        print(f"  LMDB entries: {stats.get('key_count', 0)}")
        if 'unique_shapes' in stats:
            print(f"  Unique shapes: {stats['unique_shapes']}")
        if 'avg_max_amplitude' in stats:
            print(f"  Avg max amplitude: {stats['avg_max_amplitude']:.2f}uV")
            print(f"  Max max amplitude: {stats['max_max_amplitude']:.2f}uV")


if __name__ == "__main__":
    main()