"""
TUAB Dataset Preprocessing Script
Convert TUAB EDF files to standardized LMDB format
"""

import os
import pickle
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

import lmdb
import mne
import numpy as np


class TUABPreprocessor:
    """Preprocessor for TUAB dataset
    
    This class handles the conversion of TUAB dataset from EDF format
    to a structured LMDB database with standardized preprocessing.
    """
    
    # Channel configuration
    DROP_CHANNELS = [
        'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 
        'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 
        'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG LUC-REF', 
        'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 
        'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 
        'EEG PG1-REF'
    ]
    DROP_CHANNELS.extend([f'EEG {i}-REF' for i in range(20, 129)])
    
    STANDARD_CHANNELS = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
        'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
        'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
        'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
        'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 
        'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
    ]
    
    # Preprocessing parameters
    DEFAULT_PARAMS = {
        'filter_low': 0.1,      # Hz
        'filter_high': 75.0,    # Hz
        'notch_freq': 50.0,     # Hz
        'target_fs': 200.0,     # Hz
        'segment_length': 2000,  # samples (10 seconds at 200 Hz)
        'units': 'uV'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize TUAB preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self.DEFAULT_PARAMS.copy()
        
        # Create error log file
        self.error_log = "tuab_processing_errors.txt"
        
    def preprocess_raw_eeg(self, raw: mne.io.Raw) -> np.ndarray:
        """Apply preprocessing pipeline to raw EEG data
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Preprocessed EEG data (n_channels, n_samples)
            
        Raises:
            ValueError: If channel order doesn't match standard
        """
        # Drop unnecessary channels
        useless_channels = []
        for ch in self.DROP_CHANNELS:
            if ch in raw.ch_names:
                useless_channels.append(ch)
        
        if useless_channels:
            raw.drop_channels(useless_channels)
        
        # Reorder channels to standard order
        if len(self.STANDARD_CHANNELS) == len(raw.ch_names):
            raw.reorder_channels(self.STANDARD_CHANNELS)
        
        # Verify channel order
        if raw.ch_names != self.STANDARD_CHANNELS:
            raise ValueError(f"Channel order mismatch. Expected {self.STANDARD_CHANNELS}, "
                           f"got {raw.ch_names}")
        
        # Apply preprocessing pipeline
        raw.filter(l_freq=self.config['filter_low'], 
                  h_freq=self.config['filter_high'])
        raw.notch_filter(self.config['notch_freq'])
        raw.resample(self.config['target_fs'], n_jobs=5)
        
        # Extract data
        eeg_data = raw.get_data(units=self.config['units'])
        
        return eeg_data
    
    def segment_eeg_data(self, eeg_data: np.ndarray) -> List[np.ndarray]:
        """Segment continuous EEG data into fixed-length segments
        
        Args:
            eeg_data: EEG data (n_channels, n_samples)
            
        Returns:
            List of EEG segments
        """
        segments = []
        n_samples = eeg_data.shape[1]
        segment_len = self.config['segment_length']
        
        for i in range(n_samples // segment_len):
            start_idx = i * segment_len
            end_idx = (i + 1) * segment_len
            segment = eeg_data[:, start_idx:end_idx]
            segments.append(segment)
        
        return segments
    
    def process_subject(self, params: Tuple[str, str, str, int, str]) -> List[Tuple[str, Dict]]:
        """Process a single subject's data
        
        Args:
            params: Tuple containing (data_dir, subject_id, output_dir, label, split_name)
            
        Returns:
            List of (sample_key, sample_data) tuples
        """
        data_dir, subject_id, lmdb_path, label, split_name = params
        samples = []
        
        # Find all files for this subject
        subject_files = []
        for file in os.listdir(data_dir):
            if subject_id in file and file.endswith('.edf'):
                subject_files.append(file)
        
        if not subject_files:
            print(f"Warning: No files found for subject {subject_id}")
            return samples
        
        # Open LMDB environment
        env = lmdb.open(lmdb_path, map_size=1024**4, readonly=False, 
                       lock=True, readahead=False, meminit=False)
        
        try:
            for file_name in subject_files:
                file_path = os.path.join(data_dir, file_name)
                print(f"Processing: {file_name} (label: {label}, split: {split_name})")
                
                try:
                    # Load and preprocess EDF file
                    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                    eeg_data = self.preprocess_raw_eeg(raw)
                    
                    # Segment the data
                    segments = self.segment_eeg_data(eeg_data)
                    
                    # Prepare each segment for storage
                    for seg_idx, segment in enumerate(segments):
                        # Create unique sample key
                        base_name = os.path.splitext(file_name)[0]
                        sample_key = f"{split_name}_{subject_id}_{base_name}_seg{seg_idx:04d}"
                        
                        # Prepare sample data
                        sample_data = {
                            'signal': segment.astype(np.float32),
                            'label': int(label),
                            'metadata': {
                                'subject_id': subject_id,
                                'file_name': file_name,
                                'segment_idx': seg_idx,
                                'split': split_name,
                                'n_channels': segment.shape[0],
                                'n_samples': segment.shape[1],
                                'duration_seconds': segment.shape[1] / self.config['target_fs']
                            }
                        }
                        
                        samples.append((sample_key, sample_data))
                        
                        # Write to LMDB
                        with env.begin(write=True) as txn:
                            txn.put(
                                key=sample_key.encode(),
                                value=pickle.dumps(sample_data)
                            )
                    
                    print(f"  Created {len(segments)} segments from {file_name}")
                    
                except Exception as e:
                    error_msg = f"Error processing {file_name}: {str(e)}\n"
                    print(f"Error: {error_msg}")
                    with open(self.error_log, "a") as f:
                        f.write(error_msg)
                    continue
        
        finally:
            env.close()
        
        return samples
    
    def get_subject_splits(
        self, 
        root_dir: str, 
        val_split: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Dict[str, List[str]]]:
        """Split subjects into train/validation/test sets
        
        Args:
            root_dir: Root directory of TUAB dataset
            val_split: Proportion of training data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing subject IDs for each split
        """
        # Set random seed
        random.seed(seed)
        
        # Define dataset paths
        train_abnormal_dir = os.path.join(root_dir, "train", "abnormal")
        train_normal_dir = os.path.join(root_dir, "train", "normal")
        test_abnormal_dir = os.path.join(root_dir, "eval", "abnormal")
        test_normal_dir = os.path.join(root_dir, "eval", "normal")
        
        # Get subject IDs
        def get_subject_ids(directory):
            if not os.path.exists(directory):
                return []
            files = [f for f in os.listdir(directory) if f.endswith('.edf')]
            subjects = list(set([f.split("_")[0] for f in files]))
            return subjects
        
        train_a_subjects = get_subject_ids(train_abnormal_dir)
        train_n_subjects = get_subject_ids(train_normal_dir)
        
        # Split train subjects into train/validation
        random.shuffle(train_a_subjects)
        random.shuffle(train_n_subjects)
        
        n_val_a = int(len(train_a_subjects) * val_split)
        n_val_n = int(len(train_n_subjects) * val_split)
        
        splits = {
            'train': {
                'abnormal': train_a_subjects[n_val_a:],
                'normal': train_n_subjects[n_val_n:]
            },
            'val': {
                'abnormal': train_a_subjects[:n_val_a],
                'normal': train_n_subjects[:n_val_n]
            },
            'test': {
                'abnormal': get_subject_ids(test_abnormal_dir),
                'normal': get_subject_ids(test_normal_dir)
            }
        }
        
        return splits
    
    def prepare_output_dirs(self, output_root: str) -> Dict[str, str]:
        """Prepare output directories for LMDB databases
        
        Args:
            output_root: Root directory for processed data
            
        Returns:
            Dictionary mapping split names to LMDB paths
        """
        lmdb_paths = {}
        
        for split in ['train', 'val', 'test']:
            lmdb_dir = os.path.join(output_root, split)
            lmdb_path = os.path.join(lmdb_dir, f"{split}_database")
            
            os.makedirs(lmdb_dir, exist_ok=True)
            lmdb_paths[split] = lmdb_path
            
            print(f"{split.capitalize()} LMDB will be created at: {lmdb_path}")
        
        return lmdb_paths
    
    def process_dataset(
        self, 
        root_dir: str, 
        output_root: str,
        n_processes: int = 24
    ) -> None:
        """Process entire TUAB dataset
        
        Args:
            root_dir: Root directory of TUAB dataset
            output_root: Directory for processed LMDB databases
            n_processes: Number of parallel processes
        """
        # Prepare output directories
        lmdb_paths = self.prepare_output_dirs(output_root)
        
        # Get subject splits
        splits = self.get_subject_splits(root_dir)
        
        # Prepare parameters for multiprocessing
        params_list = []
        
        for split_name, split_dict in splits.items():
            lmdb_path = lmdb_paths[split_name]
            
            # Abnormal subjects (label=1)
            for label_type, subjects, label in [
                ('abnormal', split_dict['abnormal'], 1),
                ('normal', split_dict['normal'], 0)
            ]:
                data_dir = os.path.join(root_dir, split_name, label_type)
                
                for subject_id in subjects:
                    params_list.append(
                        (data_dir, subject_id, lmdb_path, label, split_name)
                    )
        
        print(f"\nTotal subjects to process: {len(params_list)}")
        
        # Process subjects in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.map(self.process_subject, params_list)
        
        # Collect metadata and statistics
        metadata = {'train': {}, 'val': {}, 'test': {}}
        
        for split_name, lmdb_path in lmdb_paths.items():
            # Count samples in each database
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                n_samples = sum(1 for _ in cursor)
            
            # Store metadata
            metadata[split_name] = {
                'n_samples': n_samples,
                'path': lmdb_path
            }
            
            print(f"{split_name.capitalize()}: {n_samples} samples")
        
        # Save overall metadata
        metadata_path = os.path.join(output_root, "dataset_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"\nProcessing complete!")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Main function for TUAB dataset preprocessing"""
    # Configuration
    CONFIG = {
        'root_dir': '/path/to/tuab/raw/data',  # Replace with actual path
        'output_root': '/path/to/tuab/processed',  # Replace with actual path
        'n_processes': 24,
        'val_split': 0.1,
        'random_seed': 42
    }
    
    # Initialize preprocessor
    preprocessor = TUABPreprocessor()
    
    # Process dataset
    preprocessor.process_dataset(
        root_dir=CONFIG['root_dir'],
        output_root=CONFIG['output_root'],
        n_processes=CONFIG['n_processes']
    )


if __name__ == "__main__":
    main()