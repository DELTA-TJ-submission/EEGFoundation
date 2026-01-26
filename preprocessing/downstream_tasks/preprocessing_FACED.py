"""
FACED Dataset Preprocessing Script
Convert processed FACED data to standardized LMDB format
"""

import os
import pickle
from typing import Dict, List, Tuple

import lmdb
import numpy as np
from scipy import signal


def preprocess_faced_sample(
    eeg_signal: np.ndarray,
    target_length: int = 6000
) -> np.ndarray:
    """Preprocess a single FACED EEG sample
    
    Processing steps:
    1. Resample to target length
    2. Reshape to temporal-spatial format
    
    Args:
        eeg_signal: Raw EEG signal (n_samples, n_channels, n_timesteps)
        target_length: Target temporal length after resampling
    
    Returns:
        Reshaped signal (n_samples, n_channels, 30, 200)
    """
    # Resample to target length
    eeg_resampled = signal.resample(eeg_signal, target_length, axis=2)
    
    # Reshape to (n_samples, n_channels, 30, 200)
    eeg_reshaped = eeg_resampled.reshape(28, 32, 30, 200)
    
    return eeg_reshaped


def extract_sliding_windows(
    sample: np.ndarray,
    window_size: int = 10,
    stride: int = 10
) -> List[np.ndarray]:
    """Extract sliding windows from a sample
    
    Args:
        sample: EEG sample (n_channels, 30, 200)
        window_size: Number of time segments per window
        stride: Step size between windows
    
    Returns:
        List of windowed samples
    """
    windows = []
    n_windows = sample.shape[1] // stride
    
    for j in range(n_windows):
        start_idx = j * stride
        end_idx = start_idx + window_size
        window = sample[:, start_idx:end_idx, :]
        windows.append(window)
    
    return windows


def process_faced_dataset(
    input_dir: str,
    output_path: str,
    data_split: Dict[str, List[str]],
    labels: np.ndarray,
    map_size: int = 6612500172
) -> None:
    """Process FACED dataset
    
    Args:
        input_dir: Directory containing processed FACED data
        output_path: Path for LMDB output
        data_split: Dictionary defining train/val/test splits
        labels: Array of labels for each sample
        map_size: LMDB map size in bytes
    """
    # Verify labels shape
    if len(labels) != 28:
        raise ValueError(f"Expected 28 labels, got {len(labels)}")
    
    # Dataset storage structure
    dataset_dict = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    # Create LMDB database
    env = lmdb.open(output_path, map_size=map_size)
    
    try:
        # Process each dataset split
        for split_name, file_list in data_split.items():
            print(f"\nProcessing {split_name} set ({len(file_list)} files)...")
            
            for file_idx, file_name in enumerate(file_list):
                file_path = os.path.join(input_dir, file_name)
                
                # Load and process each file
                with open(file_path, 'rb') as f:
                    raw_data = pickle.load(f)
                
                # Preprocess the entire file
                processed_data = preprocess_faced_sample(raw_data)
                
                # Process each sample in the file
                n_samples = processed_data.shape[0]
                
                for sample_idx in range(n_samples):
                    sample = processed_data[sample_idx]
                    label = labels[sample_idx]
                    
                    # Extract sliding windows
                    windows = extract_sliding_windows(sample)
                    
                    # Store each window
                    for window_idx, window in enumerate(windows):
                        # Create unique sample key
                        sample_key = f"{file_name}-{sample_idx}-{window_idx}"
                        
                        # Prepare data for storage
                        sample_data = {
                            'signal': window.astype(np.float32),
                            'label': int(label),
                            'metadata': {
                                'file': file_name,
                                'original_sample_idx': sample_idx,
                                'window_idx': window_idx,
                                'shape': window.shape
                            }
                        }
                        
                        # Write to LMDB
                        with env.begin(write=True) as txn:
                            txn.put(
                                key=sample_key.encode(),
                                value=pickle.dumps(sample_data)
                            )
                        
                        # Record sample key
                        dataset_dict[split_name].append(sample_key)
                    
                    # Print progress periodically
                    if sample_idx % 7 == 0:  # Print every 7 samples
                        print(f"  File {file_idx+1}/{len(file_list)}: "
                              f"Sample {sample_idx+1}/{n_samples}, "
                              f"Label: {label}, Windows: {len(windows)}")
    
    finally:
        # Save dataset metadata
        with env.begin(write=True) as txn:
            # Save sample keys
            txn.put(
                key=b'__keys__',
                value=pickle.dumps(dataset_dict)
            )
            
            # Save dataset statistics
            txn.put(
                key=b'__metadata__',
                value=pickle.dumps({
                    'n_samples_total': {
                        'train': len(dataset_dict['train']),
                        'val': len(dataset_dict['val']),
                        'test': len(dataset_dict['test'])
                    },
                    'n_files': {
                        'train': len(data_split['train']),
                        'val': len(data_split['val']),
                        'test': len(data_split['test'])
                    },
                    'n_channels': 32,
                    'input_shape': '(32, 10, 200)',
                    'n_classes': len(set(labels)),
                    'class_distribution': {
                        int(cls): int(np.sum(labels == cls)) 
                        for cls in np.unique(labels)
                    },
                    'window_info': {
                        'window_size': 10,
                        'stride': 10,
                        'windows_per_sample': 3
                    }
                })
            )
        
        env.close()
    
    # Print summary
    print(f"\nProcessing completed!")
    print(f"Total samples:")
    print(f"  Training set: {len(dataset_dict['train'])}")
    print(f"  Validation set: {len(dataset_dict['val'])}")
    print(f"  Test set: {len(dataset_dict['test'])}")
    print(f"  Total: {sum(len(v) for v in dataset_dict.values())}")


def get_data_split(
    all_files: List[str],
    train_ratio: float = 0.6667,
    val_ratio: float = 0.1667
) -> Dict[str, List[str]]:
    """Split files into train/validation/test sets
    
    Args:
        all_files: List of all file names
        train_ratio: Proportion of files for training
        val_ratio: Proportion of files for validation
    
    Returns:
        Dictionary with train/val/test file lists
    """
    n_files = len(all_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    return {
        'train': all_files[:n_train],
        'val': all_files[n_train:n_train + n_val],
        'test': all_files[n_train + n_val:]
    }


def main():
    """Main function"""
    # Configuration
    CONFIG = {
        'input_dir': '/path/to/faced/processed/data',  # Replace with actual path
        'output_path': '/path/to/faced/processed_lmdb',  # Replace with actual path
        'map_size': 6612500172,  # Approximately 6.6GB
        
        # Labels for each sample 
        'labels': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 
                           4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]),
        
        # Dataset split ratios
        'split_ratios': {
            'train': 0.6667,  # ~80 files for 120 total
            'val': 0.1667,    # ~20 files for 120 total
            'test': 0.1667    # ~20 files for 120 total
        }
    }
    
    # Get all files
    all_files = sorted([f for f in os.listdir(CONFIG['input_dir']) 
                       if f.endswith('.pkl') or f.endswith('.pickle')])
    
    if not all_files:
        raise FileNotFoundError(f"No data files found in {CONFIG['input_dir']}")
    
    print(f"Found {len(all_files)} data files")
    
    # Create data split
    data_split = get_data_split(
        all_files, 
        train_ratio=CONFIG['split_ratios']['train'],
        val_ratio=CONFIG['split_ratios']['val']
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(data_split['train'])} files")
    print(f"  Validation: {len(data_split['val'])} files")
    print(f"  Test: {len(data_split['test'])} files")
    
    # Create output directory
    output_dir = os.path.dirname(CONFIG['output_path'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset
    process_faced_dataset(
        input_dir=CONFIG['input_dir'],
        output_path=CONFIG['output_path'],
        data_split=data_split,
        labels=CONFIG['labels'],
        map_size=CONFIG['map_size']
    )


if __name__ == '__main__':
    main()