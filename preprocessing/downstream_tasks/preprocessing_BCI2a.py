"""
BCI Competition IV 2a Dataset Preprocessing Script
Convert raw MAT files to standardized LMDB format
"""

import os
import pickle
from typing import Dict, List, Tuple

import lmdb
import numpy as np
from scipy import io, signal
from scipy.signal import butter, lfilter, resample


def create_bandpass_filter(
    lowcut: float, 
    highcut: float, 
    fs: float, 
    order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Butterworth bandpass filter coefficients
    
    Args:
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling rate (Hz)
        order: Filter order
    
    Returns:
        Filter coefficients (b, a)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def preprocess_eeg_sample(
    raw_signal: np.ndarray,
    fs: float = 250.0,
    lowcut: float = 0.3,
    highcut: float = 50.0,
    t_start: float = 2.0,
    t_end: float = 6.0,
    target_fs: float = 200.0
) -> np.ndarray:
    """Preprocess a single EEG sample
    
    Processing pipeline:
    1. Demeaning
    2. Bandpass filtering
    3. Time window extraction
    4. Resampling
    
    Args:
        raw_signal: Raw EEG signal (n_channels, n_samples)
        fs: Original sampling rate
        lowcut: Bandpass low cutoff
        highcut: Bandpass high cutoff
        t_start: Window start time (seconds)
        t_end: Window end time (seconds)
        target_fs: Target sampling rate
    
    Returns:
        Preprocessed signal (n_channels, n_timesteps)
    """
    # 1. Demean
    signal_centered = raw_signal - np.mean(raw_signal, axis=0, keepdims=True)
    
    # 2. Bandpass filter
    b, a = create_bandpass_filter(lowcut, highcut, fs)
    signal_filtered = lfilter(b, a, signal_centered, axis=-1)
    
    # 3. Extract time window
    start_idx = int(t_start * fs)
    end_idx = int(t_end * fs)
    signal_windowed = signal_filtered[:, start_idx:end_idx]
    
    # 4. Resample
    n_original = signal_windowed.shape[-1]
    n_target = int(n_original * target_fs / fs)
    signal_resampled = resample(signal_windowed, n_target, axis=-1)
    
    # 5. Reshape to spatial-temporal format
    n_channels = signal_resampled.shape[0]
    n_timesteps = signal_resampled.shape[1]
    signal_reshaped = signal_resampled.reshape(n_channels, 4, n_timesteps // 4)
    
    return signal_reshaped


def extract_segments(
    raw_data: np.ndarray,
    events: np.ndarray,
    total_length: int
) -> List[Tuple[int, int]]:
    """Extract event segments from continuous data
    
    Args:
        raw_data: Raw continuous data
        events: Array of event start indices
        total_length: Total data length
    
    Returns:
        List of event segments [(start_idx, end_idx), ...]
    """
    events_list = events.tolist()
    events_list.append(total_length)
    
    segments = []
    for i in range(len(events_list) - 1):
        segments.append((events_list[i], events_list[i + 1]))
    
    return segments


def process_bci2a_dataset(
    input_dir: str,
    output_path: str,
    data_split: Dict[str, List[str]]
) -> None:
    """Process BCI2a dataset
    
    Args:
        input_dir: Directory containing raw MAT files
        output_path: Path for LMDB output
        data_split: Dataset split dictionary
    """
    # Dataset storage structure
    dataset_dict = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    # Create LMDB database
    map_size = 1024 * 1024 * 1024  # 1GB
    env = lmdb.open(output_path, map_size=map_size)
    
    try:
        # Process each dataset split
        for split_name, file_list in data_split.items():
            print(f"\nProcessing {split_name} set...")
            
            for file_name in file_list:
                file_path = os.path.join(input_dir, file_name)
                print(f"  Processing file: {file_name}")
                
                # Load MAT file
                mat_data = io.loadmat(file_path)
                n_trials = len(mat_data['data'][0])
                
                # Process each trial
                for trial_idx in range(3, n_trials):  # Skip first 3 trials
                    trial_data = mat_data['data'][0, trial_idx][0, 0]
                    
                    # Extract data
                    raw_eeg = trial_data[0][:, :22]  # 22 EEG channels
                    events = trial_data[1][:, 0]     # Event markers
                    labels = trial_data[2][:, 0]     # Class labels
                    
                    # Extract event segments
                    data_length = raw_eeg.shape[0]
                    segments = extract_segments(raw_eeg, events, data_length)
                    
                    # Process each event segment
                    for seg_idx, ((start, end), label) in enumerate(zip(segments, labels)):
                        # Extract raw signal
                        raw_segment = raw_eeg[start:end].T  # Convert to (channels, time)
                        
                        # Preprocess
                        processed_sample = preprocess_eeg_sample(raw_segment)
                        
                        # Create sample key
                        sample_key = f"{file_name[:-4]}-{trial_idx}-{seg_idx}"
                        
                        # Prepare data for storage
                        sample_data = {
                            'signal': processed_sample.astype(np.float32),
                            'label': int(label - 1),  # Convert to 0-based indexing
                            'metadata': {
                                'file': file_name,
                                'trial': trial_idx,
                                'segment': seg_idx,
                                'original_label': int(label)
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
                        
                        if seg_idx % 20 == 0:  # Print every 20 samples
                            print(f"    {sample_key}: label={label-1}, shape={processed_sample.shape}")
    
    finally:
        # Save all sample keys
        with env.begin(write=True) as txn:
            txn.put(
                key=b'__keys__',
                value=pickle.dumps(dataset_dict)
            )
            txn.put(
                key=b'__metadata__',
                value=pickle.dumps({
                    'n_samples': {
                        'train': len(dataset_dict['train']),
                        'val': len(dataset_dict['val']),
                        'test': len(dataset_dict['test'])
                    },
                    'n_channels': 22,
                    'input_shape': '(22, 4, 200)',
                    'classes': 4
                })
            )
        
        env.close()
    
    print(f"\nProcessing completed!")
    print(f"Training set: {len(dataset_dict['train'])} samples")
    print(f"Validation set: {len(dataset_dict['val'])} samples")
    print(f"Test set: {len(dataset_dict['test'])} samples")


def main():
    """Main function"""
    # Configuration
    CONFIG = {
        'input_dir': '/path/to/bci2a/raw/data',  # Replace with actual path
        'output_path': '/path/to/processed/bci2a_lmdb',  # Replace with actual path
        
        # Dataset splits
        'data_split': {
            'train': [
                'A01E.mat', 'A01T.mat', 'A02E.mat', 'A02T.mat',
                'A03E.mat', 'A03T.mat', 'A04E.mat', 'A04T.mat',
                'A05E.mat', 'A05T.mat'
            ],
            'val': [
                'A06E.mat', 'A06T.mat', 'A07E.mat', 'A07T.mat'
            ],
            'test': [
                'A08E.mat', 'A08T.mat', 'A09E.mat', 'A09T.mat'
            ],
        },
        
        # Preprocessing parameters
        'preprocess_params': {
            'fs': 250.0,
            'lowcut': 0.3,
            'highcut': 50.0,
            't_start': 2.0,
            't_end': 6.0,
            'target_fs': 200.0
        }
    }
    
    # Create output directory
    output_dir = os.path.dirname(CONFIG['output_path'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset
    process_bci2a_dataset(
        input_dir=CONFIG['input_dir'],
        output_path=CONFIG['output_path'],
        data_split=CONFIG['data_split']
    )


if __name__ == '__main__':
    main()