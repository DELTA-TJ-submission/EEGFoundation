# A foundation model for tokenized spatial-temporal representation learning of electroencephalography signal data


## Overview

EEGFoundation is a novel foundation model that treats neural dynamics as a discrete semantic language, overcoming the limitations of vision-based EEG analysis paradigms. By implementing amplitude-aware tokenization and channel-independent pretraining on a 27,000+ hour EEG corpus, the model learns universal neural oscillation patterns that generalize across diverse EEG analysis tasks.



![Main_fig1](./photos/Main_fig1.png)

**Fig.1 The EEGFoundation framework for spatiotemporal sequence modeling**

## Model Architecture

EEGFoundation follows a three-stage hierarchical approach:

1. **Amplitude-Aware Tokenization**: Continuous EEG signals are normalized and quantized into discrete symbolic tokens that preserve micro-voltage fluctuations while filtering high-frequency noise.

2. **Temporal Pretraining**: Using a RoFormer encoder with Rotary Position Embeddings, the model learns universal temporal dynamics from channel-independent EEG streams.

3. **Spatiotemporal Fusion**: Cross-channel attention dynamically aggregates local representations into a coherent global context for robust downstream task performance.

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/EEGFoundation_github.git
cd EEGFoundation_github

# Install dependencies (Python 3.10+ required)
pip install torch>=2.0.0 transformers>=4.30.0 numpy>=1.24.0 scipy>=1.10.0
pip install mne>=1.4.0 einops>=0.6.0 matplotlib>=3.7.0
```

### Basic Usage

```python
from src.models.downstream_EEGFoundation import load_downstream_model
import torch
import numpy as np

# Load pre-trained model for motor imagery classification
model = load_downstream_model(
    model_path="models/BCIC-2a_model.pth",
    config_path="configs/BCIC_IV_2a_config.json"
)

# Prepare input data (example)
batch_size = 2
num_channels = 20
seq_length = 2000

eeg_signal = torch.randn(batch_size, num_channels, seq_length).float()
embedding = torch.randn(batch_size, 512).float()

# Forward pass
with torch.no_grad():
    outputs = model(input_ids=eeg_signal, embedding_data=embedding)
    predictions = torch.softmax(outputs['logits'], dim=-1)

print(f"Predictions shape: {predictions.shape}")
```

### Demo Data

```python
import numpy as np

# Load example data
demo_data = np.load("demo_data/eeg_data.npy")
print(f"Demo data shape: {demo_data.shape}")

# The demo_data directory contains:
# - eeg_data.npy: Sample EEG recordings
# - downstream_eeg_data.npz: Processed data for downstream tasks
# - make_data.py: Script to generate synthetic EEG data
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
