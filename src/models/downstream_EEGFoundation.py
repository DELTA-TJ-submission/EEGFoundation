"""
EEG Downstream Task Model
HuggingFace-compatible EEG downstream model for classification tasks
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any, List
import json
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from einops import rearrange
from transformers import AutoConfig, AutoModel
import numpy as np

from utils import MultiHeadAttention


@dataclass
class EEGFoundationDownstreamClassifierConfig:
    """Configuration for EEGFoundationDownstreamClassifier using dataclass format"""
    
    # Model architecture parameters
    seq_len: int = 2000
    patch_size: int = 150
    stride: int = 1
    d_model: int = 512
    num_classes: int = 2
    num_channel: int = 20
    
    # Attention and normalization
    rms_norm: bool = False
    
    # Embedding parameters
    embedding_dim: int = 512
    projection_embedding_dim: int = 512
    
    # Classification head parameters
    classification_dropout: float = 0.5
    classification_hidden_dim: int = 512
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    
    # Additional parameters
    model_type: str = "eeg-foundation-downstream-classifier"
    
    # Path parameters (added for loading pretrained models)
    pretrained_model_path: Optional[str] = None
    config_path: Optional[str] = None


class EEGFoundationDownstreamClassifierConfigHF(PretrainedConfig):
    """HuggingFace compatible configuration for EEGFoundationDownstreamClassifier"""
    
    model_type = "eeg-foundation-downstream-classifier"
    
    def __init__(
        self,
        # Model architecture parameters
        seq_len: int = 2000,
        patch_size: int = 150,
        stride: int = 1,
        d_model: int = 512,
        num_classes: int = 2,
        num_channel: int = 20,
        
        # Attention and normalization
        rms_norm: bool = False,        
        # Embedding parameters
        embedding_dim: int = 512,
        projection_embedding_dim: int = 512,
        
        # Classification head parameters
        classification_dropout: float = 0.5,
        classification_hidden_dim: int = 512,
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        
        # Path parameters (added for loading pretrained models)
        pretrained_model_path: Optional[str] = None,
        config_path: Optional[str] = None,
       
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Model architecture parameters
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_channel = num_channel
        
        # Attention and normalization
        self.rms_norm = rms_norm

        # Embedding parameters
        self.embedding_dim = embedding_dim
        self.projection_embedding_dim = projection_embedding_dim
        
        # Classification head parameters
        self.classification_dropout = classification_dropout
        self.classification_hidden_dim = classification_hidden_dim
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Path parameters
        self.pretrained_model_path = pretrained_model_path
        self.config_path = config_path
        
        # Calculate derived parameters
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 2
        
    @classmethod
    def from_dataclass(cls, config: EEGFoundationDownstreamClassifierConfig) -> 'EEGFoundationDownstreamClassifierConfigHF':
        """Create from dataclass configuration"""
        return cls(
            seq_len=config.seq_len,
            patch_size=config.patch_size,
            stride=config.stride,
            d_model=config.d_model,
            num_classes=config.num_classes,
            num_channel=config.num_channel,
            rms_norm=config.rms_norm,
            embedding_dim=config.embedding_dim,
            projection_embedding_dim=config.projection_embedding_dim,
            classification_dropout=config.classification_dropout,
            classification_hidden_dim=config.classification_hidden_dim,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            pretrained_model_path=config.pretrained_model_path,
            config_path=config.config_path
        )
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'EEGFoundationDownstreamClassifierConfigHF':
        """Load configuration from JSON file
        
        Args:
            json_file: Path to JSON configuration file
            
        Returns:
            Configuration object
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def save_to_json(self, json_file: str) -> None:
        """Save configuration to JSON file
        
        Args:
            json_file: Path to save JSON configuration file
        """
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Save to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {json_file}")


class EEGFoundationDownstreamClassifier(PreTrainedModel):
    """EEG Foundation Downstream Classifier Model
    
    Main model for EEG downstream classification tasks
    """
    
    config_class = EEGFoundationDownstreamClassifierConfigHF
    base_model_prefix = "eeg_foundation_downstream_classifier"
    
    def __init__(
        self, 
        config: Union[EEGFoundationDownstreamClassifierConfigHF, EEGFoundationDownstreamClassifierConfig],
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize EEGFoundationDownstreamClassifier model
        
        Args:
            config: Configuration object
            model_path: Path to pretrained model weights (.pth file). 
                       If None, random initialization is used.
            config_path: Path to configuration JSON file. Required for loading pretrained models.
        """
        # Convert dataclass to HF config if needed
        if isinstance(config, EEGFoundationDownstreamClassifierConfig):
            config = EEGFoundationDownstreamClassifierConfigHF.from_dataclass(config)
        
        super().__init__(config)
        
        # Store configuration
        self.config = config
        
        # Store paths
        self.model_path = model_path
        self.config_path = config_path
        
        # If config_path is provided, update config
        if config_path is not None:
            self.config.config_path = config_path
        if model_path is not None:
            self.config.pretrained_model_path = model_path
        
        # If config_path is None but we have model_path, we need config_path
        if model_path is not None and config_path is None:
            raise ValueError(
                "config_path must be provided when model_path is specified. "
                "Please provide the path to the configuration JSON file."
            )
        
        # Calculate derived parameters
        self.patch_num = config.patch_num
        
        # Patch padding layer
        self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
        
        # Input projection layer
        self.in_layer = nn.Linear(config.patch_size, config.d_model)
        
        # Attention mechanism
        self.basic_attn = MultiHeadAttention(d_model=config.d_model)
        
        # Embedding projection
        self.projection_embedding = nn.Linear(config.embedding_dim, config.projection_embedding_dim)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(config.classification_dropout),
            nn.Linear(config.d_model * self.patch_num * config.num_channel, config.classification_hidden_dim),
            nn.BatchNorm1d(config.classification_hidden_dim),
            nn.GELU(),
            nn.Linear(config.classification_hidden_dim, config.num_classes)
        )
        
        # Initialize weights
        self.post_init()
        
        # Load pretrained weights if model_path is provided
        if model_path is not None and os.path.exists(model_path):
            self.load_pretrained_weights(model_path)
        elif model_path is not None and not os.path.exists(model_path):
            warnings.warn(f"Model path {model_path} does not exist. Using random initialization.")
    
    def load_pretrained_weights(self, model_path: str) -> None:
        """Load pretrained weights from a file
        
        Args:
            model_path: Path to the pretrained model weights file (.pth)
        """
        print(f"Loading pretrained weights from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        try:
            # Load state dict
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                state_dict = torch.load(model_path, map_location='cpu')
            else:
                # Try to load as a PyTorch model
                state_dict = torch.load(model_path, map_location='cpu')
            
            # Check if state_dict contains 'state_dict' key (common in PyTorch Lightning models)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                print("  - Detected PyTorch Lightning checkpoint format")
            
            # Remove 'model.' prefix if present (common in some saved models)
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                print("  - Removed 'model.' prefix from state_dict keys")
            
            # Load the state dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"  - Missing keys in state dict: {len(missing_keys)} keys")
                if len(missing_keys) < 10:  # Only print all if not too many
                    for key in missing_keys:
                        print(f"    - {key}")
            else:
                print("  - All keys matched successfully")
            
            if unexpected_keys:
                print(f"  - Unexpected keys in state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) < 10:  # Only print all if not too many
                    for key in unexpected_keys:
                        print(f"    - {key}")
            
            print(f"✓ Successfully loaded pretrained weights")
            
        except Exception as e:
            raise RuntimeError(f"Error loading pretrained weights from {model_path}: {e}")
    
    def norm(self, x: torch.Tensor, dim: int = 0, 
             means: Optional[torch.Tensor] = None, 
             stdev: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple]:
        """Normalize input tensor
        
        Args:
            x: Input tensor
            dim: Dimension to normalize over
            means: Precomputed means (optional)
            stdev: Precomputed standard deviations (optional)
            
        Returns:
            Normalized tensor or (normalized tensor, means, stdev)
        """
        if means is not None:  
            return x * stdev + means
        else: 
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5).detach() 
            x /= stdev
            return x, means, stdev 
    
    def rms_norm(self, x: torch.Tensor, dim: int = 0, 
                 scale: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple]:
        """RMS normalization
        
        Args:
            x: Input tensor
            dim: Dimension to normalize over
            scale: Precomputed scale (optional)
            
        Returns:
            Normalized tensor or (normalized tensor, scale)
        """
        if scale is not None:
            return x * scale
        else:
            # Compute root mean square
            rms = torch.sqrt(torch.mean(x.pow(2), dim=dim, keepdim=True) + 1e-5).detach()
            x = x / rms
            return x, rms
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        embedding_data: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attn_scores: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for EEGFoundationDownstreamClassifier model
        
        Args:
            input_ids: Input EEG signals (already preprocessed)
            attention_mask: Attention mask (not used, for compatibility)
            token_type_ids: Token type IDs (not used, for compatibility)
            embedding_data: Precomputed embeddings
            labels: Labels for classification
            return_attn_scores: Whether to return attention scores
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure input_ids are provided
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        # Ensure embedding_data are provided
        if embedding_data is None:
            raise ValueError("embedding_data must be provided")
        
        B, C = input_ids.size(0), input_ids.size(1)
        
        # Ensure input is float32
        x = input_ids.float()
        
        # 1. Input normalization
        if self.config.rms_norm:
            x, _ = self.rms_norm(x, dim=2)  
        else:
            x, _, _ = self.norm(x, dim=2)  
        
        # 2. Patch processing
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.config.patch_size, step=self.config.stride)
        x = self.in_layer(x)
        
        # 3. Add embeddings
        embedding_data = self.projection_embedding(embedding_data)  # (B, embedding_dim) -> (B, d_model)
        embedding_data = embedding_data.unsqueeze(1).unsqueeze(1)  # (B, d_model) -> (B, 1, 1, d_model)
        x = x + embedding_data  # Broadcast to (B, C, patch_num, d_model)
        
        x = rearrange(x, 'b c m l -> (b c) m l')
        if return_attn_scores:
            x, attn_weights, attn_scores = self.basic_attn(x, x, x, return_scores=True)
        else:
            x, attn_weights = self.basic_attn(x, x, x)
        x = rearrange(x, '(b c) m l -> b (c m l)', b=B, c=C)
        
        # 5. Classification head
        logits = self.classification_head(x)
        
        # 6. Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            if return_attn_scores:
                output = output + (attn_scores,)
            return output
        
        output_dict = {
            "logits": logits,
            "loss": loss,
        }
        
        if return_attn_scores:
            output_dict["attention_scores"] = attn_scores
        
        return output_dict
    
    def save_pretrained(
        self,
        save_directory: str,
        save_config: bool = True,
        save_weights: bool = True,
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin"
    ) -> None:
        """Save model and configuration
        
        Args:
            save_directory: Directory to save model
            save_config: Whether to save configuration
            save_weights: Whether to save model weights
            config_filename: Name of config file
            model_filename: Name of model weights file
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        if save_config:
            config_path = os.path.join(save_directory, config_filename)
            self.config.save_to_json(config_path)
        
        # Save model weights
        if save_weights:
            model_path = os.path.join(save_directory, model_filename)
            torch.save(self.state_dict(), model_path)
            print(f"Model weights saved to: {model_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "seq_len": self.config.seq_len,
            "patch_size": self.config.patch_size,
            "stride": self.config.stride,
            "d_model": self.config.d_model,
            "num_classes": self.config.num_classes,
            "num_channel": self.config.num_channel,
            "patch_num": self.patch_num,
            "rms_norm": self.config.rms_norm,
            "pretrained_model_path": self.config.pretrained_model_path,
            "config_path": self.config.config_path
        }
    
    def get_parameter_stats(self) -> Dict[str, Any]:
        """Get detailed parameter statistics
        
        Returns:
            Dictionary with parameter statistics
        """
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024**2)  # Assuming float32
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> 'EEGFoundationDownstreamClassifier':
        """Create model from pretrained weights
        
        Args:
            model_path: Path to pretrained model weights (.pth file). 
                       If None, random initialization is used.
            config_path: Path to configuration JSON file. Required for loading pretrained models.
            **kwargs: Additional arguments passed to the constructor
            
        Returns:
            EEGFoundationDownstreamClassifier model
        """
        # Validate inputs
        if config_path is None:
            raise ValueError("config_path must be provided. Please provide the path to the configuration JSON file.")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config = EEGFoundationDownstreamClassifierConfigHF.from_json_file(config_path)
        
        # Update config with paths
        config.pretrained_model_path = model_path
        config.config_path = config_path
        
        # Create model instance
        model = cls(config=config, model_path=model_path, config_path=config_path, **kwargs)
        
        return model


# Register the model with Auto classes
AutoConfig.register("eeg-foundation-downstream-classifier", EEGFoundationDownstreamClassifierConfigHF)
AutoModel.register(EEGFoundationDownstreamClassifierConfigHF, EEGFoundationDownstreamClassifier)


def load_downstream_config_from_json(config_path: str) -> EEGFoundationDownstreamClassifierConfigHF:
    """Load configuration from JSON file
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Configuration object
    """
    config = EEGFoundationDownstreamClassifierConfigHF.from_json_file(config_path)
    print(f"Loaded config from {config_path}")
    return config


def load_downstream_model(
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None
) -> EEGFoundationDownstreamClassifier:
    """Load downstream model with optional pretrained weights
    
    Args:
        model_path: Path to model weights file (.pth). If None, random initialization is used.
        config_path: Path to configuration JSON file. Required.
        
    Returns:
        Loaded model with pretrained weights (if model_path is provided) or random initialization
        
    Raises:
        ValueError: If config_path is None
        FileNotFoundError: If config_path or model_path doesn't exist
    """
    if config_path is None:
        raise ValueError("config_path must be provided. Please provide the path to the configuration JSON file.")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if model_path is None:
        print("No model_path provided. Initializing with random weights.")
        config = load_downstream_config_from_json(config_path)
        model = EEGFoundationDownstreamClassifier(config)
    else:
        # Check if model weights file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        print(f"Loading configuration from: {config_path}")
        print(f"Loading weights from: {model_path}")
        
        # Load configuration
        config = load_downstream_config_from_json(config_path)
        
        # Create model
        model = EEGFoundationDownstreamClassifier.from_pretrained(model_path=model_path, config_path=config_path)
    
    return model


def get_default_downstream_config() -> EEGFoundationDownstreamClassifierConfig:
    """Get default configuration for EEG downstream classifier
    
    Returns:
        Default configuration
    """
    return EEGFoundationDownstreamClassifierConfig()


def create_downstream_classifier(
    seq_len: int = 2000,
    patch_size: int = 150,
    stride: int = 1,
    d_model: int = 512,
    num_classes: int = 2,
    num_channel: int = 20,
    rms_norm: bool = False,
    embedding_dim: int = 512,
    projection_embedding_dim: int = 512,
    classification_dropout: float = 0.5,
    classification_hidden_dim: int = 512,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> EEGFoundationDownstreamClassifier:
    """Create EEG downstream classifier with specified parameters
    
    Args:
        seq_len: Sequence length
        patch_size: Patch size
        stride: Stride for patch extraction
        d_model: Model dimension
        num_classes: Number of classes
        num_channel: Number of EEG channels
        rms_norm: Whether to use RMS normalization
        embedding_dim: Embedding dimension
        projection_embedding_dim: Projection embedding dimension
        classification_dropout: Dropout for classification head
        classification_hidden_dim: Hidden dimension for classification head
        learning_rate: Learning rate
        weight_decay: Weight decay
        model_path: Path to pretrained model weights (.pth). If None, random initialization.
        config_path: Path to configuration JSON file. Required if model_path is provided.
        
    Returns:
        Initialized EEGFoundationDownstreamClassifier model
    """
    # Validate inputs
    if model_path is not None and config_path is None:
        raise ValueError(
            "config_path must be provided when model_path is specified. "
            "Please provide the path to the configuration JSON file."
        )
    
    # Create dataclass configuration
    config = EEGFoundationDownstreamClassifierConfig(
        seq_len=seq_len,
        patch_size=patch_size,
        stride=stride,
        d_model=d_model,
        num_classes=num_classes,
        num_channel=num_channel,
        rms_norm=rms_norm,
        embedding_dim=embedding_dim,
        projection_embedding_dim=projection_embedding_dim,
        classification_dropout=classification_dropout,
        classification_hidden_dim=classification_hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pretrained_model_path=model_path,
        config_path=config_path
    )
    
    # Create model
    if model_path is not None and config_path is not None:
        # Load pretrained model
        model = EEGFoundationDownstreamClassifier.from_pretrained(
            model_path=model_path, 
            config_path=config_path
        )
    else:
        # Create new model
        model = EEGFoundationDownstreamClassifier(config)
    
    return model


def create_model_from_paths(
    model_weights_path: str,
    config_path: str,
    device: str = "cpu",
    **kwargs
) -> EEGFoundationDownstreamClassifier:
    """Create model from specified weights and config files
    
    Args:
        model_weights_path: Path to model weights file (.pth)
        config_path: Path to configuration JSON file
        device: Device to load model on ("cpu" or "cuda")
        **kwargs: Additional arguments passed to the model
        
    Returns:
        Loaded model with pretrained weights
        
    Raises:
        FileNotFoundError: If config_path or model_weights_path doesn't exist
    """
    print(f"Creating EEGFoundationDownstreamClassifier from files:")
    print(f"  - Model weights: {model_weights_path}")
    print(f"  - Config file: {config_path}")
    print(f"  - Device: {device}")
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")
    
    # Load configuration
    config = load_downstream_config_from_json(config_path)
    
    # Create model
    model = EEGFoundationDownstreamClassifier.from_pretrained(
        model_path=model_weights_path,
        config_path=config_path,
        **kwargs
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model successfully created and moved to {device}")
    
    return model


# Example config.json for downstream task
EEG_FOUNDATION_DOWNSTREAM_CONFIG_EXAMPLE = {
    "seq_len": 2000,
    "patch_size": 150,
    "stride": 100,
    "d_model": 512,
    "num_classes": 2,
    "num_channel": 20,
    "rms_norm": False,
    "embedding_dim": 512,
    "projection_embedding_dim": 512,
    "classification_dropout": 0.5,
    "classification_hidden_dim": 512,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "model_type": "eeg-foundation-downstream-classifier"
}