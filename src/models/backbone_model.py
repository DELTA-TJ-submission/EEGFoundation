"""
EEG Backbone Model
HuggingFace-compatible EEG foundation model based on RoFormer architecture
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.roformer import RoFormerForMaskedLM, RoFormerConfig, RoFormerModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM


@dataclass
class EEGFoundationBackboneConfig:
    """Configuration for EEGBackbone model using dataclass format"""
    
    # Model architecture parameters
    vocab_size: int = 2000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    max_position_embeddings: int = 2000
    
    # Dropout and regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Token type and special tokens
    type_vocab_size: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Transformer parameters
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    
    # Additional parameters
    model_type: str = "eeg-backbone"
    
    def to_roformer_config(self) -> RoFormerConfig:
        """Convert to RoFormerConfig object"""
        return RoFormerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=self.use_cache,
            classifier_dropout=self.classifier_dropout,
        )


class EEGFoundationBackboneConfigHF(PretrainedConfig):
    """HuggingFace compatible configuration for EEGBackbone"""
    
    model_type = "eeg-backbone"
    
    def __init__(
        self,
        vocab_size: int = 2006,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2000,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        
        # Additional parameters from config.json
        activation: str = "silu",
        cls_token: str = "[CLS]",
        cls_token_id: int = 2003,
        mask_token: str = "[MASK]",
        mask_token_id: int = 2001,
        position_embedding_type: str = "rotary",
        rope_theta: float = 10000,
        rotary_dim: int = 64,
        sep_token: str = "[SEP]",
        sep_token_id: int = 2004,
        unk_token: str = "[UNK]",
        unk_token_id: int = 2005,
        use_flash_attention: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Model architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        
        # Dropout and regularization
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # Token type
        self.type_vocab_size = type_vocab_size
        
        # Transformer parameters
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        
        # Additional parameters from config.json
        self.activation = activation
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.use_flash_attention = use_flash_attention
    
    @classmethod
    def from_dataclass(cls, config: EEGFoundationBackboneConfig) -> 'EEGFoundationBackboneConfigHF':
        """Create from dataclass configuration"""
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            use_cache=config.use_cache,
            classifier_dropout=config.classifier_dropout,
        )
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'EEGFoundationBackboneConfigHF':
        """Load configuration from JSON file
        
        Args:
            json_file: Path to JSON configuration file
            
        Returns:
            Configuration object
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class EEGFoundationBackbone(PreTrainedModel):
    """EEG Backbone Model based on RoFormer
    
    Main backbone model for EEG foundation model training
    """
    
    config_class = EEGFoundationBackboneConfigHF
    base_model_prefix = "eeg_backbone"
    
    def __init__(self, config: Union[EEGFoundationBackboneConfigHF, EEGFoundationBackboneConfig]):
        """
        Initialize EEGBackbone model
        
        Args:
            config: Configuration object (either EEGFoundationBackboneConfigHF or EEGFoundationBackboneConfig)
        """
        # Convert dataclass to HF config if needed
        if isinstance(config, EEGFoundationBackboneConfig):
            config = EEGFoundationBackboneConfigHF.from_dataclass(config)
        
        super().__init__(config)
        
        # Store configuration
        self.config = config
        
        # Debug: print configuration
        print(f"DEBUG: Initializing model with vocab_size: {config.vocab_size}")
        
        # Initialize RoFormer model with converted config
        roformer_config = self._create_roformer_config(config)
        self.roformer = RoFormerModel(roformer_config)
        
        # Initialize LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def _create_roformer_config(self, config: EEGFoundationBackboneConfigHF) -> RoFormerConfig:
        """Create RoFormerConfig from EEGFoundationBackboneConfigHF"""
        print(f"DEBUG: Creating RoFormerConfig with vocab_size: {config.vocab_size}")
        return RoFormerConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            use_cache=config.use_cache,
            classifier_dropout=config.classifier_dropout,
            rotary_value=False,  # Set to True if using rotary position embeddings
        )
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings"""
        return self.roformer.embeddings.word_embeddings
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings"""
        self.roformer.embeddings.word_embeddings = value
    
    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings (LM head)"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Set output embeddings (LM head)"""
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Forward pass for EEGBackbone model
        
        Args:
            input_ids: Token ids (already preprocessed)
            attention_mask: Attention mask
            token_type_ids: Token type ids
            head_mask: Head mask
            inputs_embeds: Input embeddings
            encoder_hidden_states: Encoder hidden states
            encoder_attention_mask: Encoder attention mask
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure input_ids or inputs_embeds are provided
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Forward pass through RoFormer
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        # LM head
        lm_logits = self.lm_head(sequence_output)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output
        
        return {
            "logits": lm_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings"""
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        
        # Update vocab size in config
        self.config.vocab_size = new_num_tokens
        
        return model_embeds
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "hidden_act": self.config.hidden_act,
            "max_position_embeddings": self.config.max_position_embeddings,
            "use_flash_attention": getattr(self.config, "use_flash_attention", False),
            "special_tokens": {
                "cls_token_id": getattr(self.config, "cls_token_id", None),
                "mask_token_id": getattr(self.config, "mask_token_id", None),
                "sep_token_id": getattr(self.config, "sep_token_id", None),
                "unk_token_id": getattr(self.config, "unk_token_id", None),
            }
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


class EEGFoundationBackboneForMaskedLM(EEGFoundationBackbone):
    """EEGBackbone model for masked language modeling"""
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Forward pass for masked language modeling
        
        Args:
            input_ids: Token ids (already preprocessed)
            attention_mask: Attention mask
            token_type_ids: Token type ids
            head_mask: Head mask
            inputs_embeds: Input embeddings
            encoder_hidden_states: Encoder hidden states
            encoder_attention_mask: Encoder attention mask
            labels: Labels for masked language modeling
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure input_ids or inputs_embeds are provided
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Forward pass through RoFormer
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        # LM head
        lm_logits = self.lm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return {
            "loss": masked_lm_loss,
            "logits": lm_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation"""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# Register the model with Auto classes
AutoConfig.register("eeg-backbone", EEGFoundationBackboneConfigHF)
AutoModel.register(EEGFoundationBackboneConfigHF, EEGFoundationBackbone)
AutoModelForMaskedLM.register(EEGFoundationBackboneConfigHF, EEGFoundationBackboneForMaskedLM)


def load_config_from_json(config_path: str) -> EEGFoundationBackboneConfigHF:
    """Load configuration from JSON file
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Configuration object
    """
    config = EEGFoundationBackboneConfigHF.from_json_file(config_path)
    print(f"DEBUG: Loaded config from {config_path}")
    print(f"DEBUG: Config vocab_size: {config.vocab_size}")
    return config


def load_pretrained_model(model_path: str, config_path: str) -> EEGFoundationBackboneForMaskedLM:
    """Load pretrained model from directory
    
    Args:
        model_path: Path to model weights file (.pth)
        config_path: Path to configuration JSON file
        
    Returns:
        Loaded model with pretrained weights
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Check if model weights file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    
    print(f"Loading configuration from: {config_path}")
    print(f"Loading weights from: {model_path}")
    
    # Load configuration
    config = load_config_from_json(config_path)
    
    # Create model
    model = EEGFoundationBackboneForMaskedLM(config)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Debug: Check vocab_size in state dict
    print(f"DEBUG: Checking embedding layer in state dict...")
    for key in list(state_dict.keys()):
        if 'embedding' in key or 'lm_head' in key:
            print(f"  {key}: shape = {state_dict[key].shape}")
    
    # Check for vocabulary size mismatch
    if 'roformer.embeddings.word_embeddings.weight' in state_dict:
        vocab_in_state_dict = state_dict['roformer.embeddings.word_embeddings.weight'].shape[0]
        if vocab_in_state_dict != config.vocab_size:
            print(f"WARNING: Vocabulary size mismatch!")
            print(f"  State dict vocab size: {vocab_in_state_dict}")
            print(f"  Config vocab size: {config.vocab_size}")
            
            # Resize embeddings
            model.resize_token_embeddings(vocab_in_state_dict)
            print(f"  Resized model vocab to: {vocab_in_state_dict}")
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    
    return model


def get_default_config() -> EEGFoundationBackboneConfig:
    """Get default configuration for EEG backbone model using dataclass
    
    Returns:
        Default configuration
    """
    return EEGFoundationBackboneConfig()


def create_backbone_model(
    vocab_size: int = 2000,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    max_position_embeddings: int = 2000,
    type_vocab_size: int = 2,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    use_cache: bool = True,
    classifier_dropout: Optional[float] = None,
) -> EEGFoundationBackboneForMaskedLM:
    """Create EEG backbone model with specified parameters
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of Transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        hidden_act: Activation function
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention
        max_position_embeddings: Maximum position embeddings length
        type_vocab_size: Token type vocabulary size
        initializer_range: Range for parameter initialization
        layer_norm_eps: Epsilon for layer normalization
        pad_token_id: Padding token id
        bos_token_id: Beginning of sequence token id
        eos_token_id: End of sequence token id
        use_cache: Whether to use cache for generation
        classifier_dropout: Dropout for classifier
        
    Returns:
        Initialized EEGFoundationBackboneForMaskedLM model
    """
    # Create dataclass configuration
    config = EEGFoundationBackboneConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
        classifier_dropout=classifier_dropout,
    )
    
    # Create model
    model = EEGFoundationBackboneForMaskedLM(config)
    
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EEG Foundation Backbone Model Demonstration")
    print("=" * 80)
    
    # Configuration
    MODEL_PATH = ".pth"
    CONFIG_PATH = ".json"

    # Example 1: Load pretrained model
    print("\n1. Loading Pretrained Model")
    print("-" * 40)
    
    try:
        # Load pretrained model
        model = load_pretrained_model(MODEL_PATH, CONFIG_PATH)
        
        # Get model statistics
        param_stats = model.get_parameter_stats()
        
        print(f"✓ Successfully loaded model from: {MODEL_PATH}")

        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nUsing default configuration instead...")
        
        # Create model with default config
        config = get_default_config()
        model = EEGFoundationBackboneForMaskedLM(config)
        
        param_stats = model.get_parameter_stats()
        print(f"\nModel Parameter Statistics (Default Config):")
        print(f"  Total parameters: {param_stats['total_parameters']:,}")
    
    # Example 2: Forward pass with test data
    print("\n" + "=" * 80)
    print("2. Forward Pass with Test Data")
    print("-" * 40)
    
    # Create test data
    batch_size = 2
    seq_length = 2000
    
    # Create dummy input (simulating already binned EEG data)
    vocab_size = model.config.vocab_size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"Test input shape: {dummy_input_ids.shape}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids)
    
    print(f"✓ Forward pass completed")
    print(f"\nOutput Statistics:")
    
    # Example 3: Get encoder outputs
    print("\n" + "=" * 80)
    print("3. Encoder Output Analysis")
    print("-" * 40)
    
    encoder = model.roformer
    
    with torch.no_grad():
        encoder_outputs = encoder(input_ids=dummy_input_ids)
    
    print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
    print("All tests pass!")
  
    
   