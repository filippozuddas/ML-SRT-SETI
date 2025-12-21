"""
Configuration loading and management utilities.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class DataConfig:
    """Data generation configuration."""
    frame_height: int = 16
    frame_width: int = 512
    original_width: int = 4096
    downscale_factor: int = 8
    num_observations: int = 6
    df: float = 2.7939677238464355
    dt: float = 18.25361108
    fch1: float = 6095.214842353016


@dataclass
class SignalConfig:
    """Signal injection configuration."""
    snr_min: float = 10
    snr_max: float = 80
    snr_base: float = 20
    snr_range: float = 40
    drift_rate_factor: float = 1.0
    width_base: float = 50
    width_drift_factor: float = 18
    noise_mean: float = 58348559
    noise_type: str = 'chi2'


@dataclass  
class ModelConfig:
    """VAE model configuration."""
    latent_dim: int = 8
    dense_units: int = 512
    kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    l1_weight: float = 0.001
    l2_weight: float = 0.01
    alpha: float = 10
    beta: float = 0.5
    gamma: float = 0


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1000
    validation_batch_size: int = 2000
    epochs_per_cycle: int = 100
    num_cycles: int = 20
    num_samples_train: int = 6000
    num_samples_val: int = 1000
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    checkpoint_dir: str = 'checkpoints'
    save_frequency: int = 1


@dataclass
class ClassifierConfig:
    """Random Forest classifier configuration."""
    n_estimators: int = 1000
    max_features: str = 'sqrt'
    bootstrap: bool = True
    n_jobs: int = -1
    num_samples: int = 4000


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    num_gpus: int = 2
    memory_limit_per_gpu: int = 22000
    mixed_precision: bool = True
    prefetch_buffer: int = 4
    num_parallel_calls: int = -1


@dataclass
class SearchConfig:
    """Search/inference configuration."""
    probability_threshold: float = 0.5
    batch_size: int = 5000
    sliding_window: bool = True


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file. 
                     If None, uses default configuration.
    
    Returns:
        Config object with all settings.
    """
    config = Config()
    
    if config_path is None:
        # Look for default config
        default_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
        if default_path.exists():
            config_path = str(default_path)
        else:
            return config
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Update each section
    if 'data' in yaml_config:
        config.data = DataConfig(**yaml_config['data'])
    
    if 'signal' in yaml_config:
        config.signal = SignalConfig(**yaml_config['signal'])
    
    if 'model' in yaml_config:
        model_dict = yaml_config['model']
        # Handle kernel_size which might be a list
        config.model = ModelConfig(**model_dict)
    
    if 'training' in yaml_config:
        config.training = TrainingConfig(**yaml_config['training'])
    
    if 'classifier' in yaml_config:
        config.classifier = ClassifierConfig(**yaml_config['classifier'])
    
    if 'hardware' in yaml_config:
        config.hardware = HardwareConfig(**yaml_config['hardware'])
    
    if 'search' in yaml_config:
        config.search = SearchConfig(**yaml_config['search'])
    
    return config


def setup_gpu(config: Config) -> None:
    """
    Configure GPU settings based on configuration.
    
    Args:
        config: Configuration object with hardware settings.
    """
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for i, gpu in enumerate(gpus[:config.hardware.num_gpus]):
                tf.config.experimental.set_memory_growth(gpu, True)
                if config.hardware.memory_limit_per_gpu > 0:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=config.hardware.memory_limit_per_gpu
                        )]
                    )
            print(f"Configured {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Enable mixed precision if configured
    if config.hardware.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (FP16) enabled")


def get_strategy(config: Config):
    """
    Get TensorFlow distribution strategy based on configuration.
    
    Args:
        config: Configuration object.
    
    Returns:
        TensorFlow distribution strategy.
    """
    import tensorflow as tf
    
    if config.hardware.num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
    else:
        strategy = tf.distribute.get_strategy()
    
    return strategy
