# SRFinder Project Summary

## Overview
SRFinder is a machine learning project focused on Signal Region Finding in particle physics, specifically dealing with jet physics data. The project implements a sophisticated multi-step training pipeline using PyTorch and custom neural network architectures.

## Project Structure
```
.
├── configs/            # YAML configuration files
│   ├── base_fvt_with_repr_norm.yml
│   ├── better_fvt_training.yml
│   ├── smeared_fvt_training.yml
│   ├── CR_fvt_training_original_features.yml
│   └── mi_test.yml
├── data/              # Data storage (gitignored)
├── run_files/         # Execution scripts and logs
├── tb_logs/           # TensorBoard logs
├── notebooks/         # Jupyter notebooks
└── src/              # Source code files
```

## Core Components

### 1. Training Pipeline
The project implements a four-step training process:

1. **Base FvT Training** (`step_1_base_fvt_training.py`)
   - Initial Feature vs Transform classifier training
   - Handles basic jet physics feature processing
   - Configurable model architecture and training parameters
   - Key parameters:
     - Signal ratio: 0.0-0.02
     - Batch size: 1024
     - Learning rate: 0.01
     - Early stopping patience: configurable

2. **Smeared FvT Training** (`step_2_smeared_fvt_training.py`)
   - Applies noise/smearing to training data
   - Builds upon base FvT model
   - Supports multiple noise scales (0.5, 1.0, 2.0, 3.0, ∞)
   - Ensemble training support

3. **Control Region Training** (`step_3_define_CR_and_train_fvt.py`)
   - Defines control regions
   - Trains FvT classifier with CR information
   - Handles signal ratio variations
   - Supports multiple SR/CR size ratios

4. **MI Testing** (`step_4_mi_test.py`)
   - Mutual Information testing
   - Model evaluation and validation
   - Statistical analysis
   - Performance metrics calculation

### 2. Model Architecture

#### FvT Classifier
- Main classifier implementation
- PyTorch Lightning based
- Supports GPU acceleration
- Configurable architecture depth:
  - Encoder depth: 4
  - Decoder depth: 1
- Feature dimensions:
  - Input jet features: 4
  - Dijet features: 6
  - Quadjet features: 6
- Custom loss functions
- Comprehensive validation

#### Feature Processing
- Input jet features (dim=4)
- Dijet features (dim=6)
- Quadjet features (dim=6)
- Custom data augmentation
- Signal/background handling
- Representation normalization option

### 3. Configuration System
- YAML-based configuration
- Experiment tracking
- Hyperparameter management:
  - Model parameters
  - Training parameters
  - Data parameters
  - Optimization settings
- Multi-run support
- Ensemble configuration
- Configuration templates for different scenarios

## Execution Methods

### 1. Single Run
```bash
python run_with_config.py --config configs/your_config.yml
```

### 2. Multiple Configurations
```bash
python run_with_multiple_configs.py -c config1.yml -c config2.yml
```

### 3. Parallel Processing
```bash
python run_multiple_configs_parallel.py -c config1.yml -c config2.yml -p 4
```

### 4. Batch Processing (SLURM)
- Supports cluster execution
- Resource management:
  - CPU: 8-16 cores
  - Memory: 8GB per core
  - GPU: 1 per job
- Job monitoring
- Email notifications

## Development Environment

### Requirements
- Python 3.11+
- PyTorch
- CUDA support
- Conda environment: "coffea_torch"

### Key Dependencies
- PyTorch Lightning
- NumPy
- pandas
- YAML
- TensorBoard
- Custom physics libraries

## Data Management

### Supported Formats
- HDF5 (.h5)
- Pickle (.pkl)
- Custom data formats

### Data Processing
- Automated preprocessing
- Feature engineering
- Signal injection
- Noise modeling
- Cross-validation splits
- Dataset seeds for reproducibility

## Experiment Management

### Configuration Options
- Signal ratios: 0.0-0.02
- Noise scales: 0.5, 1.0, 2.0, 3.0, ∞
- Model architectures:
  - Encoder depth: 4
  - Decoder depth: 1
- Training parameters:
  - Batch size: 1024
  - Learning rate: 0.01
  - Max epochs: 100
- Validation settings:
  - Validation ratio: 0.33
  - Early stopping patience

### Monitoring
- TensorBoard integration
- Custom logging
- Performance metrics
- Resource utilization
- Training history tracking

## Performance Optimization
- GPU acceleration
- Parallel processing
- Memory management
- Batch size optimization
- Learning rate scheduling:
  - ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 10
  - Min LR: 0.0002

## Documentation
- Code documentation
- Configuration guides
- Execution instructions
- Performance reports
- Experiment tracking

## Future Improvements
1. Enhanced parallelization
2. Additional model architectures
3. Improved data preprocessing
4. Extended validation metrics
5. Advanced visualization tools 