# zFastVS - Zelixir's Fast Virtual Screening Toolkit
zFastVS is a deep learning-based virtual screening toolkit designed for ultra-fast prediction of small molecule binding affinity to target proteins.

## Overview
FastVS combines molecular docking and machine learning approaches to provide a complete virtual screening pipeline:

- Molecular Docking: Uses AutoDock Vina and SFCT for molecular docking

- Feature Extraction: Supports multiple molecular fingerprints (Morgan, NYAN)

- Deep Learnin and machine learning: Implements various ML models (Random Forest, AdaBoost, MLP) for binding score  (mainly vina score) prediction

- High Performance: Optimized for batch processing and cluster computing with SLURM support

## Features
- Multi-modal Fingerprints: Morgan fingerprints (512-bit) and NYAN fingerprints for comprehensive molecular representation

- Flexible Docking: Supports both Vina and SFCT scoring functions

- Multiple ML Models: Random Forest, AdaBoost, and Neural Network models

- Cluster Ready: Built-in SLURM support for high-throughput screening

- Easy Integration: Simple API for docking, feature extraction, and model training

## Installation
Prerequisites
```
Python 3.7+
RDKit
scikit-learn
NumPy
Open Babel
```

## Dependencies
bash
```pip install rdkit scikit-learn numpy```

The following two core tools should be installed:
- OnionNet-SFCT (https://github.com/zhenglz/OnionNet-SFCT.git)
- NYAN (https://github.com/Chokyotager/NotYetAnotherNightshade.git)


## Quick Start
### 1. Prepare Input Data

```
from fastvs.core.io import load_smiles_file
# Load SMILES file
smiles_data = load_smiles_file("molecules.smi")
```

### 2. Run Molecular Docking

```
from fastvs.core.docking import smiles_docking
from fastvs.core.utils import load_configs

configs = load_configs("configs.json")

# Run docking
results = smiles_docking(
    smiles_dict=smiles_data,
    output_dpath="output",
    receptor_fpath="receptor.pdbqt",
    pocket_center=[x, y, z],
    pocket_size=15,
    configs=configs
)
```

### 3. Extract Molecular Features
```
from fastvs.core.fingerprint import FeaturizeMolecules

featurizer = FeaturizeMolecules(
    molecules_dict=results,
    fp_types=['morgan', 'nyan'],
    configs=configs
)

features = featurizer.get_fp()
```

4. Train ML Model
```
from fastvs.core.training import BindingScoreTrainer

trainer = BindingScoreTrainer(
    molecules_dict_train=train_data,
    features_train=train_features,
    molecules_dict_test=test_data,
    features_test=test_features,
    output_dpath='models',
    method='RF'
)

trainer.train()
```

## Project Structure
```
fastvs/
├── core/
│   ├── data.py           # Molecule data structures
│   ├── docking.py        # Docking and scoring
│   ├── fingerprint.py    # Molecular fingerprints
│   ├── io.py            # Input/output operations
│   ├── models.py        # ML models
│   ├── training.py      # Model training
│   ├── slurm.py         # SLURM cluster support
│   └── utils.py         # Utility functions
├── data/
│   └── configs.json     # Configuration file
└── bin/
    └── *.sh            # Helper scripts

```

## Configuration
Create a configs.json file:

json
{
    "sfct": {
        "script": "/path/to/sfct_script.py"
    },
    "nyan": {
        "python": "/path/to/python",
        "script": "/path/to/nyan_script.py"
    }
}


## Usage Examples
Batch Docking on Cluster
```
from fastvs.core.docking import smiles_docking
from fastvs.core.slurm import run_tasks

# Large-scale docking with SLURM
results = smiles_docking(
    smiles_dict=large_smiles_set,
    output_dpath="large_scale_output",
    receptor_fpath="target.pdbqt",
    pocket_center=[x, y, z],
    configs=configs,
    remove_temp=False
)
```

## Custom Model Training
```
from fastvs.core.models import MLPModel

# Custom neural network model
model = MLPModel(
    X=train_features,
    y=train_scores,
    Xtest=test_features,
    ytest=test_scores,
    out_dpath='custom_models',
    hidden_layers=[1024, 512, 256]
)
model.train()
accuracy = model.evaluate()
```

## Citation
To be updated

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Contact
For questions and support, please contact: [your.email@example.com]