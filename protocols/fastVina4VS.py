import shutil
import pandas as pd
import os, sys
import argparse
from fastvs.core.utils import run_command, make_temp_dpath, encode_string, load_configs
from fastvs.core.data import MoleculeData
from fastvs.core.io import *
from fastvs.core.slurm import run_tasks
from fastvs.core.fingerprint import FeaturizeMolecules
from fastvs.core.training import BindingScoreTrainer
from fastvs.core.docking import smiles_docking
    

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='zFastVS - Virtual Screening Pipeline')
    
    # Required arguments
    parser.add_argument('--smiles_file', type=str, required=True,
                        help='Path to input SMILES file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--receptor', type=str, required=True,
                        help='Path to receptor PDBQT file')
    
    # Optional arguments with defaults
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory path (default: output)')
    parser.add_argument('--train_batch', type=int, default=5000,
                        help='Number of molecules per training batch (default: 5000)')
    parser.add_argument('--test_size', type=int, default=5000,
                        help='Number of molecules for test set (default: 5000)')
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of training rounds (default: 20)')
    parser.add_argument('--method', type=str, default='MLP',
                        choices=['MLP', 'RF', 'Adaboost'],
                        help='Machine learning method (default: MLP)')
    
    # Pocket center coordinates (required but with example default)
    parser.add_argument('--pocket_x', type=float, required=True,
                        help='X coordinate of pocket center')
    parser.add_argument('--pocket_y', type=float, required=True,
                        help='Y coordinate of pocket center')
    parser.add_argument('--pocket_z', type=float, required=True,
                        help='Z coordinate of pocket center')
    parser.add_argument('--pocket_size', type=int, default=15,
                        help='Pocket size for docking (default: 15)')
    
    # Additional options
    parser.add_argument('--cofactor', type=str, default=None,
                        help='Path to cofactor file (optional)')
    parser.add_argument('--fp_types', nargs='+', default=['morgan', 'nyan'],
                        choices=['morgan', 'nyan'],
                        help='Fingerprint types to use (default: morgan nyan)')
    
    return parser.parse_args()


def main():
    """Main function to run the virtual screening pipeline"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'models'), exist_ok=True)
    
    # Load data and configurations
    print(f"Loading SMILES data from {args.smiles_file}")
    data = load_smiles_file(args.smiles_file)
    configs = load_configs(args.config)
    
    # Prepare pocket center
    pocket_center = [args.pocket_x, args.pocket_y, args.pocket_z]
    
    print(f"Dataset size: {len(data)} molecules")
    print(f"Pocket center: {pocket_center}")
    print(f"Pocket size: {args.pocket_size}")
    print(f"Training method: {args.method}")
    print(f"Fingerprint types: {args.fp_types}")
    
    # Split data into test and remaining
    print(f"Splitting data: {args.test_size} for testing, rest for training")
    data_test, data_remaining = split_smiles_dict_by_num(data, n=args.test_size)
    
    # Test data processing
    test_scores_path = os.path.join(args.output, 'test_scores.npy')
    if os.path.exists(test_scores_path):
        print("Loading existing test scores...")
        res_test = load_molecule_dataset(test_scores_path)
    else:
        res_test = {}
    
    print("Running docking for TEST data...")
    res_test = smiles_docking(
        data_test, 
        output_dpath=args.output,
        receptor_fpath=args.receptor, 
        pocket_center=pocket_center,
        pocket_size=args.pocket_size,
        configs=configs,
        cofactor_fpath=args.cofactor,
        molecules_dict=res_test
    )

    save_molecule_dataset(res_test, test_scores_path)
    print(f"Test data saved to {test_scores_path}")

    # Generate test features
    print("Generating test features...")
    feat = FeaturizeMolecules(res_test, configs=configs, fp_types=args.fp_types)
    feat_test = feat.get_fp(fp_data_prev={})

    # Training loop
    scores = []
    feat_train = {}
    data_train, data_remaining = split_smiles_dict_by_num(data_remaining, n=args.train_batch)
    
    for i in range(args.rounds):
        print(f"\n[INFO] =========== Training Round {i+1}/{args.rounds} ==============")
        
        # Load or create training data
        train_scores_path = os.path.join(args.output, 'train_scores.npy')
        if os.path.exists(train_scores_path):
            print("Loading existing training scores...")
            res_train = load_molecule_dataset(train_scores_path)
        else:
            res_train = {}

        print("Running docking for TRAINING data...")
        res_train = smiles_docking(
            data_train, 
            output_dpath=args.output,
            receptor_fpath=args.receptor, 
            pocket_center=pocket_center,
            pocket_size=args.pocket_size,
            configs=configs,
            cofactor_fpath=args.cofactor,
            molecules_dict=res_train
        )
        
        # Generate training features
        print("Generating training features...")
        feat = FeaturizeMolecules(res_train, configs=configs, fp_types=args.fp_types)
        feat_train = feat.get_fp(feat_train)
        save_molecule_dataset(res_train, train_scores_path)

        # Prepare next batch
        if len(data_remaining) > 0:
            data_train, data_remaining = split_smiles_dict_by_num(data_remaining, n=args.train_batch)
        else:
            print("Warning: No more data remaining for training")
            break

        # Train model
        print(f"Training {args.method} model...")
        trainer = BindingScoreTrainer(
            res_train, feat_train, res_test, 
            feat_test, 
            output_dpath=os.path.join(args.output, 'models'),
            method=args.method
        )
        trainer.train()
    
        # Record scores
        scores.append([
            i, 
            trainer.model.accuracies['mse'], 
            trainer.model.accuracies['r2'],
            trainer.model.accuracies.get('pcc', 0),
            trainer.model.accuracies.get('spc', 0)
        ])

        # Save progress
        df = pd.DataFrame(scores, columns=['idx', 'mse', 'r2', 'pcc', 'spc'])
        scores_path = os.path.join(args.output, f'{args.method.lower()}_model_scores.csv')
        df.to_csv(scores_path, header=True, index=False)
        print(f"Progress saved to {scores_path}")

    print(f"\n[INFO] Training completed! Results saved in {args.output}")


if __name__ == "__main__":
    main()