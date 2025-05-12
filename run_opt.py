import argparse
import datetime
import yaml
import copy
from pathlib import Path
import optuna
from exp.exp_main import Exp_Main  # Ensure Exp_Main uses the provided config for training/testing

def get_all_pair_ids(dataset_name):
    """
    Return a list of all available sub-dataset IDs for a given dataset.
    Replace this stub with your actual logic.
    """
    # For example, for dataset PDE_KS, return all available IDs.
    return list(range(1, 6))  # e.g. [1,2,3,4,5]

def parse_pair_ids(pair_id_field, dataset_name):
    """
    Parse the pair_id field from configuration.
    
    Input formats:
      - Single integer -> [integer]
      - List of integers -> as is
      - Range string, e.g. "1-3" -> [1,2,3]
      - Omitted or "all" -> all available sub-dataset IDs
    """
    if pair_id_field is None or str(pair_id_field).lower() == 'all':
        return get_all_pair_ids(dataset_name)
    if isinstance(pair_id_field, int):
        return [pair_id_field]
    if isinstance(pair_id_field, list):
        return pair_id_field
    if isinstance(pair_id_field, str) and '-' in pair_id_field:
        parts = pair_id_field.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return list(range(start, end+1))
            except ValueError:
                raise ValueError(f"Invalid range specification for pair_id: {pair_id_field}")
    try:
        return [int(pair_id_field)]
    except Exception as e:
        raise ValueError(f"Unable to parse pair_id from: {pair_id_field}") from e

def objective(trial, config_template, pair_results_dir):
    """
    Objective function for hyperparameter tuning.
    Create a deep copy of the configuration, update the model parameters based on trial suggestions,
    and run a training run via Exp_Main. Return the metric that Optuna will optimize.
    
    (You must adapt this function according to your experiment logic.)
    """
    import copy
    config = copy.deepcopy(config_template)
    # Example: update model hyperparameters (adjust keys and suggestion ranges accordingly)
    config['model']['parameters']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    config['model']['parameters']['hidden_size'] = trial.suggest_int('hidden_size', 32, 512, step=32)
    # Ensure there is a training section if needed:
    if 'training' not in config:
        config['training'] = {}
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Initialize experiment with updated configuration.
    exp = Exp_Main(config)
    
    # Use trial number to create a unique setting name.
    setting = f"trial_{trial.number}"
    print(f">>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train(setting)
    print(f">>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    metrics = exp.test(setting)
    
    # Save the trial configuration and metrics for reference.
    trial_dir = pair_results_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    with open(trial_dir / 'evaluation_results.yaml', 'w') as f:
        yaml.dump(metrics, f)
    
    # Return the metric you wish to minimize. Adjust key as needed.
    return metrics.get('validation_loss', 0)

def main(config_path):
    # Load configuration template
    with open(config_path, 'r') as f:
        config_template = yaml.safe_load(f)

    # Parse dataset name and pair_id field
    dataset_config = config_template.get('dataset', {})
    dataset_name = dataset_config.get('name', None)
    if not dataset_name:
        raise ValueError("No dataset name specified in the configuration.")

    pair_id_field = dataset_config.get('pair_id', None)
    pair_ids = parse_pair_ids(pair_id_field, dataset_name)
    if not pair_ids:
        raise ValueError("No valid pair_id found in the configuration.")

    # Create overall results directory for this batch of experiments
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    overall_results_dir = Path(f"results/hyperparameter_tuning/{batch_id}")
    overall_results_dir.mkdir(parents=True, exist_ok=True)

    # Save the original configuration for record-keeping
    with open(overall_results_dir / 'config_template.yaml', 'w') as f:
        yaml.dump(config_template, f)

    # Iterate through each pair_id sequentially
    for pid in pair_ids:
        print(f"\nRunning hyperparameter optimization for pair_id: {pid}")
        # Create a deep copy of the configuration for the current pair
        config = copy.deepcopy(config_template)
        config['dataset']['pair_id'] = pid

        # Create a sub-directory for this pair_id's results
        pair_results_dir = overall_results_dir / f"pair_{pid}"
        pair_results_dir.mkdir(parents=True, exist_ok=True)

        # Run hyperparameter optimization for this configuration
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, config, pair_results_dir), n_trials=20)

        best_trial = study.best_trial
        print(f"Best trial for pair {pid}: {best_trial.number}")
        print(f"Best parameters for pair {pid}: {best_trial.params}")
        with open(pair_results_dir / 'best_trial.yaml', 'w') as f:
            yaml.dump({'number': best_trial.number, 'params': best_trial.params, 'value': best_trial.value}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)