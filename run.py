import argparse
import datetime
import yaml
import copy
from pathlib import Path
from exp.exp_main import Exp_Main  # ensure Exp_Main provides train() and test() methods

def get_all_pair_ids(dataset_name):
    """
    Return a list of all available sub-dataset IDs for a given dataset.
    Replace this stub with your actual logic.
    """
    # Example stub – you may use actual data loading to discover available pair_ids.
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
    # If none of the above, try to convert to int and make a list
    try:
        return [int(pair_id_field)]
    except Exception as e:
        raise ValueError(f"Unable to parse pair_id from: {pair_id_field}") from e

def main(config_path):
    # Load configuration template
    with open(config_path, 'r') as f:
        config_template = yaml.safe_load(f)

    # Determine the available dataset and pair_ids
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
    overall_results_dir = Path(f"results/experiment/{batch_id}")
    overall_results_dir.mkdir(parents=True, exist_ok=True)

    # Save the original configuration for record-keeping
    with open(overall_results_dir / 'config_template.yaml', 'w') as f:
        yaml.dump(config_template, f)

    # Iterate through each pair ID sequentially
    for pid in pair_ids:
        print(f"Running experiment for pair_id: {pid}")
        # Create a fresh copy of the configuration for each pair
        config = copy.deepcopy(config_template)
        config['dataset']['pair_id'] = pid

        # Create a sub-directory for this pair_id's results
        pair_results_dir = overall_results_dir / f"pair_{pid}"
        pair_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize and run the experiment
        exp = Exp_Main(config)
        print(f">>> Training for pair_{pid} ...")
        exp.train(f"pair_{pid}")  # Essential training call
        print(f">>> Testing for pair_{pid} ...")
        exp.test(f"pair_{pid}")   # Essential testing call

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)