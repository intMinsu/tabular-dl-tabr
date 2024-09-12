# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<
# lib/data.py
from dataclasses import dataclass, replace
from typing import Any, Generic, Iterable, Optional, TypeVar, Union, cast
from torch import Tensor

from lib import KWArgs
import argparse
import os
import torch
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional
import lib
from bin.tabr import Model 
from lib.data import are_valid_predictions
import numpy as np
from torch.utils.data import DataLoader

# Assuming you have a custom PyTorch dataset class defined elsewhere in your code
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load data from file or other sources
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x 

@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]

# Define inference function
@torch.inference_mode()
def inference(model, dataset, parts: list[str], C, show_metric: bool):
    model.eval()
    predictions = {}

    part = 'test'
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)  
    x_features = next(iter(data_loader))

    eval_batch_size = 128
    part_predictions = []

    X_train_array = np.load('/home/minsu/sait-competition-2024/submodules/tabr/data/sait/X_num_train.npy')
    Y_train_array = np.load('/home/minsu/sait-competition-2024/submodules/tabr/data/sait/Y_train.npy')

    X_test_array = np.load('/home/minsu/sait-competition-2024/submodules/tabr/data/sait/X_num_test.npy')
    Y_test_array = np.load('/home/minsu/sait-competition-2024/submodules/tabr/data/sait/Y_test.npy')

    for n, batch_x in enumerate(data_loader):
        batch_x_dict = {'num': batch_x}

        # RETREIVAL
        candidate_x, candidate_y = (
            {
                'num': torch.zeros(1, x_features.shape[1]),
            },
            torch.zeros(1)
        )

        output = model(x_=batch_x_dict, y=None,
                        candidate_x_=candidate_x, candidate_y=candidate_y,
                        context_size=C.context_size, is_train=False)

        print(f"[{n} Iter] {batch_x.shape} {output.shape}")
        part_predictions.append(output.cpu().numpy())  # Convert to numpy
        
    #Concatenate predictions for this part
    predictions[part] = np.concatenate(part_predictions, axis=0)
    
    # Show metrics if required
    if show_metric:
        raise NotImplementedError("Add ELBO in DDOM proposal")
        # metrics = (
        #     dataset.calculate_metrics(predictions, report['prediction_type'])
        #     if are_valid_predictions(predictions)
        #     else {x: {'score': -999999.0} for x in predictions}
        # )
    else:
        metrics = None
    
    return predictions, metrics


def evaluate(model,
            dataset,
            parts: list[str],
            C,
            checkpoint_path: Path, 
            show_metric: bool = False):

    predictions, metrics = inference(model, dataset, parts, C, show_metric)

    if show_metric:
        print(metrics)
    else:
        print("show_metric = False")

    # Save predictions somewhere (e.g., in a file or a database)
    save_path = checkpoint_path / 'our-prediction'
    np.save(save_path, predictions['test'])
    
    print(f"Predictions saved to {save_path}")
    return predictions

def main(path: Path, function: Optional[str] = None):

    evaluation_dir = lib.get_path(path)
    assert evaluation_dir.name.endswith('-evaluation')

    # TODO : ADD ENSENBLE
    n_ensembles = 3
    ensemble_size = 5

    total_checkpoints_size = n_ensembles * ensemble_size

    #dataset = CustomDataset('data/sait-ours/X_test.npy')
    dataset = CustomDataset('data/sait/X_num_test.npy')

    template_config = lib.load_config(evaluation_dir / '0.toml')
    config = deepcopy(template_config)
    C = lib.make_config(Config, config)

    model = Model(
        n_num_features=11,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
        **C.model,
    )
    
    # Load model weights from checkpoint
    checkpoint_path = evaluation_dir / '0'
    model.load_state_dict(lib.load_checkpoint(checkpoint_path)['model'])
    print(f"{str(checkpoint_path)} successfully loaded")

    predictions = evaluate(model, dataset, 
                        ['our-test'],
                        C=C,
                        checkpoint_path=checkpoint_path, 
                        show_metric=False)

    print(predictions)

    # Implement later
    single_outputs = []
    for ensemble_id in range(total_checkpoints_size):
        single_outputs.append(str(evaluation_dir) + '/' + str(ensemble_id))

    print(f"evaluation_dir : {evaluation_dir}")

if __name__ == "__main__":
    lib.configure_libraries()
    lib.run_cli(main)
    
    
    