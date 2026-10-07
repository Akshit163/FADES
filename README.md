# FADES
## Configuration

All model hyperparameters, training settings, and dataset paths are managed centrally in `config.py`.

Example settings you can modify:

* Learning rate, batch size, and weight decay
* Epochs and early stopping patience
* Model dimensions (embedding sizes, hidden layers)
* Paths to training and testing datasets

## Training

To initiate the training pipeline, execute your main training script:

```bash
python train.py
```

## Usage

Here is a quick example of how to import your model and run inference:

```python
from model import FADESModel
from config import load_config

# Load parameters
cfg = load_config()

# Initialize the model
model = FADESModel(cfg)

# Example forward pass
# predictions = model(sample_input)
```
