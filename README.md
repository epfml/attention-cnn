# Cifar 10/100 default implementation

MLO internal cifar 10 / 100 reference implementation.

- Single machine
- Variable batch sizes
- ...


## Getting started

- Install Python 3 and `pip`.
- Clone this repository and open it.
- `pip install -r requirements.txt`



## Code organization

### train.py
This file contains the training loop and it sets up the optimization task. It contains a global `config` dictionary that should contain all configurable parameters. This file can be run standalone (`python3 ./train.py`) or by a manager script (see below).


### experiments/
To do an experiment with specific settings for the `config` dictionary, you can import `train.py` as a module and overwrite its placeholder definitions for `config`, `log_metric` and `output_dir`.

A proper experiment could look like this:

```python
import train

train.output_dir = 'output/tuning/lr{}_mom{}'.format(lr, mom)
os.makedirs(train.output_dir)

# Configure the experiment
train.config = dict(
    dataset='Cifar100',
    model='resnet18',
    optimizer='SGD',
    optimizer_decay_at_epochs=[30, 60, 90, 120, 150, 180, 210, 240, 270],
    optimizer_decay_with_factor=2.0,
    optimizer_learning_rate=lr,
    optimizer_momentum=mom,
    optimizer_weight_decay=0.0005,
    batch_size=128,
    num_epochs=2,
    seed=42,
)

# Save the config
with open(os.path.join(train.output_dir, 'config.json'), 'w') as fp:
    json.dump(train.config, fp, indent=' ')

# Configure the logging of scalar measurements
logfile = utils.logging.JSONLogger(os.path.join(train.output_dir, 'metrics.json'))
train.log_metric = logfile.log_metric

# Train
best_accuracy = train.main()
```

The `experiments/` directory contains an example of a hyperparameter [grid search](experiments/grid_search_demo.py).


### models/
This directory contains model definitions for many popular computer vision networks. They were copied from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and slightly extended by Quentin, Praneeth and Thijs.


### hyperparameters/
This directory is supposed to contain reference settings for hyperparameters, together with the accuracy they are expected to achieve.


### utils/
Miscelaneous utilities. At the time of writing these docs, this contains accumulators for running averages and max, and a simple logging class.

