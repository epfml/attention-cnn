#!/usr/bin/env python3

import json
import os
import sys

# Make sure the root of the project is in the python path.
# @todo: Should we do this with absolute imports?
sys.path.append('..')
import train
import utils.logging


for lr in [0.5, 0.2, 0.1]:
    for mom in [0.9, 0]:
        # Define a fresh output directory
        train.output_dir = 'output/tuning/lr{}_mom{}'.format(lr, mom)
        os.makedirs(train.output_dir)

        # Configure the experiment
        train.config = dict(
            dataset='Cifar100',
            model='resnet18',
            optimizer='SGD',
            optimizer_decay_at_epochs=[150, 250],
            optimizer_decay_with_factor=10.0,
            optimizer_learning_rate=lr,
            optimizer_momentum=mom,
            optimizer_weight_decay=0.0005,
            batch_size=256,
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

        # Keep track of the accuracies achieved
        print(lr, mom, best_accuracy)
