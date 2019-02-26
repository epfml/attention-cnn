#!/usr/bin/env python3

import os
import sys
import syslog

# Make sure the root of the project is in the python path.
# @todo: Should we do this with absolute imports?
sys.path.append('..')
import train
import utils.logging


train.output_dir = 'output/vgg'
train.config = dict(
    dataset='Cifar10',
    model='vgg11',
    optimizer='SGD',
    optimizer_decay_at_epochs=[30, 60, 90, 120, 150, 180, 210, 240, 270],
    optimizer_decay_with_factor=2.0,
    optimizer_learning_rate=0.05,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0005,
    batch_size=128,
    num_epochs=300,
    seed=42,
)
logfile = utils.logging.JSONLogger(os.path.join(train.output_dir, 'metrics.json'))
train.log_metric = logfile.log_metric
train.main()
