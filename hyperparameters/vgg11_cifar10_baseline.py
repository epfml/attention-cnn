config = dict(
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

test_accuracy = -1 # @todo
