config = dict(
    dataset='Cifar10',
    model='resnet18',
    optimizer='SGD',
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=256,
    num_epochs=300,
    seed=42,
)

test_accuracy = -1 # @todo
