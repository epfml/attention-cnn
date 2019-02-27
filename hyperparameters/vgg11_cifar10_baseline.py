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

best_test_accuracy = 0.9104999816417694
best_test_accuracy_reached_at_epoch = 121

final_test_accuracy = 0.9039999830722811 # at epoch 299, of course

runtime = 1 + 6./60. # hours, on a Tesla V100


# if __name__ == '__main__':
#     import sys
#     import os
#     sys.path.append('..')
#     import train

#     train.output_dir = os.path.join('output', os.path.basename(__file__))
#     os.makedirs(train.output_dir)
#     train.config = config
#     best_accuracy = train.main()
#     with open(os.path.join(train.output_dir, 'best_accuracy.txt'), 'w') as fp:
#         fp.write(str(best_accuracy))
