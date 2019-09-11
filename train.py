import os
import time
from enum import Enum

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import argparse
import models
from utils.learning_rate import linear_warmup_cosine_lr_scheduler
import utils.accumulators
from models.transformer import PositionalEncodingType
from timer import default
from utils.data import MaskedDataset
from tensorboardX import SummaryWriter
from collections import OrderedDict
from termcolor import colored
from utils.logging import get_num_parameter, human_format, DummySummaryWriter, sizeof_fmt
from utils.plotting import plot_attention_positions_all_layers
from utils.config import parse_cli_overides
import yaml
import enum

import tabulate

timer = default()

# fmt: off
config = OrderedDict(
    dataset="Cifar10",
    model="bert",

    # === OPTIMIZER ===
    optimizer="SGD",
    optimizer_cosine_lr=False,
    optimizer_warmup_ratio=0.0,  # period of linear increase for lr scheduler
    optimizer_decay_at_epochs=[80, 150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=300,
    num_epochs=300,
    seed=42,

    # === From BERT ===
    vocab_size_or_config_json_file=-1,
    hidden_size=128,  # 768,
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,

    # === BERT IMAGE===
    positional_encoding=PositionalEncodingType.Learned,
    max_positional_encoding=8,                           # relative position encoding will only consider distances in [-k,k],
                                                         # and the ones larger than k will be regarded as k.
    use_local_attention=False,                           # use local attention in BertSelfAttentionDilation or not
    shared_position_embedding=False,                     # sharing the position embedding and lookup matrices among all
                                                         # the BertSelfAttentionDilation layers
    attention_type="gaussian",                           # type of attention : "dilation" or "gaussian"
    attention_isotropic_gaussian=False,
    attention_gaussian_blur_trick=False,                 # use a computational trick for gaussian attention to avoid computing the attention probas
    pooling_concatenate_size=2,                          # concatenate the pixels value by patch of pooling_concatenate_size x pooling_concatenate_size to redude dimension
    pooling_use_resnet=False,

    # === LOGGING ===
    only_time_one_epoch=False,  # show timer after 1 epoch and stop
    only_list_parameters=False,
    num_keep_checkpoints=10,
    plot_attention_positions=True,
    output_dir="./output.tmp",
)
# fmt: on

output_dir = "./output.tmp"  # Can be overwritten by a script calling this


def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    """
    Directory structure:

      output_dir
        |-- config.yaml
        |-- best.checkpoint
        |-- last.checkpoint
        |-- tensorboard logs...
    """

    global output_dir
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok = True)

    # save config in YAML file
    store_config()

    # create tensorboard writter
    writer = SummaryWriter(logdir=output_dir, max_queue=100, flush_secs=10)
    print(f"Tensorboard logs saved in '{output_dir}'")

    # Set the seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset(test_batch_size=config["batch_size"])
    model = get_model(device)

    max_steps = config["num_epochs"]
    if config["optimizer_cosine_lr"]:
        max_steps *= len(training_loader.dataset) // config["batch_size"] + 1

    optimizer, scheduler = get_optimizer(model.parameters(), max_steps)
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()
    checkpoint_every_n_epoch = None
    if config["num_keep_checkpoints"] > 0:
        checkpoint_every_n_epoch = max(1, config["num_epochs"] // config["num_keep_checkpoints"])
    global_step = 0

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))

        if (
            "bert" in config["model"]
            and config["plot_attention_positions"]
            and config["attention_type"] == "gaussian"
        ):
            if not config["attention_gaussian_blur_trick"]:
                plot_attention_positions_all_layers(model, (16, 16), writer, epoch)
            else:
                # TODO plot gaussian without attention weights
                pass

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Update the optimizer's learning rate
        if config["optimizer_cosine_lr"]:
            scheduler.step(global_step)
        else:
            scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)

        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        time_i = 0
        loader_time_context = timer("loader")
        loader_time_context.__enter__()
        for batch_x, batch_y in tqdm(training_loader):

            loader_time_context.__exit__(None, None, None)
            with timer("move_to_device"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # batch_mask = batch_mask.to(device)
            batch_mask = None

            batch_size, _, width, height = batch_x.shape

            # Compute gradients for the batch
            optimizer.zero_grad()

            if config["pooling_use_resnet"]:
                # , image_out, reconstruction, reconstruction_mask
                prediction = model(batch_x)  # , batch_mask)
            else:
                # prediction, image_out
                prediction = model(batch_x)  # , batch_mask)
                # reconstruction = batch_x
                # reconstruction_mask = batch_mask

            with timer("loss"):
                classification_loss = criterion(prediction, batch_y)
                loss = classification_loss

            acc = accuracy(prediction, batch_y)

            with timer("backward"):
                loss.backward()

            # Do an optimizer step
            with timer("weights_update"):
                optimizer.step()

            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/accuracy", acc, global_step)

            global_step += 1

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

            loader_time_context = timer("loader")
            loader_time_context.__enter__()

        # Log training stats
        log_metric(
            "accuracy", {"epoch": epoch, "value": mean_train_accuracy.value()}, {"split": "train"}
        )
        log_metric(
            "cross_entropy", {"epoch": epoch, "value": mean_train_loss.value()}, {"split": "train"}
        )
        log_metric("lr", {"epoch": epoch, "value": scheduler.get_lr()[0]}, {})

        # Evaluation
        with torch.no_grad():
            model.eval()
            mean_test_accuracy = utils.accumulators.Mean()
            mean_test_loss = utils.accumulators.Mean()
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                mean_test_loss.add(loss.item(), weight=len(batch_x))
                mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            "accuracy", {"epoch": epoch, "value": mean_test_accuracy.value()}, {"split": "test"}
        )
        log_metric(
            "cross_entropy", {"epoch": epoch, "value": mean_test_loss.value()}, {"split": "test"}
        )
        writer.add_scalar("eval/classification_loss", mean_test_loss.value(), epoch)
        writer.add_scalar("eval/accuracy", mean_test_accuracy.value(), epoch)

        # Store checkpoints for the best model so far
        is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
        if is_best_so_far:
            store_checkpoint("best.checkpoint", model, epoch, mean_test_accuracy.value())
        if (epoch + 1) % checkpoint_every_n_epoch == 0:
            store_checkpoint("{:04d}.checkpoint".format(epoch), model, epoch, mean_test_accuracy.value())

        # writer.flush()

        if config["only_time_one_epoch"]:
            print(timer.summary())
            exit(0)

    # Store a final checkpoint
    store_checkpoint(
        "final.checkpoint", model, config["num_epochs"] - 1, mean_test_accuracy.value()
    )
    writer.close()

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    """
    Log timeseries data.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def get_dataset(test_batch_size=100, shuffle_train=True, num_workers=2, data_root="./data"):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config["dataset"] == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif config["dataset"] == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(config["dataset"]))

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
    test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return training_loader, test_loader


def get_optimizer(model_parameters, max_steps):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config["optimizer_learning_rate"],
            momentum=config["optimizer_momentum"],
            weight_decay=config["optimizer_weight_decay"],
        )
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=config["optimizer_learning_rate"])
    else:
        raise ValueError("Unexpected value for optimizer")

    if config["optimizer_cosine_lr"]:
        scheduler = linear_warmup_cosine_lr_scheduler(
            optimizer, config["optimizer_warmup_ratio"], max_steps
        )

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["optimizer_decay_at_epochs"],
            gamma=1.0 / config["optimizer_decay_with_factor"],
        )

    return optimizer, scheduler


def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11": lambda: models.VGG("VGG11", num_classes, batch_norm=False),
        "vgg11_bn": lambda: models.VGG("VGG11", num_classes, batch_norm=True),
        "vgg13": lambda: models.VGG("VGG13", num_classes, batch_norm=False),
        "vgg13_bn": lambda: models.VGG("VGG13", num_classes, batch_norm=True),
        "vgg16": lambda: models.VGG("VGG16", num_classes, batch_norm=False),
        "vgg16_bn": lambda: models.VGG("VGG16", num_classes, batch_norm=True),
        "vgg19": lambda: models.VGG("VGG19", num_classes, batch_norm=False),
        "vgg19_bn": lambda: models.VGG("VGG19", num_classes, batch_norm=True),
        "resnet10": lambda: models.ResNet10(num_classes=num_classes),
        "resnet18": lambda: models.ResNet18(num_classes=num_classes),
        "resnet34": lambda: models.ResNet34(num_classes=num_classes),
        "resnet50": lambda: models.ResNet50(num_classes=num_classes),
        "resnet101": lambda: models.ResNet101(num_classes=num_classes),
        "resnet152": lambda: models.ResNet152(num_classes=num_classes),
        "bert": lambda: models.BertImage(config, num_classes=num_classes),
    }[config["model"]]()

    # compute number of parameters
    num_params, _ = get_num_parameter(model, trainable=False)
    num_bytes = num_params * 32 // 8  # assume float32 for all
    print(f"Number of parameters: {human_format(num_params)} ({sizeof_fmt(num_bytes)} for float32)")
    num_trainable_params, trainable_parameters = get_num_parameter(model, trainable=True)
    print("Number of trainable parameters:", human_format(num_trainable_params))

    if config["only_list_parameters"]:
        print(tabulate.tabulate(trainable_parameters))
        exit()

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model


def store_config():
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(dict(config), f, sort_keys=False)


def store_checkpoint(filename, model, epoch, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    time.sleep(
        1
    )  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {"epoch": epoch, "test_accuracy": test_accuracy, "model_state_dict": model.state_dict()},
        path,
    )


if __name__ == "__main__":
    # if directly called from CLI (not as module)
    # we parse the parameters overides
    config = parse_cli_overides(config)
    main()
