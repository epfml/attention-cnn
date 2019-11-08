import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
from tqdm import tqdm
import yaml
import enum
from enum import Enum
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
from termcolor import colored
import tabulate

import models
from utils.data import MaskedDataset
from utils.logging import get_num_parameter, human_format, DummySummaryWriter, sizeof_fmt
from utils.plotting import plot_attention_positions_all_layers
from utils.config import parse_cli_overides
from utils.learning_rate import linear_warmup_cosine_lr_scheduler
import utils.accumulators


# fmt: off
config = OrderedDict(
    dataset="Cifar10",
    model="bert",
    load_checkpoint_file=None,
    no_cuda=False,

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
    position_encoding_size=-1,              # dimension of the position embedding for relative attention, if -1 will default to  hidden_size
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=16,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,

    # === BERT IMAGE===
    add_positional_encoding_to_input=False,
    use_learned_2d_encoding=False,
    share_position_encoding=False,           # share learned relative position encoding for all layers
    use_attention_data=False,                # use attention between pixel values instead of only positional
    use_gaussian_attention=True,
    attention_isotropic_gaussian=False,
    prune_degenerated_heads=False,           # remove heads with Sigma^{-1} close to 0 or very singular (kappa > 1000) at epoch 0
    reset_degenerated_heads=False,           # reinitialize randomly the heads mentioned above
    fix_original_heads_position=False,       # original heads (not pruned/reinit) position are fixed to their original value
    fix_original_heads_weights=False,        # original heads (not pruned/reinit) value matrix are fixed to their original value
    gaussian_spread_regularizer=0.,          # penalize singular covariance gaussian attention

    gaussian_init_sigma_std=0.01,
    gaussian_init_mu_std=2.,
    attention_gaussian_blur_trick=False,     # use a computational trick for gaussian attention to avoid computing the attention probas
    pooling_concatenate_size=2,              # concatenate the pixels value by patch of pooling_concatenate_size x pooling_concatenate_size to redude dimension
    pooling_use_resnet=False,

    # === LOGGING ===
    only_list_parameters=False,
    num_keep_checkpoints=0,
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
    device = torch.device("cuda:0" if not config["no_cuda"] and torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset(test_batch_size=config["batch_size"])
    model = get_model(device)

    print_parameters(model)
    if config["only_list_parameters"]:
        print_flops(model)

    if config["load_checkpoint_file"] is not None:
        restore_checkpoint(config["load_checkpoint_file"], model, device)

    # for each layer, which heads position to block list[list[int]]
    original_heads_per_layer = None

    if config["prune_degenerated_heads"]:
        assert config["model"] == "bert" and config["use_gaussian_attention"]
        with torch.no_grad():
            heads_to_prune = find_degenerated_heads(model)
            model.prune_heads(heads_to_prune)
            original_heads_per_layer = [
                torch.tensor(list(range(model.encoder.layer[layer_idx].attention.self.num_attention_heads)))
                for layer_idx in range(config["num_hidden_layers"])
            ]

        print_parameters(model)
        print_flops(model)

    if config["reset_degenerated_heads"]:
        assert config["model"] == "bert" and config["use_gaussian_attention"]
        with torch.no_grad():
            heads_to_reset = find_degenerated_heads(model)
            model.reset_heads(heads_to_reset)
            original_heads_per_layer = [
                torch.tensor([
                    head_idx
                    for head_idx in range(model.encoder.layer[layer_idx].attention.self.num_attention_heads)
                    if head_idx not in heads_to_reset.get(layer_idx, [])
                ])
                for layer_idx in range(config["num_hidden_layers"])
            ]

    if config["only_list_parameters"]:
        exit()

    max_steps = config["num_epochs"]
    if config["optimizer_cosine_lr"]:
        max_steps *= len(training_loader.dataset) // config["batch_size"] + 1

    optimizer, scheduler = get_optimizer(model.named_parameters(), max_steps)
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()
    checkpoint_every_n_epoch = None
    if config["num_keep_checkpoints"] > 0:
        checkpoint_every_n_epoch = max(1, config["num_epochs"] // config["num_keep_checkpoints"])
    else:
        checkpoint_every_n_epoch = 999999999999
    global_step = 0

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))

        if (
            "bert" in config["model"]
            and config["plot_attention_positions"]
            and (config["use_gaussian_attention"] or config["use_learned_2d_encoding"])
        ):
            if not config["attention_gaussian_blur_trick"]:
                plot_attention_positions_all_layers(model, 9, writer, epoch)
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

        for batch_x, batch_y in tqdm(training_loader):

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

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

            classification_loss = criterion(prediction, batch_y)
            loss = classification_loss

            if config["gaussian_spread_regularizer"] > 0:
                gaussian_regularizer_loss = config["gaussian_spread_regularizer"] * get_singular_gaussian_penalty(model)
                loss += gaussian_regularizer_loss

            acc = accuracy(prediction, batch_y)

            loss.backward()

            # set blocked gradient to 0
            if config["fix_original_heads_position"] and original_heads_per_layer is not None:
                for layer_idx, heads_to_fix in enumerate(original_heads_per_layer):
                    model.encoder.layer[layer_idx].attention.self.attention_spreads.grad[heads_to_fix].zero_()
                    model.encoder.layer[layer_idx].attention.self.attention_centers.grad[heads_to_fix].zero_()

            if config["fix_original_heads_weights"] and original_heads_per_layer is not None:
                for layer_idx, heads_to_fix in enumerate(original_heads_per_layer):
                    layer = model.encoder.layer[layer_idx]
                    n_head = layer.attention.self.num_attention_heads
                    d_head = layer.attention.self.attention_head_size
                    mask = torch.zeros([n_head, d_head], dtype=torch.bool)
                    for head in heads_to_fix:
                        mask[head] = 1
                    mask = mask.view(-1)
                    layer.attention.self.value.weight.grad[:, mask].zero_()

            # Do an optimizer step
            optimizer.step()

            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/classification-loss", classification_loss, global_step)
            if config["gaussian_spread_regularizer"] > 0:
                writer.add_scalar("train/gaussian_regularizer_loss", gaussian_regularizer_loss, global_step)
            writer.add_scalar("train/accuracy", acc, global_step)

            global_step += 1

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))


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
        if epoch % checkpoint_every_n_epoch == 0:
            store_checkpoint("{:04d}.checkpoint".format(epoch), model, epoch, mean_test_accuracy.value())

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
    elif config["dataset"].startswith("/"):
        train_data = torch.load(config["dataset"] + ".train")
        test_data = torch.load(config["dataset"] + ".test")
        training_set = TensorDataset(train_data["data"], train_data["target"])
        test_set = TensorDataset(test_data["data"], test_data["target"])

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

def split_dict(d, first_predicate):
    """split the dictionary d into 2 dictionaries, first one contains elements validating first_predicate"""
    first, second = OrderedDict(), OrderedDict()
    for key, value in d.items():
        if first_predicate(key):
            first[key] = value
        else:
            second[key] = value
    return first, second

def get_optimizer(model_named_parameters, max_steps):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config["optimizer"] == "SGD":
        without_weight_decay, with_weight_decay = split_dict(
            OrderedDict(model_named_parameters),
            lambda name: "attention_spreads" in name or "attention_centers" in name
        )

        optimizer = torch.optim.SGD(
            [
                {"params": with_weight_decay.values()},
                {"params": without_weight_decay.values(), "weight_decay": 0.}
            ],
            lr=config["optimizer_learning_rate"],
            momentum=config["optimizer_momentum"],
            weight_decay=config["optimizer_weight_decay"],
        )
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model_named_parameters.values(), lr=config["optimizer_learning_rate"])
    else:
        raise ValueError("Unexpected value for optimizer")

    if config["optimizer"] == "Adam":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.)
        print("Adam optimizer ignore all learning rate schedules.")
    elif config["optimizer_cosine_lr"]:
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
    num_classes = 2
    if config["dataset"] == "Cifar100":
        num_classes = 100
    elif config["dataset"] == "Cifar10":
        num_classes = 10

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

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model

def print_parameters(model):
    # compute number of parameters
    num_params, _ = get_num_parameter(model, trainable=False)
    num_bytes = num_params * 32 // 8  # assume float32 for all
    print(f"Number of parameters: {human_format(num_params)} ({sizeof_fmt(num_bytes)} for float32)")
    num_trainable_params, trainable_parameters = get_num_parameter(model, trainable=True)
    print("Number of trainable parameters:", human_format(num_trainable_params))

    if config["only_list_parameters"]:
        # Print detailed number of parameters
        print(tabulate.tabulate(trainable_parameters))


def print_flops(model):
    shape = None
    if config["dataset"] in ["Cifar10", "Cifar100"]:
        shape = (1, 3, 32, 32)
    else:
        print(f"Unknown dataset {config['dataset']} input size to compute # FLOPS")
        return

    try:
        from thop import profile
    except:
        print("Please `pip install thop` to compute # FLOPS")
        return

    model = model.train()
    input_data = torch.rand(*shape)
    num_flops, num_params = profile(model, inputs=(input_data, ))
    print("Number of FLOPS:", human_format(num_flops))


def find_degenerated_heads(model):
    """
    returns a dict of degenerated head per layer like {layer_idx -> [head_idx, ...]}
    """
    model_params = dict(model.named_parameters())
    degenerated_heads = OrderedDict()
    degenerated_reasons = []

    for layer_idx in range(config["num_hidden_layers"]):
        prune_heads = []
        sigmas_half_inv = model_params["encoder.layer.{}.attention.self.attention_spreads".format(layer_idx)]

        for head_idx in range(config["num_attention_heads"]):
            head_is_degenerated = False


            if config["attention_isotropic_gaussian"]:
                sigma_inv = sigmas_half_inv[head_idx]
                if sigma_inv ** 2 < 1e-5:
                    degenerated_reasons.append("Sigma too low -> uniform attention: sigma**-2= {}".format(sigma_inv ** 2))
                    head_is_degenerated = True
            else:
                sigma_half_inv = sigmas_half_inv[head_idx]
                sigma_inv = sigma_half_inv.transpose(0, 1) @ sigma_half_inv
                eig_values = torch.eig(sigma_inv)[0][:, 0].abs()
                condition_number = eig_values.max() / eig_values.min()

                if condition_number > 1000:
                    degenerated_reasons.append("Covariance matrix is ill defined: condition number = {}".format(condition_number))
                    head_is_degenerated = True
                elif eig_values.max() < 1e-5:
                    degenerated_reasons.append("Covariance matrix is close to 0: largest eigen value = {}".format(eig_values.max()))
                    head_is_degenerated = True

            if head_is_degenerated:
                prune_heads.append(head_idx)

        if prune_heads:
            degenerated_heads[layer_idx] = prune_heads

    if degenerated_heads:
        print("Degenerated heads:")
        reasons = iter(degenerated_reasons)
        table = [(layer, head, next(reasons)) for layer, heads in degenerated_heads.items() for head in heads]
        print(tabulate.tabulate(table, headers=["layer", "head", "reason"]))

    return degenerated_heads

def get_singular_gaussian_penalty(model):
    """Return scalar high when attention covariance get very singular
    """
    if config["attention_isotropic_gaussian"]:
        # TODO move at setup
        print("Singular gaussian penalty ignored as `attention_isotropic_gaussian` is True")
        return 0

    condition_numbers = []
    for layer in model.encoder.layer:
        for sigma_half_inv in layer.attention.self.attention_spreads:
            sigma_inv = sigma_half_inv.transpose(0, 1) @ sigma_half_inv
            eig_values = torch.eig(sigma_inv)[0][:, 0].abs()
            condition_number = eig_values.max() / eig_values.min()
            condition_numbers.append(condition_number)

    return torch.mean((torch.tensor(condition_numbers)  - 1) ** 2)

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

    # remove buffer from checkpoint
    # TODO should not hard code
    def keep_state_dict_keys(key):
        if "self.R" in key:
            return False
        return True

    time.sleep(
        1
    )  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "epoch": epoch,
            "test_accuracy": test_accuracy,
            "model_state_dict": OrderedDict([
                (key, value) for key, value in model.state_dict().items() if keep_state_dict_keys(key)
            ]),
        },
        path,
    )

def restore_checkpoint(filename, model, device):
    """Load model from a checkpoint"""
    print("Loading model parameters from '{}'".format(filename))
    with open(filename, "rb") as f:
        checkpoint_data = torch.load(f, map_location=device)

    try:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    except RuntimeError as e:
        print(colored("Missing state_dict keys in checkpoint", "red"), e)
        print("Retry import with current model values for missing keys.")
        state = model.state_dict()
        state.update(checkpoint_data["model_state_dict"])
        model.load_state_dict(state)


if __name__ == "__main__":
    # if directly called from CLI (not as module)
    # we parse the parameters overides
    config = parse_cli_overides(config)
    main()
