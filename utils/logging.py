import json
import os
from tensorboardX import SummaryWriter


def get_num_parameter(model, trainable=False):
    if trainable:
        params = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    else:
        params = [(n, p) for (n, p) in model.named_parameters()]

    total_params = sum(p.numel() for (n, p) in params)
    num_param_list = [(n, p.numel()) for (n, p) in params]

    return total_params, num_param_list


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


class DummySummaryWriter:
    """Mock a TensorboardX summary writer but does not do anything"""

    def __init__(self):
        def noop(*args, **kwargs):
            pass

        s = SummaryWriter()
        for f in dir(s):
            if not f.startswith("_"):
                self.__setattr__(f, noop)


class JSONLogger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, filename, auto_save=True):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.filename = filename
        self.values = []
        self.auto_save = auto_save

        # Ensure the output directory exists
        directory = os.path.dirname(self.filename)
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    def log_metric(self, name, values, tags):
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})
        print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))
        if self.auto_save:
            self.save()

    def save(self):
        """
        Save the internal memory to a file
        """
        with open(self.filename, "w") as fp:
            json.dump(self.values, fp, indent=" ")
