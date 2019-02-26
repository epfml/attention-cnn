import json
import os


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
        self.values.append({
            'measurement': name,
            **values,
            **tags,
        })
        print("{name}: {values} ({tags})".format(
            name=name, values=values, tags=tags))
        if self.auto_save:
            self.save()

    def save(self):
        """
        Save the internal memory to a file
        """
        with open(self.filename, 'w') as fp:
            json.dump(self.values, fp, indent=' ')
