class Config:
    config = {
        "training": None,
        "validation": None,
        "layers": None,
        "activations": None,
        "loss_type": None,
        "learning_rate": None,
        "no_epochs": None,
        "L2_regularization": None
    }

    def __init__(self, file_path):
        lines = self.read_file(file_path)
        self.parse_config(lines)

    def read_file(self, file):
        """
        :param file: The file path
        :return:
        """
        f = open(file)
        lines = f.readlines()
        f.close()
        return lines

    def parse_config(self, lines):
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments and square brackets.
            if line == "" or line.startswith("#") or line.startswith("["):
                continue
            key, attr = line.split("=")
            key, attr = key.strip(), attr.strip()
            if key in self.config.keys():
                self.config[key] = attr
            else:
                print("Unknown key in config file.")
