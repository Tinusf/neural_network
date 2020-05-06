class Config:
    config_types = {
        "training": "str",
        "validation": "str",
        "layers": "list_int",
        "activations": "list_str",
        "loss_type": "str",
        "learning_rate": "float",
        "no_epochs": "int",
        "L2_regularization": "float"
    }
    config = {}

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
            if key in self.config_types:
                if self.config_types[key] == "float":
                    self.config[key] = float(attr)
                elif self.config_types[key] == "int":
                    self.config[key] = int(attr)
                elif self.config_types[key] == "list_int":
                    # Remove all spaces
                    attr = attr.replace(" ", "")
                    if attr == "0":
                        self.config[key] = []
                    else:
                        # Split on comma.
                        attr_list = attr.split(",")
                        attr_list_int = [int(attr) for attr in attr_list]
                        self.config[key] = attr_list_int
                elif self.config_types[key] == "list_str":
                    # Remove all spaces
                    attr = attr.replace(" ", "")
                    # Split on comma.
                    attr_list = attr.split(",")
                    self.config[key] = attr_list
                elif self.config_types[key] == "str":
                    self.config[key] = attr

            else:
                print("Unknown key in config file.")
