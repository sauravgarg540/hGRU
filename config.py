import configparser


# Method to convert configuration to list of dictionaries
def configuration(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    sections = config.sections()
    configuration = []
    for section in sections:
        temp = {}
        temp[section] = dict(config[section])
        configuration.append(temp)
    return configuration