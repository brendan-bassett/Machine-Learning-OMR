import json
import logging
import os

import jsonschema
from jsonschema import validate


# ============== CONSTANTS =========================================================================================

ROOT_PATH = os.path.realpath(os.path.dirname(__file__))

# ============== FUNCTIONS =========================================================================================

def get_schema():
    with open('ds2_schema.json', 'r') as file:
        schema = json.load(file)
    return schema


def validate_json(json_data):
    """REF: https://json-schema.org/ """
    # Describe what kind of json you expect.
    execute_api_schema = get_schema()

    try:
        validate(instance=json_data, schema=execute_api_schema)
    except jsonschema.exceptions.ValidationError as err:
        logging.info(err)
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message


# ============== MAIN CODE =========================================================================================

logging.info("Setting up loggger...")

log_path = os.path.join(ROOT_PATH + '/logs/validate_with_schema.log')
if not os.path.exists(log_path):
    with open(log_path, 'w+') as lg:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
logger = logging.getLogger()
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logging.info("Converting json to python object...")

# Convert json to python object.
with open("ds2_dense_test.json", "r") as file:
    json_data = json.load(file)

# Validate it
is_valid, msg = validate_json(json_data)
logging.info(msg)
