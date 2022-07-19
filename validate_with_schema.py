import json
import jsonschema
from jsonschema import validate


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
        print(err)
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message


# Convert json to python object.
with open("ds2_dense_test.json", "r") as file:
    json_data = json.load(file)

# validate it
is_valid, msg = validate_json(json_data)
print(msg)
