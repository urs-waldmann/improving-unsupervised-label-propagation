"""
Helper script to validate a configuration against the config schema.
"""

import argparse
import json

import jsonschema as jss


def validate_schema(config_path: str, schema_path: str):
    with open(config_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    with open(schema_path, 'r', encoding='utf8') as f:
        schema = json.load(f)

    # Raises exception if the validation failed
    try:
        jss.validate(data, schema=schema)
        print('Configuration validated successfully.')
    except jss.exceptions.ValidationError as e:
        print('Schema validation failed.')
        print(e)
    except jss.exceptions.SchemaError as e:
        print('Schema is invalid.')
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema',
                        type=str,
                        default='./schema/config-schema.json',
                        help='Path to the schema file.')
    parser.add_argument('--config',
                        type=str,
                        default='./label_propagation/config-test.json',
                        help='Path to the config file.')
    args = parser.parse_args()
    validate_schema(args.config, args.schema)
