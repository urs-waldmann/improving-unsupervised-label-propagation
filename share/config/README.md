# Configuration files

The configurations are saved in json format and must pass a json schema validation.

## IDE support

The schema can be configured in IDEs like PyCharm, such that auto-completion and correctness checks are available.

For PyCharm, create the mapping from `./share/config/schema/config-schema.json` to `./share/config/label_propagation` in
```
Settings > Languages & Frameworks > Schemas and DTDs > JSON Schema Mappings
```

Alternatively, the correctness of a configuration can be checked with the `validation_example.py` script by calling

```
python ./validation_example.py --config <path-to-config.json>
```

## Modifying the schema

<https://json-schema.org/> is a good resource to learn about JSON schemata, but simple modifications should be straight
forward, because the existing parts can be used as reference.

Some parts of the schema are a little unintuitive, because the schema evolved over time and these parts are necessary
for backwards compatibility.
