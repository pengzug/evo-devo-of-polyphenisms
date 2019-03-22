# Evolution of polyphenisms

## Example usage

```bash
$ python edop.py -c config.toml
```

`config.toml` is a configuration file in TOML format that you can use to specify model configuration:

```toml
n = 500  # Population size
num_gen = 1000  # Number of generations
```

Alternatively, you can override configuration settings from the command line:

```bash
$ python edop.py -s n 500 -s num_gen 1000
```

To see all available CLI arguments, run

```bash
$ python edop.py --help
```
