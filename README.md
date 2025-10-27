# Overtourism Modeling

This repository contains code for modeling overtourism in
[Lake Molveno](https://en.wikipedia.org/wiki/Lake_Molveno) in
Italy. We develop this repository in the context of our R&D
activities at [Fondazione Bruno Kessler](https://www.fbk.eu/en/).

## Getting Started

The code is written in Python. We recommend using [uv](https://astral.sh/uv)
for managing the Python version, the virtual environment, and the
dependencies. Please, refer to `uv` documentation regarding how to
install it for your operating system.

Once `uv` is installed, use these commands:

```bash
git clone git@github.com/tn-aixpa/overtourism
cd overtourism
uv venv
source .venv/bin/activate
uv sync --dev
```

## Dependencies

See [pyproject.toml](pyproject.toml) for the full list of dependencies. In terms of
code developed at [Fondazione Bruno Kessler](https://www.fbk.eu/en/), the main
dependencies are the following:

- [dt-model](https://github.com/tn-aixpa/dt-model): Digital Twin modeling library.

Dependencies are anyway automatically installed by `uv sync --dev`.

### Updating specific dependencies

To update a specific dependency, use the following command:

```bash
uv sync --dev -P "${dependency}"
```

For example,

```bash
uv sync --dev -P dt-model
```

updates `dt-model` to the latest version.

## Usage

- [Data Preparation](./docs/howto/data.md)
- [Build and run Backend application](./docs/howto/backend.md)
- [Build and run Frontend application](./docs/howto/frontend.md)

## License

```
SPDX-License-Identifier: Apache-2.0
```
