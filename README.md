# Leopard-EM: Python based template matching

[![License](https://img.shields.io/pypi/l/Leopard-EM.svg?color=green)](https://github.com/Lucaslab-Berkeley/Leopard-EM/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/Leopard-EM.svg?color=green)](https://pypi.org/project/Leopard-EM)
[![Python Version](https://img.shields.io/pypi/pyversions/Leopard-EM.svg?color=green)](https://python.org)
[![CI](https://github.com/Lucaslab-Berkeley/Leopard-EM/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucaslab-Berkeley/Leopard-EM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lucaslab-Berkeley/Leopard-EM/branch/main/graph/badge.svg)](https://github.com/Lucaslab-Berkeley/Leopard-EM)

Leopard-EM (**L**ocation & ori**E**ntati**O**n of **PAR**ticles found using two-**D**imensional t**E**mplate **M**atching) is a python package for running two-dimensional template matching (2DTM) on cryo-EM images.

<!-- ## Documentation and Examples

See the `/examples` directory for a set of Jupyter notebooks demonstrating some basic usage of the package.
More extensive documentation can be found at (TODO: Add link to documentation site). -->

## Basic Installation

The newest released version of the package can be installed from PyPI using pip:

```bash
pip install leopard-em
```

## Usage

### Template Matching

Inputs to the template matching programs can be configured with Pydantic models (see online documentation for examples and use cases).
Alternatively, configurations can be set in YAML files and loaded into the `MatchTemplateManager` object.
The [example YAML configuration file](match_template_example_config.yaml) acts as a template for configuring your own runs.
Once configured with the proper paths, parameters, etc., the program can run as follows:

```python
from leopard_em.pydantic_models import MatchTemplateManager

YAML_CONFIG_PATH = "path/to/mt_config.yaml"
ORIENTATION_BATCH_SIZE = 8

def main():
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mt_manager.run_match_template(ORIENTATION_BATCH_SIZE)
    df = mt_manager.results_to_dataframe()
    df.to_csv("/path/to/results.csv")

# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
```

### Template Refinement

Particle orientations and locations can be refined using the `RefineTemplateManager` objects after a template matching run.
The `RefineTemplateManager` is similarly a set of Pydantic models capable of configuration via YAML files.
The [example YAML configuration file](refine_template_example_config.yaml) acts as a template for configuring your own runs.
Once configured with the proper paths, parameters, etc., the program can run as follows:

```python
from leopard_em.pydantic_models import RefineTemplateManager

YAML_PATH = "/path/to/rt_config.yaml"
ORIENTATION_BATCH_SIZE = 80

def main():
    rt_manager = RefineTemplateManager.from_yaml(YAML_PATH)
    rt_manager.run_refine_template(
        output_dataframe_path="/path/to/refined_results.csv",
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()

```

## Installation for Development

The package can be installed from source in editable mode with the optional development libraries via pip.

```bash
git clone https://github.com/Lucaslab-Berkeley/Leopard-EM.git
cd Leopard-EM
pip install -e '.[dev,test, docs]'
```

Further information on development and contributing to the repo can be found in our online documentation.