# Testing

<!-- **Note:** Running the tests together fails due to some internal TensorFlow problem, avoid running: -->
All commands must to be run at repository folder level. 

Before running any test, generate a sample dataset (see below).

### Table of contents

- [Prepare sample dataset](#prepare-sample-dataset)
- [Run tests](#run-tests)
    - [Run all tests](#run-all-tests)
    - [Run specific module tests](#run-specific-module-tests)

## Prepare sample dataset

The sample dataset is generated using the [unsplash.txt](../data/unsplash.txt) file. Run

```bash
bash tests/prepare-sample-dataset.sh
```

which places the sample dataset in folder folder [tests/data](data).

## Run tests

### Run all tests

You can run all tests by using

```bash
python -m unittest discover tests -v
```

> **Note:** `test_colorization.py` will train a simple model for 20 epochs, might take few seconds.

### Run specific module tests

You can run single test cases

```bash
python -m unittest -v \
    tests/batching/test_write_read_base.py \
    tests/batching/test_write_read_variable.py
```

Or run them individually:
```bash
python -m tests.batching.test_write_read_single_image -v
```