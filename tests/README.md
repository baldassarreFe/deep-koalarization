# Testing

<!-- **Note:** Running the tests together fails due to some internal TensorFlow problem, avoid running: -->

### Run all tests

You can run all tests by using

```bash
python3.6 -m unittest discover tests -v
```

> **Note:** `test_colorization.py` will train a simple model for 20 epochs, might take few seconds.

### Run specific module tests

You can run single test cases

```bash
python3.6 -m unittest -v \
    tests/batching/test_write_read_base.py \
    tests/batching/test_write_read_variable.py
```

Or run them individually:
```bash
python3.6 -m tests.batching.test_write_read_single_image -v
```