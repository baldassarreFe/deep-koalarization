# Testing

Running the tests together fails due to some internal tensorflow problem,
so avoid running:

```bash
python3 -m unittest -v
```

Rather specify the single test cases

```bash
python3 -m unittest -v tests/batching/test_write_read_base.py \
    tests/batching/test_write_read_variable.py
```

Or run them individually:

```bash
python3 -m tests.batching.test_write_read_single_image -v
```