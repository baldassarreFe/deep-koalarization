# *#filters*

## Libraries

```bash
pip3 install opencv-python, PIL
```

## Imagemagik filters

These filters are accessed straight from the command line executing the corresponding
shell command. They operate on a file in place, so before using them it's important
to save the original.

## Open CV filters

These filters are applied mathematically on the image. Before applying the image
need to be loaded in memory and at the end it must be saved.

The methods using opencv filters are wrapped to they can be used as the
imagemagick ones.
