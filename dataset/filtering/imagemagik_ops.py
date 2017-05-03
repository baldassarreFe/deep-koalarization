import subprocess

from PIL import Image


class ImageMagickException(Exception):
    """Raised for failed ImageMagick commands"""

    def __init__(self, command, e):
        self.output = 'The command\n{} returned and error:\n{}' \
            .format(command, e.output)

    def __str__(self):
        return self.output


def execute(command):
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise ImageMagickException(command, e) from e


def colortone(source_file, dest_file, color, level, negate=False):
    execute(
        ("convert {source} \( -clone 0 -fill '{color}' -colorize 100% \)" +
         " \( -clone 0 -colorspace gray {negate} \)" +
         " -compose blend -define compose:args={arg0},{arg1} -composite {dest}")
            .format(
            source=source_file,
            dest=dest_file,
            color=color,
            negate='-negate' if negate else '',
            arg0=level,
            arg1=100 - level)
    )


def border(source_file, dest_file, color='black', width=20):
    execute(
        "convert {source} -bordercolor {color} -border {bwidth}x{bwidth} {dest}"
            .format(
            source=source_file,
            dest=dest_file,
            color=color,
            bwidth=width
        )
    )


def vignette(source_file, dest_file, color_1='none', color_2='black',
             crop_factor=1.5):
    width, height = Image.open(source_file).size
    crop_x = int(width * crop_factor)
    crop_y = int(width * crop_factor)

    execute(
        ("convert \( {source} \) " +
         "\( -size {crop_x}x{crop_y} radial-gradient:{color_1}-{color_2} " +
         "-gravity center -crop {width}x{height}+0+0 +repage \) " +
         "-compose multiply -flatten {dest}")
            .format(
            source=source_file,
            dest=dest_file,
            width=width,
            height=height,
            crop_x=crop_x,
            crop_y=crop_y,
            color_1=color_1,
            color_2=color_2
        )
    )
