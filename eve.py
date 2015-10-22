#!/usr/bin/env python
"""
the Easy Vision Environment

EVE provides easy-to-use functionality for performing common image
processing and computer vision tasks.  The intention is for them to be
used during interactive sessions, from the Python interpreter's command
prompt or from an enhanced interpreter such as ipython as well as in
scripts.

EVE is built principally on top of the popular numpy ('numerical
python') extension to Python.  Images are represented as numpy arrays,
usually 32-bit floating-point ones, indexed by line, pixel and channel,
in that order: image[line,pixel,channel].  The choice of a
floating-point representation is deliberate: it permits images that have
been captured from sensors with more than 8 bits dynamic range to be
processed (e.g., astronomical images and digital radiographs); it
supports Fourier-space processing; and it avoids having to worry about
rounding values except at output.  Images in EVE may also contain any
number of channels, so EVE can be used with e.g. remote sensing or
hyperspectral imagery.

Other Python extensions are loaded by those routines that need them.  In
particular, PIL (the 'Python Imaging Extension') is used for the input
and output of common image file formats, though not for any processing.
scipy ('scientific python') is used by several routines, and so are a
few other extensions here and there.

On the other hand, EVE is slow.  If you're thinking of using EVE instead
of openCV for real-time video processing, forget it!  This is partly
because of the interpreted nature of Python and partly because EVE
attempts to provide algorithms that are understandable rather than fast:
it is intended as a prototyping environment rather than a real-time
delivery one.  (This also makes it useful for teaching how vision
algorithms work, of course.)  In the fullness of time, it is intended to
hook either OpenCV or dedicated C code backends for common functions
that could usefully be speeded up, and also to investigate the use of
GPUs -- but not yet.

EVE was written by Adrian F. Clark <alien@essex.ac.uk>, though several
routines are adapted from code written by others; such code is
attributed in the relevant routines.  EVE is made available entirely
freely: you are at liberty to use it in your own work, either as is or
after modification.  The author would be very happy to hear of
improvements or enhancements that you may make.
"""
from __future__ import division
import math, numpy, os, platform, re, string, struct, sys, tempfile

#-------------------------------------------------------------------------------
# Symbolic constants.
# The operating system we are running under, used to select the appropriate
# external program for display or grabbing images and a few other things.
systype = platform.system ()

tiny = 1.0e-7            # the smallest number worth bothering about
max_image_value = 255.0  # the largest value normally put into an image

character_height = 13    # height of characters in draw_text()
character_width = 10     # width of characters in draw_text()
character_bitmap = {
        ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '!': [0x00,0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18],
        '"': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x36,0x36,0x36,0x36],
        '#': [0x00,0x00,0x00,0x66,0x66,0xff,0x66,0x66,0xff,0x66,0x66,0x00,0x00],
        '$': [0x00,0x00,0x18,0x7e,0xff,0x1b,0x1f,0x7e,0xf8,0xd8,0xff,0x7e,0x18],
        '%': [0x00,0x00,0x0e,0x1b,0xdb,0x6e,0x30,0x18,0x0c,0x76,0xdb,0xd8,0x70],
        '&': [0x00,0x00,0x7f,0xc6,0xcf,0xd8,0x70,0x70,0xd8,0xcc,0xcc,0x6c,0x38],
        "'": [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x1c,0x0c,0x0e],
        '(': [0x00,0x00,0x0c,0x18,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x18,0x0c],
        ')': [0x00,0x00,0x30,0x18,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x18,0x30],
        '*': [0x00,0x00,0x00,0x00,0x99,0x5a,0x3c,0xff,0x3c,0x5a,0x99,0x00,0x00],
        '+': [0x00,0x00,0x00,0x18,0x18,0x18,0xff,0xff,0x18,0x18,0x18,0x00,0x00],
        ',': [0x00,0x00,0x30,0x18,0x1c,0x1c,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '-': [0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0x00,0x00,0x00,0x00,0x00],
        '.': [0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '/': [0x00,0x60,0x60,0x30,0x30,0x18,0x18,0x0c,0x0c,0x06,0x06,0x03,0x03],
        '0': [0x00,0x00,0x3c,0x66,0xc3,0xe3,0xf3,0xdb,0xcf,0xc7,0xc3,0x66,0x3c],
        '1': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x78,0x38,0x18],
        '2': [0x00,0x00,0xff,0xc0,0xc0,0x60,0x30,0x18,0x0c,0x06,0x03,0xe7,0x7e],
        '3': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0x7e,0x07,0x03,0x03,0xe7,0x7e],
        '4': [0x00,0x00,0x0c,0x0c,0x0c,0x0c,0x0c,0xff,0xcc,0x6c,0x3c,0x1c,0x0c],
        '5': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0xfe,0xc0,0xc0,0xc0,0xc0,0xff],
        '6': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc7,0xfe,0xc0,0xc0,0xc0,0xe7,0x7e],
        '7': [0x00,0x00,0x30,0x30,0x30,0x30,0x18,0x0c,0x06,0x03,0x03,0x03,0xff],
        '8': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xe7,0x7e,0xe7,0xc3,0xc3,0xe7,0x7e],
        '9': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x03,0x7f,0xe7,0xc3,0xc3,0xe7,0x7e],
        ':': [0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x38,0x38,0x00,0x00,0x00,0x00],
        ';': [0x00,0x00,0x30,0x18,0x1c,0x1c,0x00,0x00,0x1c,0x1c,0x00,0x00,0x00],
        '<': [0x00,0x00,0x06,0x0c,0x18,0x30,0x60,0xc0,0x60,0x30,0x18,0x0c,0x06],
        '=': [0x00,0x00,0x00,0x00,0xff,0xff,0x00,0xff,0xff,0x00,0x00,0x00,0x00],
        '>': [0x00,0x00,0x60,0x30,0x18,0x0c,0x06,0x03,0x06,0x0c,0x18,0x30,0x60],
        '?': [0x00,0x00,0x18,0x00,0x00,0x18,0x18,0x0c,0x06,0x03,0xc3,0xc3,0x7e],
        '@': [0x00,0x00,0x3f,0x60,0xcf,0xdb,0xd3,0xdd,0xc3,0x7e,0x00,0x00,0x00],
        'A': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xff,0xc3,0xc3,0xc3,0x66,0x3c,0x18],
        'B': [0x00,0x00,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'C': [0x00,0x00,0x7e,0xe7,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xe7,0x7e],
        'D': [0x00,0x00,0xfc,0xce,0xc7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc7,0xce,0xfc],
        'E': [0x00,0x00,0xff,0xc0,0xc0,0xc0,0xc0,0xfc,0xc0,0xc0,0xc0,0xc0,0xff],
        'F': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xfc,0xc0,0xc0,0xc0,0xff],
        'G': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xcf,0xc0,0xc0,0xc0,0xc0,0xe7,0x7e],
        'H': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xff,0xc3,0xc3,0xc3,0xc3,0xc3],
        'I': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7e],
        'J': [0x00,0x00,0x7c,0xee,0xc6,0x06,0x06,0x06,0x06,0x06,0x06,0x06,0x06],
        'K': [0x00,0x00,0xc3,0xc6,0xcc,0xd8,0xf0,0xe0,0xf0,0xd8,0xcc,0xc6,0xc3],
        'L': [0x00,0x00,0xff,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0],
        'M': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xdb,0xff,0xff,0xe7,0xc3],
        'N': [0x00,0x00,0xc7,0xc7,0xcf,0xcf,0xdf,0xdb,0xfb,0xf3,0xf3,0xe3,0xe3],
        'O': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xe7,0x7e],
        'P': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'Q': [0x00,0x00,0x3f,0x6e,0xdf,0xdb,0xc3,0xc3,0xc3,0xc3,0xc3,0x66,0x3c],
        'R': [0x00,0x00,0xc3,0xc6,0xcc,0xd8,0xf0,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'S': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0x7e,0xe0,0xc0,0xc0,0xe7,0x7e],
        'T': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0xff],
        'U': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3],
        'V': [0x00,0x00,0x18,0x3c,0x3c,0x66,0x66,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3],
        'W': [0x00,0x00,0xc3,0xe7,0xff,0xff,0xdb,0xdb,0xc3,0xc3,0xc3,0xc3,0xc3],
        'X': [0x00,0x00,0xc3,0x66,0x66,0x3c,0x3c,0x18,0x3c,0x3c,0x66,0x66,0xc3],
        'Y': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x3c,0x3c,0x66,0x66,0xc3],
        'Z': [0x00,0x00,0xff,0xc0,0xc0,0x60,0x30,0x7e,0x0c,0x06,0x03,0x03,0xff],
        '[': [0x00,0x00,0x3c,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3c],
       '\\': [0x00,0x03,0x03,0x06,0x06,0x0c,0x0c,0x18,0x18,0x30,0x30,0x60,0x60],
        ']': [0x00,0x00,0x3c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x3c],
        '^': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xc3,0x66,0x3c,0x18],
        '_': [0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '`': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x38,0x30,0x70],
        'a': [0x00,0x00,0x7f,0xc3,0xc3,0x7f,0x03,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'b': [0x00,0x00,0xfe,0xc3,0xc3,0xc3,0xc3,0xfe,0xc0,0xc0,0xc0,0xc0,0xc0],
        'c': [0x00,0x00,0x7e,0xc3,0xc0,0xc0,0xc0,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'd': [0x00,0x00,0x7f,0xc3,0xc3,0xc3,0xc3,0x7f,0x03,0x03,0x03,0x03,0x03],
        'e': [0x00,0x00,0x7f,0xc0,0xc0,0xfe,0xc3,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'f': [0x00,0x00,0x30,0x30,0x30,0x30,0x30,0xfc,0x30,0x30,0x30,0x33,0x1e],
        'g': [0x7e,0xc3,0x03,0x03,0x7f,0xc3,0xc3,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'h': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xfe,0xc0,0xc0,0xc0,0xc0],
        'i': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x18,0x00],
        'j': [0x38,0x6c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x00,0x00,0x0c,0x00],
        'k': [0x00,0x00,0xc6,0xcc,0xf8,0xf0,0xd8,0xcc,0xc6,0xc0,0xc0,0xc0,0xc0],
        'l': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x78],
        'm': [0x00,0x00,0xdb,0xdb,0xdb,0xdb,0xdb,0xdb,0xfe,0x00,0x00,0x00,0x00],
        'n': [0x00,0x00,0xc6,0xc6,0xc6,0xc6,0xc6,0xc6,0xfc,0x00,0x00,0x00,0x00],
        'o': [0x00,0x00,0x7c,0xc6,0xc6,0xc6,0xc6,0xc6,0x7c,0x00,0x00,0x00,0x00],
        'p': [0xc0,0xc0,0xc0,0xfe,0xc3,0xc3,0xc3,0xc3,0xfe,0x00,0x00,0x00,0x00],
        'q': [0x03,0x03,0x03,0x7f,0xc3,0xc3,0xc3,0xc3,0x7f,0x00,0x00,0x00,0x00],
        'r': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xe0,0xfe,0x00,0x00,0x00,0x00],
        's': [0x00,0x00,0xfe,0x03,0x03,0x7e,0xc0,0xc0,0x7f,0x00,0x00,0x00,0x00],
        't': [0x00,0x00,0x1c,0x36,0x30,0x30,0x30,0x30,0xfc,0x30,0x30,0x30,0x00],
        'u': [0x00,0x00,0x7e,0xc6,0xc6,0xc6,0xc6,0xc6,0xc6,0x00,0x00,0x00,0x00],
        'v': [0x00,0x00,0x18,0x3c,0x3c,0x66,0x66,0xc3,0xc3,0x00,0x00,0x00,0x00],
        'w': [0x00,0x00,0xc3,0xe7,0xff,0xdb,0xc3,0xc3,0xc3,0x00,0x00,0x00,0x00],
        'x': [0x00,0x00,0xc3,0x66,0x3c,0x18,0x3c,0x66,0xc3,0x00,0x00,0x00,0x00],
        'y': [0xc0,0x60,0x60,0x30,0x18,0x3c,0x66,0x66,0xc3,0x00,0x00,0x00,0x00],
        'z': [0x00,0x00,0xff,0x60,0x30,0x18,0x0c,0x06,0xff,0x00,0x00,0x00,0x00],
        '{': [0x00,0x00,0x0f,0x18,0x18,0x18,0x38,0xf0,0x38,0x18,0x18,0x18,0x0f],
        '|': [0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18],
        '}': [0x00,0x00,0xf0,0x18,0x18,0x18,0x1c,0x0f,0x1c,0x18,0x18,0x18,0xf0],
        '~': [0x00,0x00,0x00,0x00,0x00,0x00,0x06,0x8f,0xf1,0x60,0x00,0x00,0x00]
        }

#-------------------------------------------------------------------------------
def add_gaussian_noise (im, mean=0.0, sd=1.0, seed=None):
    """
    Add Gaussian-distributed noise to each pixel of an image.

    Arguments:
      im  the image to which noise will be added
    mean  the mean of the Gaussian-distributed noise (default: 0.0)
      sd  the standard deviation of the noise (default: 1.0)
    seed  if supplied, this is used to seed the random number generator
    """
    if not seed is None: numpy.random.seed (seed)
    im += numpy.random.normal (mean, sd, im.shape)

#-------------------------------------------------------------------------------
def annular_mean (im, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                  ahi=math.pi):
    """
    Return the mean of an annular region of an image.

    Arguments:
     im  the image to be examined
     y0  the y-value of the centre of the rotation (default: centre pixel)
     x0  the x-value of the centre of the rotation (default: centre pixel)
    rlo  the inner radius of the annular region
    rhi  the outer radius of the annular region
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)
    ave = num = 0.0
    # Cycle through the image.
    for y in xrange (0, ny):
        yy = (y - y0)**2
        for x in xrange (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            for c in xrange (0, nc):
                if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                    ave += im[y,x,c]
                    num += 1
    if num > 0: ave /= num
    return ave

#-------------------------------------------------------------------------------
def annular_set (im, v, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                 ahi=math.pi):
    """
    Set an annular region of an image.

    Arguments:
     im  the image to be set (modified)
      v  value to which the region is to be set
     y0  the y-value of the centre of the rotation (default: centre pixel)
     x0  the x-value of the centre of the rotation (default: centre pixel)
    rlo  the inner radius of the annular region
    rhi  the outer radius of the annular region
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)
    # Cycle through the image.
    for y in xrange (0, ny):
        yy = (y - y0)**2
        for x in xrange (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                im[y,x] = v

#-------------------------------------------------------------------------------
def ascii_art (im, using=["*       ", "@#+-    ", "#XXXX/' "], fd=sys.stdout,
               ff=False, aspect_ratio=1.95, width=132, border="tblr",
               reverse=False, limits=None):
    """
    Output an image as characters, optionally with overprinting.

    Arguments:
              im  image to be printed
           using  list of characters, each defining one layers of
                  overprinting in order of decreasing blackness
                  (default: three layers ["*       ", "@#+-    ", "#8XXX/' "])
             fd   file on which the output is written (default: sys.stdout)
             ff   if True, output a form-feed before printing the image
                  (default: false)
    aspect_ratio  ratio of character height to width (default: 1.95)
           width  number of characters in each line of output (default: 132)
          border  how the image should have its border drawn, a string of
                  characters from 'news' or 'tblr'
         reverse  if True, reverse the contrast (default: False)
          limits  if supplied, a list comprising the minimum and maximum 
                  image values that should be used for determining the
                  mapping onto characters; the limits between the image
                  should be clipped
    """
    # If using is a string, we don't overprint.  A good set of characters in
    # that case is using="#@X+/' " for terminal windows with a light
    # background.  If using is a list, we assume it's a list of strings,
    # giving the different levels of overprinting.  In that case, the default
    # set looks OK.
    chars = []
    if isinstance (using, str):
        nover = 1
        chars.append(using)
    elif isinstance (using, list):
        chars = using
        nover = len(using)
    else:
        raise ValueError, 'Illegal argument type'
    nvals = len (chars[0])

    # Work out how many pixels we're to print across the output.
    (ny, nx, nc) = sizes (im)
    if nx < width: xmax = nx
    else:          xmax = width - 2
    xinc = nx / xmax
    yinc = xinc * aspect_ratio
    ymax = int (ny / yinc + 0.5)

    # Work out the grey-level scaling factor.
    if limits is None: lo, hi = extrema (im)
    else:              lo, hi = limits
    if hi == lo: hi += 1
    fac = (nvals - 1) / (hi - lo)

    # Decide which borders we're to print.  If we'r to print the top or
    # bottom border, work it out.
    doN = doE = doS = doW = False
    if string.find (border, 'n') >= 0 or string.find (border, 't') >= 0:
        doN = True
    if string.find (border, 'e') >= 0 or string.find (border, 'r') >= 0:
        doE = True
    if string.find (border, 's') >= 0 or string.find (border, 'b') >= 0:
        doS = True
    if string.find (border, 'w') >= 0 or string.find (border, 'l') >= 0:
        doW = True
    if doN or doS:
        sep = '+'
        for x in xrange (0, xmax):
            if x % 5 == 4: sep += '+'
            else:          sep += '-'
        sep += '+'

    # Arrange to print a form-feed, if necessary, and print the top border.
    if ff: ffc = ''
    else:  ffc = ''
    if doN:
        print >>fd, ffc + sep
        ffc = ''

    # Print the image.
    buf = numpy.zeros (xmax, int)
    fy = 0
    for y in xrange (0, ymax):
        dy = fy - int(fy)
        dy1 = 1.0 - dy
        ylo = int(fy) % ny
        yhi = (ylo + 1) % ny
        ib = -1
        # For each pixel along a line, average all the channels to one and
        # scale the value into the appropriate number of levels.
        fx = 0
        for x in xrange (0, xmax):
            dx = fx - int(fx)
            dx1 = 1.0 - dx
            xlo = int(fx) % nx
            xhi = (xlo + 1) % nx
            v = 0
            for c in xrange (0, nc):
                vc = dx1 * dy1 * im[ylo,xlo,c] + \
                     dx  * dy1 * im[ylo,xhi,c] + \
                     dx1 * dy  * im[yhi,xlo,c] + \
                     dx  * dy  * im[yhi,xhi,c]
                v += vc
            v /= nc
            v = int ((v - lo) * fac + 0.5)
            if v < 0:      v = 0
            if v >= nvals: v = nvals - 1
            ib += 1
            buf[ib] = v
            fx += xinc
        fy += yinc

        # Print the line, including the borders if appropriate.  The traditional
        # way of doing this is as a series of lines using the carriage return
        # character '\r' so that subsequent lines over-print the first; but '\r'
        # is the end-of-line delimiter on Macintoshes, which confuses things.
        # So we have to backspace after each character in order to over-print.
        # And so technology moves on...
        if doW or doE:
            if y % 5 == 4: mark = '+'
            else:          mark = '|'
        else:              mark = ' '
        line = ' '
        if doW: line = mark
        for x in xrange (0, len(buf)):
            for ov in xrange (0, nover-1):
                line += chars[ov][buf[x]] + '\b'
            line += chars[nover-1][buf[x]]
        if doE: line += mark
        print >>fd, ffc + line
        ffc = ''

    # Print the bottom border.
    if doS: print >>fd, sep

#-------------------------------------------------------------------------------
def binarize (im, threshold, bg=0.0, fg=max_image_value):
    """
    Binarize an image, returning the result.

    Arguments:
           im  image to be binarized
    threshold  the threshold to be used for binarization
           bg  value to which pixels at or below the threshold will be set
               (default: 0.0)
           fg  value to which pixels equal to or above the threshold will
               be set (default: 255.0)
    """
    bim = image (im)
    set (bim, bg)
    bim[numpy.where (im >= threshold)] = fg
    return bim

#-------------------------------------------------------------------------------
def blend_pixel (im, y, x, v, opac):
    """
    Blend the value v into the pixel im[y,x] according to the opacity opac.

    Arguments:
      im  image in which the pixel is drawn (modified)
       y  y-position of the pixel to be modified
       x  x-position of the pixel to be modified
       v  new value to which the pixel is to be set
    opac  opacity with which the value will be drawn into the pixel
    """
    ny, nx, nc = sizes (im)
    if y >= 0 and y < ny and x >= 0 and x < nx:
        if not isinstance (v, list): v = [v] * nc
        for c in xrange (0, nc):
            im[y,x,c] = v[c] * opac + (1.0 - opac) * im[y,x,c]

#-------------------------------------------------------------------------------
def canny (im, lo, hi):
    """
    Perform edge detection in im using the Canny operator.

    Three EVE images are returned: the first contains the gradient
    magnitudes at each pixel; the second contains the gradient
    magnitudes after non-maximum supression and the third contains the
    final edges after hysteresis thresholding.

    Arguments:
    im  image in which the edges are to be found
    lo  threshold below which edge segments are discarded
    hi  threshold above which edge segments are definitely edges


    The original implementation of this operator was by Zachary Pincus
    <zachary.pincus@yale.edu>, adapted to work with EVE-format images.

    """
    import scipy
    import scipy.ndimage as ndimage

    # Convert the EVE-format image into one compatible with scipy.
    ny, nx, nc = sizes (im)
    if nc == 1: sci_im = im[:,:,0]
    else:       sci_im = mono(im)[:,:,0]

    # The following filter kernels are for calculating the value of
    # neighbours in the required directions.
    _N  = scipy.array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 1, 0]], dtype=bool)
    _NE = scipy.array([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 0]], dtype=bool)
    _W  = scipy.array([[0, 0, 0],
                       [1, 0, 1],
                       [0, 0, 0]], dtype=bool)
    _NW = scipy.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1]], dtype=bool)

    # After quantizing the orientations of gradients, vertical
    # (north-south) edges get values of 3, northwest-southeast edges get
    # values of 2, and so on, as below.
    _NE_d = 0
    _W_d = 1
    _NW_d = 2
    _N_d = 3
    grad_x = ndimage.sobel(sci_im, 0)
    grad_y = ndimage.sobel(sci_im, 1)
    grad_mag = scipy.sqrt(grad_x**2+grad_y**2)
    grad_angle = scipy.arctan2(grad_y, grad_x)

    # Scale the angles in the range [0,3] and then round to quantize.
    quantized_angle = scipy.around(3 * (grad_angle + numpy.pi) / (numpy.pi * 2))

    # Perform non-maximal suppression.  An edge pixel is only good if
    # its magnitude is greater than its neighbours normal to the edge
    # direction.  We quantize the edge direction into four angles, so we
    # only need to look at four sets of neighbours.
    NE = ndimage.maximum_filter(grad_mag, footprint=_NE)
    W  = ndimage.maximum_filter(grad_mag, footprint=_W)
    NW = ndimage.maximum_filter(grad_mag, footprint=_NW)
    N  = ndimage.maximum_filter(grad_mag, footprint=_N)
    thinned = (((grad_mag > W)  & (quantized_angle == _N_d )) |
               ((grad_mag > N)  & (quantized_angle == _W_d )) |
               ((grad_mag > NW) & (quantized_angle == _NE_d)) |
               ((grad_mag > NE) & (quantized_angle == _NW_d)) )
    thinned_grad = thinned * grad_mag

    # Perform hysteresis thresholding: find seeds above thr high
    # threshold, then expand out until the line segment goes below the
    # low threshold.
    high = thinned_grad > hi
    low = thinned_grad > lo
    canny_edges = ndimage.binary_dilation(high, structure=scipy.ones((3,3)),
                                          iterations=-1, mask=low)

    # Convert the results back to EVE-format images and return them.
    gm = image ((ny,nx,1))
    tm = image ((ny,nx,1))
    ce = image ((ny,nx,1))
    gm[:,:,0] = grad_mag[:,:]
    tm[:,:,0] = thinned_grad[:,:]
    ce[:,:,0] = canny_edges[:,:] * max_image_value
    return gm, tm, ce

#-------------------------------------------------------------------------------
def centroid (im, c=0):
    """
    Return the centroid of a channel of an image.

    This routine is normally used on a binarized image (see binarize())
    after labelling (see label_regions() and labelled_region()) to
    locate the centres of regions.

    Arguments:
    im  image for which the centroid is to be found
     c  channel to be examined (default: 0)
    """
    m00 = m01 = m10 = 0.0
    ny, nx, nc = sizes (im)
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            m00 += im[y,x,c]
            m10 += im[y,x,c] * y
            m01 += im[y,x,c] * x
    if m00 < tiny:
        y = ny / 2.0
        x = nx / 2.0
    else:
        y = m10 / m00
        x = m01 / m00
    return [y, x]

#-------------------------------------------------------------------------------
def clip (im, lo, hi):
    """
    Ensure all pixels in an image are in the range lo to hi.

    Arguments:
    im  the image to be clipped (modified)
    lo  the lowest value to be in the image after clipping
    hi  the highest value to be in the image after clipping
    """
    numpy.clip (im, lo, hi, out=im)

#-------------------------------------------------------------------------------
def compare (im1, im2, tol=tiny, report=20, indent='  ', fd=sys.stdout):
    """
    Compare two images, reporting up to report pixels that differ.
    The routine returns the number of differences found.

    Arguments:
       im1  image to be compared with im2
       im2  image to be compared with im1
       tol  minimum amount by which pixels must differ (default: eve.tiny)
    report  the maximum number of differences reported (default: 20)
            (the presence of further differences is indicated by '...')
    indent  indentation output before a difference (default: '  ')
        fd  file on which the output is to be written (default: sys.stdout)
    """
    ny, nx, nc = sizes (im1)
    ndiffs = 0
    diffs = []
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            for c in xrange (0, nc):
                if abs (im1[y,x,c] - im2[y,x,c]) > tol:
                    ndiffs += 1
                    if ndiffs <= report:
                        diffs.append ([y,x,c])
    if ndiffs > 0 and report > 0:
        print >>fd, ndiffs, 'differences found:'
        for d in xrange (0, len(diffs)):
            y,x,c = diffs[d]
            print >>fd, indent, y,x,c, '->', im1[y,x,c], '&', im2[y,x,c]
        if ndiffs > report: print >>fd, indent, '...'
    return ndiffs

#-------------------------------------------------------------------------------
def contrast_stretch (im, low=0.0, high=max_image_value):
    """
    Stretch the contrast in the image to the supplied low and high values.

    Arguments:
      im  image whose contrast is to be stretched (modified)
     low  new value to which the lowest value in im is to be scaled
          (default: 0.0)
    high  new value to which the highest value in im is to be scaled
          (default: 255.0)
    """
    oldmin, oldmax = extrema (im)
    fac = (high - low) / (oldmax - oldmin)
    # For some reason, the following line doesn't work but the subsequent
    # three do!
    # im = (im - oldmin) * fac + low
    im -= oldmin
    im *= fac
    im += low

#-------------------------------------------------------------------------------
def convolve (im, mask, statistic='sum'):
    """
    Perform a convolution of im with mask, returning the result.

    Arguments:
           im  the image to be convolved with mask (modified)
         mask  the convolution mask to be used
    statistic  one of:
                  sum  conventional convolution
                 mean  conventional convolution
               median  median filtering
                  min  grey-scale shrink (reduces light areas)
                  max  grey-scale expand (enlarges light areas)
    """
    ny, nx, nc = sizes (im)
    my, mx, mc = sizes (mask)
    yo = my // 2
    xo = mx // 2

    # Create an output image of the same size as the input.
    result = image (im)

    # We need a special case for 'min' statistic to erase the mask elements
    # that are zero.
    nzeros = len ([x for x in mask.ravel() if x == 0])

    # Loop over the pixels in the image.  For each pixel position, multiply
    # the region around it with the mask, summing the elements and storing
    # that in the equivalent pixel of the output image.
    v = numpy.zeros ((my*mx*mc))
    vi = 0
    for yi in xrange (0, ny):
        for xi in xrange (0, nx):
            for ym in xrange (0, my):
                yy = (ym + yi - yo) % ny
                for xm in xrange (0, mx):
                    xx = (xm + xi - xo) % nx
                    v[vi] = im[yy,xx,0] * mask[ym,xm,0]
                    vi += 1
            if   statistic == 'sum':    ave = numpy.sum (v)
            elif statistic == 'mean':   ave = numpy.mean (v)
            elif statistic == 'max':    ave = numpy.max (v)
            elif statistic == 'min':    ave = numpy.min (v[nzeros:])
            elif statistic == 'median': ave = numpy.median (v)
            result[yi,xi,0] = ave
            vi = 0
    return result

#-------------------------------------------------------------------------------
def copy (im):
    """
    Copy the pixels from image 'im' into a new image, which is returned.

    Arguments:
    im  the image to be copied
    """
    return im.copy ()

#-------------------------------------------------------------------------------
def correlate (im1, im2):
    """
    Return the unnormalized Fourier correlation surface between two images.

    Arguments:
    im1  image to be correlated with im2
    im2  image to be correlated with im1
    """
    # Calculate the normalization factor (which doesn't appear to be right).
    v1 = sd (im1)**2
    v2 = sd (im2)**2
    fac = math.sqrt (v1 * v2)
    # Transform, invert one, multiply, invert, shift peaks to the right
    # place, normalize, and return the result.
    temp1 = numpy.fft.fft2 (im1, axes=(-3,-2))
    temp2 = numpy.fft.fft2 (im2, axes=(-3,-2))
    reflect_horizontally (temp2)
    reflect_vertically (temp2)
    temp2 *= temp1
    temp1 = numpy.fft.ifft2 (temp2, axes=(-3,-2))
    temp1 = numpy.fft.fftshift (temp1, axes=(-3,-2))
    temp1 /= fac
    return temp1

#-------------------------------------------------------------------------------
def correlation_coefficient (im1, im2):
    """
    Return the correlation coefficient between two images.

    Arguments:
    im1  image to be correlated with im2
    im2  image to be correlated with im1
    """
    ny, nx, nc = sizes (im1)
    sumx = sumy = sumxx = sumyy = sumxy = 0.0
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            for c in xrange (0, nc):
                v1 = im1[y,x,c]
                v2 = im2[y,x,c]
                sumx += v1
                sumy += v2
                sumxx += v1 * v1
                sumxy += v1 * v2
                sumyy += v2 * v2
    n = ny * nx * nc
    v1 = sumxy - sumx * sumy / n
    v2 = math.sqrt((sumxx-sumx*sumx/n) * (sumyy-sumy*sumy/n))
    return v1 / v2

#-------------------------------------------------------------------------------
def covariance (im):
    """
    Return the covariance matrix and means of the channels of im.

    Arguments:
    im  image for which the covariance matrix is to be calculated
    """
    ny, nx, nc = sizes (im)
    covmat = numpy.ndarray ((nc, nc))
    ave = numpy.ndarray ((nc))
    for c in xrange (0, nc):
        ch = get_channel (im, c)
        ave[c] = mean (ch)
    for c1 in xrange (0, nc):
        ch1 = get_channel (im, c1)
        for c2 in xrange (0, c1+1):
            ch2 = get_channel (im, c2)
            covmat[c1,c2] = ((ch1 - ave[c1]) * (ch2 - ave[c2])).mean()
            covmat[c2,c1] = covmat[c1,c2]
    return covmat, ave

#-------------------------------------------------------------------------------
def cumulative_histogram (im, bins=64, limits=None, disp=False):
    """
    Find the cumulative histogram of an image.

    Arguments:
        im  image for which the cumulative histogram is to be found
      bins  number of bins in the histogram (default: 64)
    limits  extrema between which the histogram is to be found
      plot  when True, the histogram will be drawn
    """
    a, h = histogram (im, bins=bins, limits=limits, disp=False)
    h = h.cumsum()
    if disp: graph (a, h, 'Cumulative histogram', 'bin', 'number of pixels',
                    style='histogram')
    return a, h

#-------------------------------------------------------------------------------
def display (im, stretch=False, program=None, wait=False, name="EVE image",
             hint=False):
    """
    Display an image using an external program.

    Arguments:
         im  image to be displayed
    stretch  if True, displayed image will be contrast-stretched
    program  external program to be used for display (default: system-dependent)
       wait  when True, allow the display program to exit before returning
    """
    if systype == 'Windows':
        if stretch:
            copy = im.copy()
            contrast_stretch (copy)
        else:
            copy = im
        output_pil (copy, '', 'display')    # temporary kludge
        return

        # According to the website (spread over two lines of comments here):
        #   http://www.velocityreviews.com/forums/
        #     t707158-python-pil-and-vista-windows-7-show-not-working.html
        # Windows Vista needs the following workaround for image display via
        # Pil to work properly:
        # Edit (e.g., with TextPad) the file
        #   C:\Python26\lib\site-packages\PIL\ImageShow.py
        # Around line 99, edit the existing line to include a ping command,
        # as follows (spread over two lines of comments here):
        #   return "start /wait %s && PING 127.0.0.1 -n 5
        #      > NUL && del /f %s" % (file, file)
        if program is None: program = 'mspaint'
        handle, fn = tempfile.mkstemp (suffix='.bmp')
        output_bmp (copy, fn)
        if hint: print >>sys.stderr, \
           'Type "Control-q" in the image window to close it.'
        line = "%s %s && ping 127.0.0.1 -n 15 > NUL && del /f %s" % \
               (program, fn, fn)
        if wait: line = 'start /wait ' + line
        else: line = 'start /wait' + line   # must wait on Windows, I think
        os.system (line)
    else:
        if program is None:
            if find_in_path ('xv'):
                program = 'xv -name "' + name + '"'
                if hint: print >>sys.stderr, \
                   'Type "q" in the image window to close it.'
            elif find_in_path ('display'):
                program = 'display'
                if hint: print >>sys.stderr, \
                   'Type "q" in the image window to close it.'
            elif systype == 'Darwin':
                if stretch:
                    copy = im.copy()
                    contrast_stretch (copy)
                else:
                    copy = im
                handle, fn = tempfile.mkstemp (suffix='.png')
                output_png (copy, fn)
                if hint: print >>sys.stderr, \
                   'Type "Command-q" in the image window to close it.'
                line = "%s '%s'; sleep 5; rm -f '%s'"
                if not wait: line = "(" + line + ")&"
                line = line % ("open -a /Applications/Preview.app", fn, fn)
                os.system (line)
                return
            else:
                raise ValueError, 'Cannot find an image display program'
        handle, fn = tempfile.mkstemp ()
        output_pnm (im, fn, stretch=stretch)
        if wait:    line = "%s %s; rm -f %s"    % (program, fn, fn)
        else:       line = "(%s %s; rm -f %s)&" % (program, fn, fn)
        os.system (line)
        os.close (handle)

#-------------------------------------------------------------------------------
def draw_border (im, v=max_image_value, width=2):
    """
    Draw a border around an image.

    Arguments:
       im  image to which the border is to be added (modified)
        v  value to which the border is to be set (default: 255)
    width  width of the border in pixels (default: 2)
    """
    ny, nx, nc = sizes (im)
    im[0:width,:,:] = v       # top
    im[ny-width-1:ny,:,:] = v #  bottom
    im[:,0:width,:] = v       # left
    im[:,nx-width-1:nx,:] = v # right

#-------------------------------------------------------------------------------
def draw_box (im, ylo, xlo, yhi, xhi, border=max_image_value, fill=None):
    """
    Draw a rectangular box, optionally filled.

    Arguments:
        im  image in which the box is to be drawn (modified)
       ylo  y-value of the lower left corner of the box
       xlo  x-value of the lower left corner of the box
       yhi  y-value of the upper right corner of the box
       xhi  x-value of the upper right corner of the box
    border  value used for drawing the box (default: 255.0)
      fill  value with which the inside of the box is to be filled
            (default: None, and so is unfilled)
    """
    draw_line_fast (im, ylo, xlo, ylo, xhi, border)
    draw_line_fast (im, ylo, xhi, yhi, xhi, border)
    draw_line_fast (im, yhi, xhi, yhi, xlo, border)
    draw_line_fast (im, yhi, xlo, ylo, xlo, border)
    if not fill is None: set_region (im, ylo+1, xlo+1, yhi, xhi, fill)

#-------------------------------------------------------------------------------
def draw_circle (im, yc, xc, r, v, fill=None):
    """
    Draw a circle with value v of radius r centred on (xc, yc) in image im.

    Arguments:
      im  image upon which the circle is to be drawn (modified)
      yc  y-value (row) of the centre of the circle
      xc  x-value (column) of the centre of the circle
       r  radius of he circle in pixels
       v  value to which pixels forming the circle are set
    fill  value with which the inside of the circle is to be filled
          (default: None, and so is unfilled)

    The circle is anti-aliased, using an algorithm due to Xiaolin Wu
    (published in "Graphics Gems II").  Although fairly fast, it is slower
    than Bresenham's algorithm, used in the routine draw_circle_fast, and
    paints a range of values into the image; if it is important that all
    pixels of the line have the value v, use draw_line_fast.
    The implementation was adapted from the PHP code in
    http://mapidev.blogspot.com/2009/03/xiaolin-wu-circle-php-implementation.html
    """
    x = xx = r
    y = yy = -1
    t = 0
    while x > y:
        y += 1
        d = math.sqrt (r**2 - y**2)
        opac = int (d + 0.5) - d
        if opac < t: x -= 1
        trans = 1.0 - opac
        im[yc+y, xc+x] = v
        blend_pixel (im, y + yc, x + xc - 1, v, trans)
        blend_pixel (im, y + yc, x + xc + 1, v, opac)
        im[yc+x, xc+y] = v
        blend_pixel (im, x + yc - 1, y + xc, v, trans)
        blend_pixel (im, x + yc + 1, y + xc, v, opac)
        im[yc+y, xc-x] = v
        blend_pixel (im, y + yc, xc - x + 1, v, trans)
        blend_pixel (im, y + yc, xc - x - 1, v, opac)
        im[yc+x, xc-y] = v
        blend_pixel (im, x + yc - 1, xc - y, v, trans)
        blend_pixel (im, x + yc + 1, xc - y, v, opac)
        im[yc-y, xc+x] = v
        blend_pixel (im, yc - y, x + xc - 1, v, trans)
        blend_pixel (im, yc - y, x + xc + 1, v, opac)
        im[yc-x, xc+y] = v
        blend_pixel (im, yc - x - 1, y + xc, v, opac)
        blend_pixel (im, yc - x + 1, y + xc, v, trans)
        im[yc-x, xc-y] = v
        blend_pixel (im, yc - x - 1, xc - y, v, opac)
        blend_pixel (im, yc - x + 1, xc - y, v, trans)
        im[yc-y, xc-x] = v
        blend_pixel (im, yc - y, xc - x - 1, v, opac)
        blend_pixel (im, yc - y, xc - x + 1, v, trans)
        t = opac
    if not fill is None: fill_outline (im, yc, yc, v)

#-------------------------------------------------------------------------------
def draw_circle_fast (im, yc, xc, r, v, fill=None):
    """
    Draw a circle with value v of radius r centred on (xc, yc) in image im.

    Arguments:
      im  image upon which the circle is to be drawn (modified)
      yc  y-value (row) of the centre of the circle
      xc  x-value (column) of the centre of the circle
       r  radius of he circle in pixels
       v  value to which pixels forming the circle are set
    fill  value with which the inside of the circle is to be filled
          (default: None, and so is unfilled)
    """
    x = 0
    y = r
    p = 3 - 2 * r
    ny, nx, nc = sizes (im)
    while x < y:
        im[yc+y,xc+x] = v
        im[yc+y,xc-x] = v
        im[yc-y,xc+x] = v
        im[yc-y,xc-x] = v
        im[yc+x,xc+y] = v
        im[yc+x,xc-y] = v
        im[yc-x,xc+y] = v
        im[yc-x,xc-y] = v
        if p < 0:
            p += 4 * x + 6
        else:
            p += 4 * (x - y) + 6
            y -= 1
        x += 1
    if x == y:
        im[yc+y,xc+x] = v
        im[yc+y,xc-x] = v
        im[yc-y,xc+x] = v
        im[yc-y,xc-x] = v
        im[yc+x,xc+y] = v
        im[yc+x,xc-y] = v
        im[yc-x,xc+y] = v
        im[yc-x,xc-y] = v
    if not fill is None: fill_outline (im, yc, xc, v)

#-------------------------------------------------------------------------------
def draw_line (im, y0, x0, y1, x1, v):
    """
    Draw a line from (x0, y0) to (x1, y1) with value v in image im.

    Arguments:
    im  image upon which the line is to be drawn (modified)
    y0  y-value (row) of the start of the line
    x0  x-value (column) of the start of the line
    y1  y-value (row) of the end of the line
    x1  x-value (column) of the end of the line
     v  value to which pixels on the line are to be set

    The line is anti-aliased, using an algorithm due to Xiaolin Wu ("An
    Efficient Antialiasing Technique", Computer Graphics July 1991); the
    code is a corrected version of that in the relevant Wikipedia entry.
    The algorithm draws pairs of pixels straddling the line, coloured
    according to proximity; pixels at the line ends are handled separately.
    Although fairly fast, it is slower than Bresenham's algorithm and
    paints a range of values into the image; if it is important that all
    pixels of the line have the value v, there is a separate routine,
    draw_line_fast, which implements Bresnham's algorithm.
    """
    if abs(y1 - y0) > abs(x1 - x0): steep = True
    else:                           steep = False
    if steep:
        y0, x0 = x0, y0
        y1, x1 = x1, y1
    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1
    dx = x1 - x0 + 0.0
    dy = y1 - y0
    if dx == 0.0: de = 1.0e30
    else:         de = dy / dx

    # Handle the first end-point.
    xend = int (x0 + 0.5)
    yend = y0 + de * (xend - x0)
    xgap = 1.0 - (x0 + 0.5 - int(x0 + 0.5))
    xpxl1 = int (xend)  # this will be used in the main loop
    ypxl1 = int (yend)
    if steep:
        blend_pixel (im, xpxl1, ypxl1,   v, 1.0 - (yend - int(yend)))
        blend_pixel (im, xpxl1, ypxl1+1, v, yend - int(yend))
    else:
        blend_pixel (im, ypxl1,   xpxl1, v, 1.0 - (yend - int(yend)))
        blend_pixel (im, ypxl1+1, xpxl1, v, yend - int(yend))
    intery = yend + de  # first y-intersection for the main loop

    # Handle the second end-point.
    xend = int (x1 + 0.5)
    yend = y1 + de * (xend - x1)
    xgap = x1 + 0.5 - int(x1 + 0.5)
    xpxl2 = int (xend)  # this will be used in the main loop
    ypxl2 = int (yend)
    if steep:
        blend_pixel (im, xpxl2, ypxl2,   v, 1.0 - (yend - int (yend)))
        blend_pixel (im, xpxl2, ypxl2+1, v, yend - int(yend))
    else:
        blend_pixel (im, ypxl2,   xpxl2, v, 1.0 - (yend - int (yend)))
        blend_pixel (im, ypxl2+1, xpxl2, v, yend - int(yend))

    # The main loop.
    for x in xrange (xpxl1+1, xpxl2):
        if steep:
            blend_pixel (im, x, int (intery),   v,
                         math.sqrt(1.0 - (intery - int(intery))))
            blend_pixel (im, x, int (intery)+1, v,
                         math.sqrt (intery - int(intery)))
        else:
            blend_pixel (im, int (intery),   x, v,
                         math.sqrt(1.0 - (intery - int(intery))))
            blend_pixel (im, int (intery)+1, x, v,
                         math.sqrt(intery - int(intery)))
        intery += de

#-------------------------------------------------------------------------------
def draw_line_fast (im, y0, x0, y1, x1, v):
    """
    Draw a line from (x0, y0) to (x1, y1) with value v in image im.

    Arguments:
    im  image upon which the line is to be drawn (modified)
    y0  y-value (row) of the start of the line
    x0  x-value (column) of the start of the line
    y1  y-value (row) of the end of the line
    x1  x-value (column) of the end of the line
     v  value to which pixels on the line are to be set

    This routine uses the classic line-drawing due to Bresenham, which
    aliases badly for most lines; if appearance is more important than
    speed, there is a separate EVE routine that implements anti-aliased
    line-drawing using an algorithm due to Xiaolin Wu.
    """
    ny, nx, nc = sizes (im)
    y0 = int (y0)
    x0 = int (x0)
    y1 = int (y1)
    x1 = int (x1)
    if abs(y1 - y0) > abs(x1 - x0): steep = True
    else:                           steep = False
    if steep:
        y0, x0 = x0, y0
        y1, x1 = x1, y1
    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1
    dx = x1 - x0 + 0.0
    dy = abs(y1 - y0)
    e = 0.0
    if dx == 0.0: de = 1.0e30
    else:         de = dy / dx
    y = y0
    if y0 < y1: ystep =  1
    else:       ystep = -1
    for x in xrange (x0,x1+1):
        if steep:
            if x >= 0 and x < ny and y >= 0 and y < nx: im[x,y,:] = v
        else:
            if x >= 0 and x < nx and y >= 0 and y < ny: im[y,x,:] = v
        e += de
        if e >= 0.5:
            y += ystep
            e -= 1.0

#-------------------------------------------------------------------------------
def draw_polygon (im, yc, xc, r, nsides, v=max_image_value, fast=False,
                  fill=False):
    """
    Draw an nsides-sided polygon of radius r centred at (yc, xc), returning
    a list of its vertices.

    Arguments:
        im  image upon which the text is to be written (modified)
        yc  y-value (row) of the centre of the polygon
        xc  x-value (column) of the centre of the polygon
    radius  radius of the circle in which the polygon is enclosed
    nsides  number of sides that the polygon is to have
         v  value to which pixels are to be set (default: 255.0)
      fast  if True, don't use anti-aliased lines (default: False)
    """
    angle = 2.0 * math.pi / nsides
    y0 = yc
    x0 = xc + r
    vertices = []
    for i in xrange (1, nsides+1):
        vertices.append ((y0, x0))
        y1 = yc + r * math.sin (i * angle)
        x1 = xc + r * math.cos (i * angle)
        if fast: draw_line_fast (im, y0, x0, y1, x1, v)
        else: draw_line (im, y0, x0, y1, x1, v)
        y0 = y1
        x0 = x1
    if fill: fill_outline (im, yc, xc, v)
    return vertices

#-------------------------------------------------------------------------------
def draw_star (im, yc, xc, radius, npoints, inner_radius=None,
               v=max_image_value, fast=False, fill=False):
    """
    Draw an npoints-pointed star of radius r centred at (yc, xc).

    Arguments:
              im  image upon which the text is to be written (modified)
              yc  y-value (row) of the centre of the star
              xc  x-value (column) of the centre of the star
          radius  radius of the circle in which the polygon is enclosed
         npoints  number of points that the star is to have
    inner_radius  radius of the inner parts of the star
               v  value to which pixels are to be set (default: 255.0)
            fast  if True, don't use anti-aliased lines (default: False)
    """
    angle = math.pi / npoints
    if inner_radius is None: inner_radius = radius / 2
    y0 = yc
    x0 = xc + radius
    np = 2 * npoints + 1
    vertices = []
    for i in xrange (1, np):
        vertices.append ((y0, x0))
        if (i // 2) * 2 == i: r = radius
        else:                 r = inner_radius
        y1 = yc + r * math.sin (i * angle)
        x1 = xc + r * math.cos (i * angle)
        if fast: draw_line_fast (im, y0, x0, y1, x1, v)
        else: draw_line (im, y0, x0, y1, x1, v)
        y0 = y1
        x0 = x1
    if fill: fill_outline (im, yc, xc, v)
    return vertices

#-------------------------------------------------------------------------------
def draw_text (im, text, y, x, v=max_image_value, align="c"):
    """
    Write text onto an image.

    Arguments:
       im  image upon which the text is to be written (modified)
     text  string of characters to be written onto the image
        y  y-value (row) at which the text is to be written
        x  x-value (column) at which the text is to be written
        v  value to which pixels in the text are to be set (default: 255.0)
    align  alignment of the text, one of (default: 'c')
           'c': centred
           'l': left-justified
           'r': right-justified

    This routine is based on C code kindly provided by Nick Glazzard of
    Speedsix.
    """
    global character_height, character_width, character_bitmap

    # Work out the start position on the image depending on the text alignment.
    if    align == 'l' or align == 'L':
         offset = 0
    elif  align == 'r' or align == 'R':
        offset = - character_width * len(text)
    else:
        offset = - character_width * len(text) // 2

    # Draw each character in turn.
    ny, nx, nc = sizes (im)
    for c in text:
        for row in xrange(character_height-1,-1,-1):
            yy = y-row
            if yy >= 0 and yy < ny:
                b = character_bitmap[c][row]
                for col in xrange (character_width-1,-1,-1):
                    if b & (1<<col):
                        xx = x+(7-col)+offset
                        if xx >= 0 and xx < nx:
                            im[yy,xx,:] = v
        offset += character_width

#-------------------------------------------------------------------------------
def examine (im, name="", format="%3.0f", lformat=None, ff=False, fd=sys.stdout,
             ylo=0, xlo=0, yhi=None, xhi=None, clo=0, chi=None):
    """
    Output an image in human-readable form.

    Arguments:
         im  image whose pixels are to be output
       name  name of the image (default : '')
     format  format used for writing pixels (default: '%3.0f')
    lformat  format used for column and row numbers (default: contextual)
         ff  if True, output a form-feed before the output (default: False)
         fd  file on which output is to be written (default: sys.stdout)
        ylo  lower y-value of the region to be output (default: 0)
        xlo  lower x-value of the region to be output (default: 0)
        yhi  upper y-value of the region to be output (default: last pixel)
        xhi  upper x-value of the region to be output (default: last pixel)
        clo  first channel of the region to be output (default: 0)
        chi  last channel of the region to be output (default: last channel)
         ff  if True, output a form feed before the image (default: False)
    """
    # Work out the width of a printed pixel by setting "0".  We use that to
    # determine lformat, unless set explicitly by the caller.
    width = len (format % 0.0)
    if lformat is None: lformat = "%%%dd" % width

    # Print the introduction.
    ny, nx, nc = sizes (im)
    py = px = pc = "s"
    if ny == 1: py = ""
    if nx == 1: px = ""
    if nc == 1: pc = ""
    if ff: ffc = ''
    else:  ffc = ''
    if name != "": name += " "
    print >>fd, ffc + \
        ("Image %sis %d line%s, %d pixel%s/line, %d channel%s ") \
        % (name, ny, py, nx, px, nc, pc)

    # Print the element numbers across the top and a line.
    if yhi is None: yhi = ny
    if xhi is None: xhi = nx
    if chi is None: chi = nc
    nyl = yhi - ylo
    nxl = xhi - xlo
    sep = ' ' * width + '+' + '-' * (width+1) * nxl + '-+'
    print >>fd, ' ' * (width+1),
    for x in xrange (xlo, xhi):
        print >>fd, lformat % (x),
    print >>fd, ""
    print >>fd, sep

    # Print the pixels of the rows, with the channels above each other.
    for y in xrange (ylo, yhi):
        for c in xrange (clo, chi):
            if c == clo:
                print >>fd, lformat % (y) + '|',
            else:
                print >>fd, len (lformat % (y)) *  ' ' + '|',
            for x in xrange (xlo, xhi):
                print >>fd, format % (im[y,x,c]),
            print >>fd, "|"
        print >>fd, sep

#-------------------------------------------------------------------------------
def effect_drawing (im, blursize=17, opacity=0.9):
    """
    Convert an image into a 'drawing' and return it.

    Arguments:
         im  image to be converted into a 'drawing'
       blur  size of the square mask to be used for blurring the image
             (default: 17)
    opacity  the opacity to be used when blending the blur with
             the original (default: 0.9)
    """
    ny, nx, nc = sizes (im)
    if nc > 1: im1 = mono (im)
    else:      im1 = copy (im)
    # Invert the contrast.
    hi = max (im1)
    im2 = hi - im1
    # Blur the image.
    blurmask = image ((blursize, blursize, 1))
    set (blurmask, 1.0)
    convolve (im2, blurmask, 'mean')
    # Blend the layers, clipping the result to keep it sensible.
    fac = opacity / max (im2)
    im2 =  im1 / (1.0 - im2 * fac)
    clip (im2, 0.0, max_image_value)
    return im2

#-------------------------------------------------------------------------------
def effect_sepia (im):
    """
    Make an image have a sepia appearance.

    Arguments:
    im  image to be made sepia (modified)
    """
    r = get_channel (im, 0)
    g = get_channel (im, 1)
    b = get_channel (im, 2)
    rr = 0.393 * r + 0.769 * g + 0.189 * b
    gg = 0.349 * r + 0.686 * g + 0.168 * b
    bb = 0.272 * r + 0.534 * g + 0.131 * b
    set_channel (im, 0, rr)
    set_channel (im, 1, gg)
    set_channel (im, 2, bb)
    clip (im, 0.0, max_image_value)

#-------------------------------------------------------------------------------
def effect_streaks (im, width=1, height=6, direction='h', occ=0.9, fg=0.0,
                    bg=max_image_value):
    """
    Return a representation of an image as horizontal or vertical streaks.

    Convert an image into a series of two-level streaks, the width of the
    streak indicating the darkness of that part of the original image.  The
    inspiration for the routine is the illustrations in the book 'The
    Cloudspotter's Guide' by Gavin Pretor-Pinny.

    Arguments:
           im  image to be processed
        width  width of each region to be processed (default: 1)
       height  width of each region to be processed (default: 6)
    direction  direction in which the streaks will go (default: 'h')
               'h': horizontal
               'v': vertical
          occ  maximum occupancy of each region (default: 0.9)
           fg  foreground value (0.0)
           bg  background value (255.0)
    """
    # Produce single-channel output, even from a multi-channel input image.
    ny, nx, nc = sizes (im)
    to = image ((ny, nx, 1))

    # Get the image range and handle the case when the image is blank.
    lo, hi = extrema (im)
    if lo >= hi:
        set (to, 0.0)
        return to

    # Process the image in regions of width x height pixels.  For each region,
    # we calculate its mean, then determine what proportion of that region
    # should be filled with the foreground value.  We then set the entire
    # region to the background value, and finally fill the relevant proportion
    # of the region with the foreground value.

    # There are two aesthetic refinements to this process.  The first is that
    # we incorporate a contrast reversal if the foreground value is darker
    # than the background one.  The second is that we reduce the proportion of
    # the region that is filled by an occupancy factor as this makes the end
    # result look better.
    if direction == 'v' or direction == 'V':
        horiz = False
        fac = width * occ / (hi - lo)
    else:
        horiz = True
        fac = height * occ / (hi - lo)
    for y in xrange (0, ny, height):
        yto = y + height
        if yto > ny:  yto = ny
        for x in xrange (0, nx, width):
            xto = x + width
            if xto > nx:  xto = nx
            reg = region (im, y, yto, x, xto)
            lo, hi = extrema (reg)
            if bg > fg:
                val = int((hi - mean(reg)) * fac + 0.5)
            else:
                val = int((mean (reg) - lo) * fac + 0.5)
            if val < 0: val = 0
            set_region (to, y, x, yto, xto, bg)
            if horiz:
                set_region (to, y, x, y + val, xto, fg)
            else:
                set_region (to, y, x, yto, x + val, fg)
    return to

#-------------------------------------------------------------------------------
def effect_solarize (im, threshold=None):
    """
    Solarize an image. (UNTESTED)

    Arguments:
           im  image to be solarized
    threshold  threshold above which the effect is applied
    """
    value = max (im)
    if threshold is None: threshold = value / 2.0
    ny, nx, nc = sizes (im)
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            for c in xrange (0, nc):
                if im[y,x,c] < threshold: im[y,x,c] = value - im[y,x,c]

#-------------------------------------------------------------------------------
def extract (im, ry, rx, yc, xc, step=1.0, angle=0.0, wrap=False, val=0,
             interpolator='gradient'):
    """
    Return a ry x rx-pixel region of im centred at (yc, xc).

    Arguments:
               im  image from which the region is to be extracted
               ry  number of pixels in the y-direction of the extracted region
               rx  number of pixels in the x-direction of the extracted region
               yc  y-position of the centre of the region to be extracted
               xc  x-position of the centre of the region to be extracted
             step  step size on im (default: 1.0)
                   or a list [ystep, xstep]
            angle  angle of sampling grid relative to im, measured anticlockwise
                   in radian (default: 0.0)
             wrap  if True, 'falling off' one size of the image will wrap
                   around to the opposite side (otherwise, such pixels will be
                   zero) (default: False)
              val  value to which pixels outside the image are set if not
                   wrapping (default: 0)
    interpolation  interpolation scheme, one of 'gradient', 'bilinear' or
                   'nearest' (default: 'gradient')

    The region extracted from im can be centred around a non-integer
    position in im, and extracted with an arbitrary step size at an
    arbitrary angle.  The interpolation schemes supported are either
    conventional bilinear or a gradient-based one described in
    P.R. Smith (Ultramicroscopy vol 6, pp 201--204, 1981), as well as
    simple nearest neighbour.
    """
    if isinstance (step, list):
        ystep, xstep = step
    else:
        ystep = xstep = step
    ny, nx, nc = sizes (im)
    region = image ((ry, rx, nc))
    y0 = yc - ry / 2
    x0 = xc - rx / 2
    if abs (angle) < tiny and \
            abs (ystep - 1.0) < tiny and abs (xstep - 1.0) < tiny and \
            ry < ny and rx < nx and y0 >= 0 and x0 >= 0 and \
            abs (y0 - int(y0)) < tiny and abs (x0 - int(x0)) < tiny:
        region = im[y0:y0+ry,x0:x0+rx]
    else:
        mim = im
        if interpolator == 'gradient':
            interp = 2
            mim = mono (im)
        elif interpolator == 'bilinear':  interp = 3
        elif interpolator == 'nearest':   interp = 1
        else:
            print >>sys.stderr, ('extract: invalid interpolator "%s"; ' + \
                  'using "nearest"') % interpolator
            interp = 1
        cosfac = math.cos (-angle)
        sinfac = math.sin (-angle)
        disty = ry / 2
        distx = rx / 2
        yst = yc + distx * xstep * sinfac - disty * ystep * cosfac
        xst = xc - distx * xstep * cosfac - disty * ystep * sinfac
        for y in xrange (0, ry):
            ypos = yst
            xpos = xst
            for x in xrange (0, rx):
                if (ypos < 0 or ypos >= ny or xpos < 0 or xpos >= nx) \
                        and not wrap:
                    region[y,x] = val
                else:
                    ylo = int (ypos)
                    dy =  ypos - ylo
                    dy1 = 1 - dy
                    ylo = (ylo + ny) % ny
                    yhi = ylo + 1
                    if yhi >= ny and not wrap:
                        region[y,x] = val
                    else:
                        yhi = (yhi + ny) % ny
                        xlo = int (xpos)
                        dx = xpos - xlo
                        dx1 = 1 - dx
                        xlo = (xlo + nx) % nx
                        xhi = xlo + 1
                        if xhi >= nx  and not wrap:
                            region[y,x] = val
                        else:
                            xhi = (xhi + nx) % nx
                            if interp == 3:
                                region[y,x] = \
                                    dy *dx*im[yhi,xhi] + dy *dx1*im[yhi,xlo] + \
                                    dy1*dx*im[ylo,xhi] + dy1*dx1*im[ylo,xlo]
                            elif interp == 2:
                                if abs (mim[ylo,xlo] - mim[yhi,xhi]) > \
                                        abs (mim[yhi,xlo] - mim[ylo,xhi]):
                                    region[y,x] = (dx-dy) * im[ylo,xhi] + \
                                        dx1*im[ylo,xlo] + dy*im[yhi,xhi]
                                else:
                                    region[y,x] = (dx1-dy) * im[ylo,xlo] + \
                                        dx*im[ylo,xhi] + dy*im[yhi,xlo]
                            else:
                                region[y,x] = im[int(ylo+0.5),int(xlo+0.5)]
                ypos -= ystep * sinfac
                xpos += xstep * cosfac
            yst += ystep * cosfac
            xst += xstep * sinfac
    return region

#-------------------------------------------------------------------------------
def extrema (im):
    """
    Return the minimum and maximum of an image.

    Arguments:
    im  image whose extrema are to be found
    """
    return [im.min(), im.max()]

#-------------------------------------------------------------------------------
def fill_outline (im, y, x, v=max_image_value, threshold=100):
    """
    Flood fill the region lying within a border.

    Arguments:
           im  image containing region to be filled (modified)
           yc  y-coordinate of point at which filling is to start
           xc  x-coordinate of point at which filling is to start
            v  value to which the filled region is to be set (default: 255)
    threshold  minimum difference in value from centre pixel at boundary
               (default: 100)

    This code is based on that written by Eric S. Raymond at
    http://mail.python.org/pipermail/image-sig/2005-September/003559.html
    This code is an elegant Python implementation of Paul Heckbert's
    classic flood-fill algorithm, presented in"Graphics Gems".
    """
    ny, nx, nc = sizes (im)
    if x < 0 or x >= nx or y < 0 or y >= ny: return
    vc = im[y,x].sum()
    if abs(vc - v) < threshold: return
    im[y,x] = v
    # At each step there is a list of edge pixels for the flood-filled 
    # region.  Check every pixel adjacent to the edge; for each, if it is
    # eligible to be coloured, colour it and add it to a new edge list.
    # Then you replace the old edge list with the new one.  Stop when the
    # list is empty.
    edge = [(y, x)]
    while edge:
        newedge = []
        for (y, x) in edge:
            for (t, s) in ((y, x+1), (y, x-1), (y+1, x), (y-1, x)):
                if s >= 0 and s < nx and t >= 0 and t < ny and \
                   abs(im[t,s].sum() - vc) < threshold:                  
                    im[t,s] = v
                    newedge.append ((t, s))
        edge = newedge

#-------------------------------------------------------------------------------
def find_in_path (prog):
    """
    Return the absolute pathname of a program which is in the search path.

    Arguments:
    prog  program whose absolute filename is to be found
    """
    # First, split the PATH variable into a list of directories, then find
    # the first program from our list that is in the path.
    path = string.split(os.environ['PATH'], os.pathsep)
    for p in path:
        fp = os.path.join(p, prog)
        if os.path.exists(fp): return os.path.abspath(fp)
    return None

#-------------------------------------------------------------------------------
def find_peaks (im, threshold):
    """
    Return a list of the peaks in an image in descending order of height.

    A peak is defined as a pixel whose value is larger than those of all
    surrounding pixels and has a value greater than threshold.  Each
    peak is described by a three-element list containing its pixel value
    and the y- and x-values at which the peak was found.

    Arguments:
           im  image whose peaks are to be found
    threshold  value used for determining which peaks are significant
    """
    ny, nx, nc = sizes (im)
    peaks = list ()
    for y in xrange (1, ny-1):
        for x in xrange (1, nx-1):
            if      im[y,x,0] > im[y-1,x-1,0] \
                and im[y,x,0] > im[y-1,x  ,0] \
                and im[y,x,0] > im[y-1,x+1,0] \
                and im[y,x,0] > im[y  ,x-1,0] \
                and im[y,x,0] > im[y  ,x+1,0] \
                and im[y,x,0] > im[y+1,x-1,0] \
                and im[y,x,0] > im[y+1,x  ,0] \
                and im[y,x,0] > im[y+1,x+1,0] \
                and im[y,x,0] > threshold:
                peaks.append ([im[y,x,0], y, x])
    # Return the peaks sorted into descending order.
    peaks.sort (reverse=True)
    return peaks

#-------------------------------------------------------------------------------
def find_skin (im, hlo=300, hhi=30, slo=10, shi=70, vlo=10, vhi=80, ishsv=False):
    """
    Return a binary mask identifying pixels that are likely to be skin.

    This routine identifies potential skin pixels in the image im.  The
    image is converted to HSV format unless ishsv is True, and then
    pixels in the HSV region bounded by [hlo:hhi], [slo:shi] and
    [vlo:vhi] are identified as being skin.  As skin is vaguely red, and
    red in HSV space is 0, hlo will normally be about 330 (degrees) and
    hhi about 30 degrees.  The returned image has non-zero pixels where
    skin has been identified.

    Note that this is not a reliable skin detector  it can be confused
    by incandescent lighting or by similarly-coloured materials such as
    wood.

    Arguments:
       im  image in which skin regions are to be found
      hlo  lowest skin hue (default: 300)
      hhi  highest skin hue (default: 30)
      slo  lowest skin saturation (default: 10)
      shi  highest skin saturation (default: 70)
      vlo  lowest skin value (default: 10)
      vhi  highest skin value (default: 80)
    ishsv  if True, the input image contains pixels in HSV format
            rather than RGB (default: False)
    """
    return segment_hsv (im, hlo, hhi, slo, shi, vlo, vhi, ishsv)

#-------------------------------------------------------------------------------
def find_threshold_otsu (im):
    """
    Return the optimal image threshold, found using Otsu's method.

    This routine is minimally adapted from the code in 'ImageP.py' by
    Tamas Haraszti, which he says is ultimately derived from Octave code
    written by Barre-Piquot.  I note, in passing, that this facility is
    also available as part of Matlab, though I've never seen the code.
    The algorithm itself is N. Otsu: 'A Threshold Selection Method from
    Gray-Level Histograms', IEEE Transactions on Systems, Man and
    Cybernetics vol 9 no 1 pp 62-66 (1979).

    Arguments:
    im  image for which the threshold is to be found
    """
    vals = im.copy ()
    mn = vals.min ()
    vals = vals - mn
    N = int (vals.max ())
    h, x = numpy.histogram (vals, bins=N)
    h = h / (h.sum() + 1.0)
    w = h.cumsum ()
    i = numpy.arange (N, dtype=float) + 1.0
    mu = numpy.zeros (N, float)
    mu = (h*i).cumsum()
    w1 = 1.0 - w
    mu0 = mu / w
    mu1 = (mu[-1] - mu) / w1
    s = w * w1 * (mu1 - mu0)**2
    return float ((s == s.max()).nonzero()[0][0]) + mn

#-------------------------------------------------------------------------------
def fourier (im, forward=True):
    """
    Perform a Fourier transform of an image.

    Note that this routine leaves the zero frequency in the centre of
    the image.

    Arguments:
         im  image to be transformed (modified)
    forward  if True, preform a forward transform (default: True)
    """
    if forward:
        # Ensure we have a complex image for the result.
        dims = sizes (im)
        res = image (dims, type=numpy.complex64)
        res = im
        # Transform, then move the origin to the centre of the image.
        temp = numpy.fft.fft2 (im, axes=(-3,-2))
        res = numpy.fft.fftshift (temp, axes=(-3,-2))
    else:
        # Move the origin from the centre to the corner, then transform.
        temp = numpy.fft.ifftshift (im, axes=(-3,-2))
        res = numpy.fft.ifft2 (temp, axes=(-3,-2))
    return res

#-------------------------------------------------------------------------------
def get_channel (im, c):
    """
    Return a channel of an image.

    Arguments:
    im  the image from which the channel is to be extracted
     c  the index of the channel that is to be extracted
    """
    ny, nx, nc = sizes (im)
    ch = image ((ny, nx, 1))
    ch[:,:,0] = im[:,:,c]
    return ch

#-------------------------------------------------------------------------------
def grab (prog=None, suffix=''):
    """
    Return an image grabed using the computer's camera.

    Arguments:
       prog  name of the program with which to capture the image; by default
             this is one of:
             isightcapture (MacOS X)
                  streamer (Linux)
    suffix  the suffix of the temporary file in whch the image is captured
            (by default, this depends on the capture program used)
    """
    if prog is None:
        if find_in_path ('isightcapture'):
            fn = tempfile.mkstemp (suffix='.png')[1]
            os.system ('isightcapture -t png ' + fn)
        elif find_in_path ('streamer'):
            fn = tempfile.mkstemp (suffix='.ppm')[1]
            os.system ('streamer -q -f ppm -o ' + fn)
    else:
        fn = tempfile.mkstemp (suffix=suffix)[1]
        os.system (prog + ' ' + fn)
    pic = image (fn)
    os.remove (fn)
    return pic

#-------------------------------------------------------------------------------
def graph_gnuplot (x, y, title=' ', xlabel='x', ylabel='y',
                   logx=False, logy=False,
                   style='lines', key=None, pause=True):
    """
    Graph data using Gnuplot.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, a valid Gnuplot line-type
            or 'histogram' (default: 'lines')
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
     pause  if True, allow the user to view the plot (and optionally save
            the data to file) before continuing (default: True)
    """
    # Work out if we're producing one or several plots.
    if isinstance (y, list): nydims = 1
    else:                    nydims = len (y.shape)
    if nydims == 1:
        ny = len (y)
    else:
        nydims = y.shape[0]
        ny = len (y[0])
    if x is None:
        x = numpy.ndarray ((ny))
        for ix in xrange (0, ny):
            x[ix] = ix
    p = os.popen ('gnuplot', 'w')
    if key is None: print >>p, 'set nokey'
    print >>p, 'set grid'
    print >>p, 'set title "' + title + '"'
    print >>p, 'set xlabel "' + xlabel + '"'
    print >>p, 'set ylabel "' + ylabel + '"'
    if logx: print >>p, 'set log x'
    if logy: print >>p, 'set log y'
    if style == 'histogram':
        print >>p, 'set style fill solid'
        extra = 'with boxes'
    else:
        print >>p, 'set style data ' + style
        extra = ''
    print >>p, 'plot',
    for dataset in xrange (0, nydims-1):
        print >>p, '"-"',
        if not key is None: print >>p, ('title "%s"' % key[dataset]),
        print >>p, ',',
    print >>p, '"-"',
    if not key is None: print >>p, ('title "%s' % key[nydims-1])
    print >>p, extra
    # We access the contents of y differently if we're producing a single plot
    # of a set of plots.
    if nydims == 1:
        for ix, iy in zip (x, y):
            print >>p, ix, iy
        print >>p, 'e'
    else:
        for dataset in xrange (0, nydims):
            for ix, iy in zip (x, y[dataset,:]):
                print >>p, ix, iy
            print >>p, 'e'
    p.flush ()
    # Exit if the user types <EOF>; give (minimal) instructions if they type
    # "?"; simply continue if they type <return>.  Anything else typed in
    # response to the prompt is assumed to be a filename and we save the data
    # that file. (And yes, that can result in silly filenames...)
    if pause: looping = True
    else:     looping = False
    while looping:
        sys.stderr.write ('CR> ')
        fn = sys.stdin.readline()
        if len(fn) < 1:
            print >>sys.stderr, "Exiting..."
            sys.exit (1)
        if len(fn) > 0 and fn == '?\n':
            print >>sys.stderr, 'fn to save data to "fn" else <return>'
            continue
        if len(fn) > 1 and fn != '':
            f = open (fn[:-1],'w')
            for ix, iy in zip(x, y):
                print >>f, ix, iy
            f.close ()
        looping = False
    p.close ()

#-------------------------------------------------------------------------------
def graph_pgfplots (x, y, fn, title=' ', xlabel='x', ylabel='y', logx=False,
                   logy=False, style='lines', key=None, preamble=True):
    """
    Graph data using LaTeX's PGFplot style file.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
        fn  the name of the file to receive the LaTeX commands for the plot
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, a valid Gnuplot line-type
            or 'histogram' (default: 'lines')
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
  preamble  if True, write out the document preamble at the top of the file
    """
    # Preparation for plotting.
    if isinstance (y, list): nydims = 1
    else:                    nydims = len (y.shape)
    if nydims == 1:
        ny = len (y)
    else:
        nydims = y.shape[0]
        ny = len (y[0])
    if x is None:
        x = numpy.ndarray ((ny))
        for ix in xrange (0, ny):
            x[ix] = ix

    # Open the file and write out the preamble.
    f = open (fn, "w")
    if preamble: print >>f, r"""
%\usepackage{pgfplots}  % <-- in the document preamble
\pgfplotsset{compat=newest}
\pgfplotsset{eve/.style={
    y tick label style={
      /pgf/number format/.cd,
      fixed,
      fixed zerofill,
      precision=1,
      /tikz/.cd
    },
    x tick label style={
      /pgf/number format/.cd,
      fixed,
      fixed zerofill,
      precision=1,
      /tikz/.cd
    },
    tick label style = {font=\sffamily\small},
    every axis label = {font=\sffamily\small},
    legend style = {font=\sffamily},
    label style = {font=\sffamily\small}}
}"""

    # Work out the line style.
    if style == "lines": mark = "none"
    else: mark = "*"

    # Work out the axis type and write out the beginning of the plot.
    if logx and logy: axis = "loglogaxis"
    elif logx: axis = "semilogxaxis"
    elif logx: axis = "semilogyaxis"
    else: axis = "axis"
    print >>f, r"\begin{figure}"
    print >>f, r"  \begin{center}"
    print >>f, r"    \begin{tikzpicture}"
    print >>f, r"      \begin{%s}[eve, xlabel=%s, ylabel=%s," % \
        (axis, xlabel, ylabel)
    print >>f, r"         width=0.8\textwidth, height=0.45\textheight]"
    print >>f, r"         \addplot[mark=%s] coordinates {" % mark
    

    # Write out the data.  We access the contents of y differently if
    # we're producing a single plot of a set of plots.
    if nydims == 1:
        for ix, iy in zip (x, y):
            print >>f, '          (%f, %f)' % (ix, iy)
        print >>f, '        };'
    else:
        for dataset in xrange (0, nydims):
            for ix, iy in zip (x, y[dataset,:]):
                print >>f, '          (%f, %f)' % (ix, iy)
            print >>f, '        };'


    # Finish the plot off.
    print >>f, r"""      \end{%s}""" % axis
    print >>f, r"""    \end{tikzpicture}
  \end{center}
  \caption{%s}
  \label{fig:%s}
\end{figure}""" % (title, title)
    f.close()

#-------------------------------------------------------------------------------
def graph (x, y, title=' ', xlabel='x', ylabel='y', logx=False, logy=False,
                   style='lines', key=None, pause=True):
    """
    Graph data using Matplotlib.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, a valid Gnuplot line-type
            or 'histogram' (default: 'lines')
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
     pause  if True, allow the user to view the plot (and optionally save
            the data to file) before continuing (default: True)
    """
    import pylab as p
    # Work out if we're producing one or several plots.
    if isinstance (y, list): nydims = 1
    else:                    nydims = len (y.shape)
    if nydims == 1:
        ny = len (y)
    else:
        nydims = y.shape[0]
        ny = len (y[0])
    # Maks sure we have some x-values to plot.
    if x is None:
        x = numpy.ndarray ((ny))
        for ix in xrange (0, ny):
            x[ix] = ix
    # Set up pylab.
    p.figure ()
    p.grid ()
    p.title (title)
    p.xlabel (xlabel)
    p.ylabel (ylabel)
    # We access the contents of y differently if we're producing a single plot
    # of a set of plots.
    if nydims == 1:
        if style == 'histogram': p.bar  (x, y, align='center')
        else:                    p.plot (x, y)
    else:
        lab = None
        for dataset in xrange (0, nydims):
            if not key is None: lab = key[dataset]
            if style == 'histogram':
                p.bar  (x, y[dataset,:], label=lab, align='center')
            else:
                p.plot (x, y[dataset,:], label=lab)
        if not key is None: p.legend ()
    p.show ()

    # Exit if the user types <EOF>; give (minimal) instructions if they type
    # "?"; simply continue if they type <return>.  Anything else typed in
    # response to the prompt is assumed to be a filename and we save the data
    # that file. (And yes, that can result in silly filenames...)
    if pause: looping = True
    else:     looping = False
    while looping:
        sys.stderr.write ('CR> ')
        fn = sys.stdin.readline()
        if len(fn) < 1:
            print >>sys.stderr, "Exiting..."
            sys.exit (1)
        if len(fn) > 0 and fn == '?\n':
            print >>sys.stderr, 'fn to save data to "fn" else <return>'
            continue
        if len(fn) > 1 and fn != '':
            f = open (fn[:-1],'w')
            for ix, iy in zip(x, y):
                print >>f, ix, iy
            f.close ()
        looping = False

#-------------------------------------------------------------------------------
def harris (im, min_distance=10, threshold=0.1, inc=2, disp=False):
    '''
    Return corners found in an image using the Harris-Stephens detector.
    Note that this routine is currently much too naive to be used in anger!

    Arguments:
              im  image to be processed
    min_distance  minimum number of pixels separating corners and
                  image boundary (default: 10)
       threshold  minimum response for a pixel to be considered a corner
                  (default: 0.1)
             inc  the increment between pixels when sub-sampling (default: 2)
            disp  if set, display the corners on a darkened copy of the image
    '''
    full_corners = harris_corners (im)
    full_corners.sort ()
    im2 = subsample (im, inc)
    half_corners = harris_corners (im2)
    half_corners.sort ()
    corners = []
    for i in xrange (0, len(full_corners)):
        fy, fx = full_corners[i]
        hy, hx = half_corners[i]
        y = fy + (fy - inc * hy)
        x = fx + (fx - inc * hx)
        corners.append ((y,x))

    # Display the corners we've found, if the caller wants to.
    if disp:
        mim = copy (im)
        mim *= 0.4
        mark_positions (mim, corners, disp=True)
    return corners

#-------------------------------------------------------------------------------
def harris_corners (im, min_distance=10, threshold=0.1):
    '''
    Detect corners in an image using the Harris-Stephens detector.
    Note that this routine returns corners with a systematic error
    inherent in the detector; a wrapper routine, harris, processes the
    image at two scales to remove the bias.

    This routine is adapted from code written by Jan Erik Solem; see
    http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html

    Arguments:
              im  image to be processed
    min_distance  minimum number of pixels separating corners and
                  image boundary (default: 10)
       threshold  minimum response for a pixel to be considered a corner
                  (default: 0.1)
    '''
    import scipy
    from scipy import signal

    # Ensure the image is monochrome.
    ny, nx, nc = sizes (im)
    if nc > 1: mim = mono (im)
    else: mim = copy (im)

    # Calculate the Gaussian kernel and its derivatives.  Using them, compute
    # the components of the structure tensor, and from them calculate the
    # determinant and trace; the ratio of these gives the response.  We
    # add a tiny amount in the final expression to avoid problems in
    # homogeneous regions in the subsequent division.  I think Alison Noble
    # was the first person to use this particular work-around  see her DPhil
    # thesis (from the robotics group at Oxford)  but it's a fairly obvoius
    # thing to do.
    size = 3
    y, x = numpy.mgrid[-size:size+1, -size:size+1]
    gauss = numpy.exp(-(x**2/float(size) + y**2/float(size)))
    gauss /= gauss.sum()

    # Calculate the x and y derivatives of a 2D gaussian with standard
    # deviation half of its size.
    gx = - x * numpy.exp(-(2.0*x/size)**2 - (2.0*y/size)**2)
    gy = - y * numpy.exp(-(2.0*x/size)**2 - (2.0*y/size)**2)
    imx = signal.convolve (im[:,:,0], gx, mode='same')
    imy = signal.convolve (im[:,:,0], gy, mode='same')
    Wxx = scipy.signal.convolve (imx*imx, gauss, mode='same')
    Wxy = scipy.signal.convolve (imx*imy, gauss, mode='same')
    Wyy = scipy.signal.convolve (imy*imy, gauss, mode='same')
    harrisim = (Wxx * Wyy - Wxy**2) / (Wxx + Wyy + tiny)

    # Find the top corner candidates above the threshold.
    corner_threshold = max (harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # Get the coordinates of candidate corners and their values, then sort them.
    cands = harrisim_t.nonzero()
    coords = [(cands[0][c], cands[1][c]) for c in xrange(len(cands[0]))]
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]
    index = scipy.argsort(candidate_values)

    # Store allowed point locations in an array then select the best points,
    # taking min_distance into account.
    allowed_locations = numpy.zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1
    corners = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            corners.append(coords[i])
            allowed_locations[(coords[i][0] - min_distance):\
                              (coords[i][0] + min_distance),\
                              (coords[i][1] - min_distance):\
                              (coords[i][1] + min_distance)] = 0

    return corners

#-------------------------------------------------------------------------------
def high_peaks (peaks, factor=0.5):
    """
    Given a sorted list of peaks, return those within factor of the highest.

    Arguments:
    peaks   list of peaks sorted into descending order
    factor  peaks of height within factor of the highest are returned
            (default: 0.5)
    """
    threshold = peaks[0][0] * factor
    n = 0
    for ht, y, x in peaks:
        if ht < threshold: break
        n += 1
    return peaks[:n]
    
#-------------------------------------------------------------------------------
def histogram (im, bins=64, limits=None, disp=False):
    """
    Find the histogram of an image.

    Arguments:
        im  image for which the histogram is to be found
      bins  number of bins in the histogram (default: 64)
    limits  extrema between which the histogram is to be found
            (default: calculated from the image)
      disp  if True, the histogram will be drawn (default: False)
    """
    h, a = numpy.histogram (im, bins, limits)
    if disp: graph_gnuplot (a, h, 'Histogram', 'bin', 'number of pixels',
                    style='histogram')
    return a, h

#-------------------------------------------------------------------------------
def hough_line (im, nr=512, na=512, yc=None, xc=None, threshold=10,\
                disp=False, dispacc=False):
    """
    Perform the Hough transform for lines of the image 'im'.

    This routine performs a straight-line Hough transform of the image 'im',
    which should normally contain output from an edge detector.  It returns a
    list of the significant peaks found (see find_peaks for a description of
    its content) and the image that forms the accumulator.  The accumulator is
    of dimension [na, nr], the distance from the origin (yc, xc) being plotted
    along the x-direction and the corresponding angle along the y-direction.

    Arguments:
           im  image for which the Hough transform is to be performed
           nr  number of radial values (x-direction of the accumulator)
           na  number of angle values (y-direction of the accumulator)
           yc  y-value of the origin on the image array (default: image centre)
           xc  x-value of the origin on the image array (default: image centre)
    threshold  minimum value for a significant peak in the accumulator
               (default: 10)
         disp  if True, draw the lines found over the image (default: false)
       dispcc  if True, display the accumulator array (default: false)
    """
    ny, nx, nc = sizes (im)
    if yc is None:  yc = ny / 2
    if xc is None:  xc = nx / 2
    acc = image ((na, nr, 1))
    ainc = math.pi / na
    rinc = nr / math.sqrt (ny**2 + nx**2)
    # Find edge points and update the Hough array.
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            v = im[y,x,0]
            if v > 0:
                for a in xrange (0, na):
                    ang = a * ainc
                    r = ((x - xc) * math.cos(ang) + (y - yc) * math.sin (ang))
                    r += ny
                    if r >= 0 and r < nr:
                        acc[a,r,0] += 1
    # Now find peaks in the accumulator.
    peaks = find_peaks (acc, threshold=threshold)

    # If the user wants to display what has been found, draw the lines over
    # the image.  (This implmentation is ugly.)
    if dispacc: display (acc)
    if disp:
        d = image ((ny, nx, 3))
        d[:,:,0] = im[:,:,0] * 0.5
        d[:,:,1] = im[:,:,0] * 0.5
        d[:,:,2] = im[:,:,0] * 0.5
        for h, a, r in peaks:
            da = a
            for y in range (0, ny):
                for x in range (0, nx):
                    t = (x - xc) * math.cos (da) + (y - yc) * math.sin (da)
                    if abs(t - r) < 1.0e-3: d[y,x,0] = max_image_value
        display (d)
    return peaks, acc

#-------------------------------------------------------------------------------
def hsv_to_rgb (im):
    """
    Convert an image from HSV space to RGB.

    This routine converts an image in which the hue, saturation and
    value components are in channels 0, 1 and 2 respectively to the RGB
    colour space.  It is assumed that hue lies in the range [0,359],
    while saturation and value are percentages; these are compatible
    with the popular display program 'xv'.  The red, green and blue
    components are returned in channels 0, 1 and 2 respectively, each in
    the range [0,255].

    This routine is adapted from code written by Frank Warmerdam
    <warmerdam@pobox.com> and Trent Hare; see
    http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/hsv_merge.py

    Arguments:
    im  image to be converted (modified)
    """
    h = im[:,:,0] / 360.0
    s = im[:,:,1] / 100.0
    v = im[:,:,2] * max_image_value / 100.0
    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    im[:,:,0] = i.choose (v, q, p, p, t, v)
    im[:,:,1] = i.choose (t, v, v, q, p, p)
    im[:,:,2] = i.choose (p, p, t, v, v, q)

#-------------------------------------------------------------------------------
def image (fromwhat, type=numpy.float32):
    """
    Create an EVE image.

    Arguments:
    fromwhat  the source from which the image is to be created, one of:
                     a string:  the name of a file to be read in
                a numpy array:  the new image is a copy of this image
              a list or tuple:  the dimensions (ny, nx, nc)
        type  the type of the image to be ceated (default: numpy.float32)
    """
    if isinstance (fromwhat, str):
        import Image
        pic = Image.open (fromwhat)
        # Something seems to be broken with at least 16-bit TIFFs...
        if pic.mode == "I;16":
            temp = numpy.fromstring(pic.tostring(), dtype=numpy.uint16)
            im = numpy.asarray (temp, dtype=type)
            nc = 1
        else:
            im = numpy.asarray (pic, dtype=type)
            nc = len (pic.getbands ())
            nx, ny = pic.size
            im.shape = [ny, nx, nc]
    elif isinstance (fromwhat, numpy.ndarray):
        ny, nx, nc = fromwhat.shape
        im = numpy.zeros ((ny, nx, nc))
    elif isinstance (fromwhat, list) or isinstance (fromwhat, tuple):
        im = numpy.zeros (fromwhat, dtype=type)
    else:
        raise ValueError, 'Illegal argument type'
    return im

#-------------------------------------------------------------------------------
def insert (im, reg, yc, xc, operation='='):
    """
    Insert image reg into im, centred at (yc, xc).

    Arguments:
           im  image into which the region is to be inserted (modified)
          reg  image which is to be inserted
           yc  y-value of im at which the centre of reg is to be inserted
           xc  x-value of im at which the centre of reg is to be inserted
    operation  way in which im is inserted into output, one of
               '=' (assign), '+' (add), '-' (subtract), '*' (multiply),
               or '/' (divide)
    """
    ny, nx, nc = sizes (reg)
    ylo = yc - ny // 2
    yhi = ylo + ny
    xlo = xc - nx // 2
    xhi = xlo + nx
    if   operation == '=': im[ylo:yhi,xlo:xhi,:]  = reg
    elif operation == '+': im[ylo:yhi,xlo:xhi,:] += reg
    elif operation == '-': im[ylo:yhi,xlo:xhi,:] -= reg
    elif operation == '*': im[ylo:yhi,xlo:xhi,:] *= reg
    elif operation == '/': im[ylo:yhi,xlo:xhi,:] /= reg
    else: raise ValueError, 'Invalid operation type'

#-------------------------------------------------------------------------------
def label_regions (im, con8=False):
    """
    Given a segmented image, return an image with its regions labelled and
    the number of regions found.

    Arguments:
      im  image to be labelled
    con8  if True, consider all 8 nearest neighbours
          if False, consider only 4 nearest neighbours (default)
    """
    import scipy.ndimage

    if con8: ele = [[[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]],
                    [[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]],
                    [[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]]]
    else:    ele = None
    res, nlabs = scipy.ndimage.measurements.label (im, structure=ele)
    return res, nlabs

#-------------------------------------------------------------------------------
def label_regions_slow (im, con8=True):
    """
    Given a segmented image, return an image with its regions labelled.

    Arguments:
      im  image to be labelled
    con8  if True, consider all 8 nearest neighbours
          if False, consider only 4 nearest neighbours
    """
    ny, nx, nc = sizes (im)
    lab = image ((ny, nx, nc), type=numpy.int32)
    vals = [0, 0, 0, 0]
    labs = [1, 0, 0, 0]

    # The upper left pixel is in region zero.
    lastlabel = 0
    equiv = [lastlabel]
    lab[0,0,0] = lastlabel

    # Process the rest of the first row of the image.
    y = 0
    for x in xrange (1, nx):
        if im[y,x,0] != im[y,x-1,0]:
            lastlabel += 1
            equiv.append (lastlabel)
        lab[y,x,0] = lastlabel

    # Process the first column of the image.
    x = 0
    for y in xrange (1, ny):
        if im[y,x,0] == im[y-1,x,0]:
            lv = lab[y-1,x,0]
        else:
            lastlabel += 1
            equiv.append (lastlabel)
            lv = lastlabel
        lab[y,x,0] = lv

    # Process the remainder of the image.
    for y in xrange (1, ny):
        y1 = y - 1
        for x in xrange (1, nx):
            if con8: nv = 4
            else:    nv = 2
            x1 = x - 1
            x2 = x + 1
            if x2 >= nx - 1 and con8: lastcol = True; nv -= 1
            else: lastcol = False
            val = im[y,x,0]
            # Get the four neighbours' values and labels, taking care not
            # to index off the end of the image.
            vals[0] = im[y, x1,0]; labs[0] = lab[y, x1,0]
            vals[1] = im[y1,x, 0]; labs[1] = lab[y1,x, 0]
            if con8:
                vals[2] = im[y1,x1,0]; labs[2] = lab[y1,x1,0]
                if not lastcol:
                    vals[3] = im[y1,x2,0]; labs[3] = lab[y1,x2,0]
            inreg = False
            for i in xrange (0, nv):
                if val == vals[i]: inreg = True
            if not inreg:
                # We're in a new region.
                lastlabel += 1
                equiv.append (lastlabel)
                lv = lastlabel
            else:
                # We must be in the same region as a neighbour.
                matches = []
                for i in xrange (0, nv):
                    if val == vals[i]: matches.append (labs[i])
                matches.sort ()
                lv = int(matches[0])
                for v in matches[1:]:
                    if equiv[v] > lv:
                        equiv[v] = lv
                    elif lv > equiv[v]:
                        equiv[lv] = equiv[v]
            lab[y,x,0] = lv

    # Tidy up the equivalence table.
    remap = list()
    nc = -1
    for i in xrange (0, len(equiv)):
        if equiv[i] == i:
            nc += 1
            v = nc
        else:
            v = i
            while equiv[v] != v:
                v = equiv[v]
            v = remap[v]
        remap.append (v)

    # Make a second pass through the image, re-labelling the regions, then
    # return the labelled image.
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            lab[y,x,0] = remap[lab[y,x,0]]
    return lab, max(lab)

#-------------------------------------------------------------------------------
def labelled_region (labim, lab, bg=0.0, fg=max_image_value):
    """
    Return a region from a labelled image.

    Arguments:
     im  labelled image
    lab  the label that defines which region of the image to return
     bg  value to which pixels outside the region are to be set (default: 0.0)
     fg  value to which pixels inside the region are to be set (default: 255.0)
    """
    im = image (labim)
    set (im, bg)
    im[numpy.where (labim == lab)] = fg
    return im

#-------------------------------------------------------------------------------
def log1 (im):
    """
    Add unity to an image and convert to logarithmic scale.

    Arguments:
    im  image
"""
    im = numpy.log (im + 1)
    return im

#-------------------------------------------------------------------------------
def lut (im, table, stretch=False, limits=None):
    """
    Use a lookup table to adjust pixel values.

    Arguments:
         im  image to be adjusted (modified)
      table  look-up table used to adjust pixel values
    stretch  if True, the image will first be contrast-stretched
             between limits
     limits  a two-element list containing the minimum and maximum
             values to be used for scaling (default: [0, 255])
    """
    ny, nx, nc = sizes (im)
    ntable = len (table)
    if stretch:
        if limits is None:
            lo = 0.0
            hi = max_image_value
        else:
            lo, hi = extrema (im)
        contrast_stretch (im, low=lo, high=hi)
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            for c in xrange (0, nc):
                v = im[y,x,c]
                if v >= 0 and v < ntable: im[y,x,c] = table[int(v)]

#-------------------------------------------------------------------------------
def mark_at_position (im, y, x, v=max_image_value, symbol='.', size=9):
    """
    Plot a marker in an image.

    Arguments:
        im  image in which the positions are to be marked (modified)
         y  y-position of the centre of the mark
         x  x-position of the centre of the mark
     value  value to which peak locations will be set (default: 255)
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
            'o'  a 3 x 3 circle
      size  size of the plotting symbol (default: 9)
    """
    half = math.ceil (size / 2)
    yy = y - half
    xx = x - half
    if symbol == '.':
        im[y,x] = v
    elif symbol == '+':
        draw_line_fast (im, yy, x, yy+size+1, x, v)
        draw_line_fast (im, y, xx, y, xx+size+1, v)
    elif symbol == 'x':
        draw_line_fast (im, yy, xx, yy+size+1, xx+size+1, v)
        draw_line_fast (im, yy, xx+size+1, yy+size+1, xx, v)
    elif symbol == 'o':
        im[y-1,x-1] = im[y-1,x] = im[y-1,x+1] = v
        im[y,x-1] = im[y,x] = im[y,x+1] = v
        im[y+1,x-1] = im[y+1,x] = im[y+1,x+1] = v
    else:
        raise ValueError, 'Unrecognised plotting point: "' + symbol + '"'

#-------------------------------------------------------------------------------
def mark_features (im, locs, v=255, fac=1.0, fast=False, disp=True, scale=1.0):
    """
    Mark the positions, sizes and orientations of features in an image.

    Given a list of feature locations such as those returned by SIFT, in
    which each element of the list consists of a list of y-position,
    x-position, scale and orientation, this routine marks them on the
    image im.  Each line drawn is drawn with value val and is scaled by
    the factor fac.  If disp is True, the result is displayed.

    Arguments:
       im  image in which the features are to be drawn (modified)
     locs  a list of the features to be drawn
      fac  scale factor for features drawn on im (default: 1.0)
     disp  if True, the resulting image is displayed (default: True)
    scale  factor to multiply the image by before marking points (default: 1.0)
    """
    im *= scale
    fac = 1.0
    ny, nx, nc = sizes (im)
    for y, x, s, o in locs:
        yy = y + fac * s * math.sin (-o)
        xx = x + fac * s * math.cos (-o)
        if yy < 0: yy = 0
        if yy >= ny: yy = ny - 1
        if xx < 0: xx = 0
        if xx >= nx: xx = nx - 1
        if fast: draw_line_fast (im, y, x, yy, xx, v)
        else: draw_line (im, y, x, yy, xx, v)
    if disp: display (im)

#-------------------------------------------------------------------------------
def mark_matches (im1, im2, loc1, loc2, scores, v=max_image_value, fast=False,
                  threshold=0.0, number=False, disp=True, name='Matches'):
    """
    Draw lines between corresponding match points, returning the result.

    Arguments:
          im1  first image for which matches have been found
          im2  second image for which matches have been found
         loc1  feature points found in im1 from SIFT or similar
         loc2  feature points found in im2 from SIFT or similar
       scores  score between features found by match_descriptors
            v  value with which the line will be drawn (default: 255)
         fast  if True, lines are drawn for speed rather than appearance
               (default: False)
    threshold  if a score is above threshold, it will be drawn
               (default: 0.0)
       number  if True, add the list element into scores alongside each line
               (default: False)
         disp  if True, the resulting image will be displayed (default: True)
         name  name passed to eve.display if the image is displayed
               (default: 'Matches')

    Given im1 and im2, a new image is formed whch displays them side by
    side, and then lines are drawn between corresponding points (stored
    in loc1 and loc2) for which the corresponding score is greater than
    threshold.  The routine is intended for displaying the results of
    feature found by SIFT and matched by match_descriptors.
    """
    ny1,nx1,nc1 = sizes (im1)
    ny2,nx2,nc2 = sizes (im2)
    ny = ny1 if ny1 > ny2 else ny2
    nx = nx1 + nx2
    nc = nc1 if nc1 > nc2 else nc2
    dim = image ((ny,nx,nc))
    dim[0:ny1,0:nx1,0:nc1] = im1
    dim[0:ny2,nx1:,0:nc2] = im2
    for i in xrange (0, len(scores)):
        i1 = scores[i][1]
        i2 = scores[i][2]
        y1 = loc1[i1,0]
        x1 = loc1[i1,1]
        y2 = loc2[i2,0]
        x2 = loc2[i2,1] + nx1
        if fast: draw_line_fast (dim, int(y1), int(x1), int(y2), int(x2), v)
        else: draw_line (dim, int(y1), int(x1), int(y2), int(x2), v)
        if number:
            offset = 3
            if x2 + character_width + offset >= nx1 + nx2:
                draw_text (dim, ('%s' % i), y2, x2-offset, v, align='r')
            else:
                draw_text (dim, ('%s' % i), y2, x2+offset, v)
            if x1 - character_width - offset <= 0:
                draw_text (dim, ('%s' % i), y1, x1+offset, v)
            else:
                draw_text (dim, ('%s' % i), y1, x1-offset, v, align='r')
    if disp: display (dim, name=name)
    return dim

#-------------------------------------------------------------------------------
def mark_peaks (im, pos, v=max_image_value, disp=False, scale=1.0,
                symbol='+', size=9, name='Peaks'):
    """
    Mark the positions of peaks in an image, such as those returned by
    find_peaks().

    Arguments:
        im  image in which the peak positions are to be marked (modified)
       pos  list of peaks, each element itself a list of [height, y, x]
         v  value to which peak locations will be set (default: 255)
      disp  if True, display the marked-up image
     scale  multiply the image by the factor before marking points
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
            'o'  a 3 x 3 blob
      size  size of the plotting symbol (default: 9)
      name  name for eve.display if the image is displayed (default: 'Peaks')
    """
    im *= scale
    for ht, y, x in pos:
        mark_at_position (im, y, x, v, symbol, size)
    if disp: display (im, name=name)

#-------------------------------------------------------------------------------
def mark_positions (im, pos, v=max_image_value, disp=False, scale=1.0,
                    symbol='+', size=9, name='Positions'):
    """
    Mark positions in an image.

    Arguments:
        im  image in which the positions are to be marked (modified)
       pos  list of positions, each element itself a list of [y, x]
         v  value to which peak locations will be set (default: 255)
      disp  if True, display the marked-up image (default: False)
     scale  multiply the image by the factor before marking points
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
      size  size of the plotting symbol (default: 9)
      name  name for eve.display if the image is displayed
            (default: 'Positions')
    """
    im *= scale
    for y, x in pos:
        mark_at_position (im, y, x, v, symbol, size)
    if disp: display (im, name=name)

#-------------------------------------------------------------------------------
def max (im):
    """
    Return the maximum of an image.

    Arguments:
    im  image for which the maximum value is to be found
    """
    return im.max()

#-------------------------------------------------------------------------------
def match_descriptors_euclidean (desc1, desc2):
    """
    Given pairs of descriptors from SIFT or similar, return the Euclidean
    distances between all pairs, sorted into ascending order.

    Arguments:
    desc1  first set of descriptors
    desc2  second set of descriptors
    """
    score = []
    for i1 in xrange (0, len(desc1)):
        d1 = desc1[i1]
        for i2 in xrange (0, len(desc2)):
            d2 = desc2[i2]
            s = ((d1 - d2)**2).sum()
            score.append ([s, i1, i2])
    score.sort()
    return score

#-------------------------------------------------------------------------------
def match_descriptors (d1, d2, factor=0.6):
    """
    Given pairs of normalized descriptors from SIFT or similar, return
    their best matches sorted into ascending order of match score.

    The match score is calculated as follows.  For each descriptor in
    d1, the scalar product is calculated with all descriptors in d2
    and the best value (smallest angle between scalar products) taken.
    If that value is greater than factor of the second-best value, the
    match is considered ambiguous and discarded; otherwise, triplet of
    the score and the indices into d1 and d2 are inserted into a list
    of scores.  When all possible combinations of d1 and d2 have been
    considered, that list is sorted into ascending order and returned.

    Arguments:
        d1  first set of descriptors
        d2  second set of descriptors
    factor  largest permissible value for a match (default: 0.6)
    """
    nd = d1.shape[0]
    score = []
    for i in xrange (0,nd):
        inprod = numpy.dot (d1[i], d2.T)
        inprod[numpy.where (inprod >  1.0)] =  1.0
        inprod[numpy.where (inprod < -1.0)] = -1.0
        angles = numpy.arccos (inprod)
        ix = numpy.argsort (angles)
        if angles[ix[0]] < factor * angles[ix[1]]:
            score.append ([angles[ix[0]], i, ix[0]])
    score.sort()
    return score

#-------------------------------------------------------------------------------
def mean (im):
    """
    Return the mean of an image.

    Arguments:
    im  image for which the mean value is to be found
    """
    ny, nx, nc = sizes (im)
    return numpy.sum (im) / (ny * nx * nc)

#-------------------------------------------------------------------------------
def min (im):
    """
    Return the minimum of an image.

    Arguments:
    im  image for which the minimum value is to be found
    """
    return im.min()

#-------------------------------------------------------------------------------
def modulus_squared (im):
    """
    Form the squared modulus of each pixel of an image.

    This routine forms the squared modulus of each pixel of an image,
    usually to form the power spectrum of a Fourier transform.

    Arguments:
    im  image for which the power spectrum is to be formed (modified)
    """
    t = im * numpy.conj(im)
    return t.real

#-------------------------------------------------------------------------------
def mono (im):
    """
    Average the channels of colour image to give a monochrome one, returning
    the result.

    Arguments:
    im  image to be converted to monochrome
    """
    ny, nx, nc = sizes (im)
    monoim = image ([ny, nx, 1])
    for c in xrange (0, nc):
        monoim[:,:,0] += im[:,:,c]
    monoim /= nc
    return monoim

#-------------------------------------------------------------------------------
def mono_to_rgb (im):
    """
    Produce a three-channel image of a monochrome image, returning the result.

    Arguments:
    im  image to be converted to monochrome
    """
    ny, nx, nc = sizes (im)
    cim = image ([ny, nx, 3])
    cim[:,:,0] = im[:,:,0]
    cim[:,:,1] = im[:,:,0]
    cim[:,:,2] = im[:,:,0]
    return cim

#-------------------------------------------------------------------------------
def mse (im1, im2):
    """
    Return the mean-squared error (mean-squared difference) between two
    images.

    Arguments:
    im1  image form which im2 is to be subtracted
    im2  image to be subtracted from im1
    """
    ny, nx, nc = sizes (im1)
    return ssd (im1, im2) / float(ny * nx * nc)

#-------------------------------------------------------------------------------
def output (im, fn):
    """
    Output an image to a file, the format being determined by its
    extension.

    Arguments:
    im  image to be output
    fn  name of the file to be written, ending in:
        ".jpg" for JPEG format
        ".png" for PNG format
        ".pnm" or ".pgm" or ".ppm" for PBMPLUS format
    """
    extn = fn[-3:]
    if   extn == 'jpg': output_pil (im, fn, 'JPEG')
    elif extn == 'png': output_pil (im, fn, 'PNG')
    elif extn == 'bmp': output_pil (im, fn, 'BMP')
    elif extn == 'pnm': output_pnm (im, fn)
    elif extn == 'pgm': output_pnm (im, fn)
    elif extn == 'ppm': output_pnm (im, fn)
    else:
        ValueError, 'Unsupported file extension'

#-------------------------------------------------------------------------------
def output_bmp (im, fn):
    """
    Output an image to a file in BMP format.

    Arguments:
    im  image to be output
    fn  name of the file to be written
    """
    output_pil (im, fn, 'BMP')

#-------------------------------------------------------------------------------
def output_jpeg (im, fn):
    """
    Output an image to a file in JPEG format.

    Arguments:
    im  image to be output
    fn  name of the file to be written
    """
    output_pil (im, fn, 'JPEG')

#-------------------------------------------------------------------------------
def output_jpg (im, fn):
    """
    Output an image to a file in JPEG format.

    Arguments:
    im  image to be output
    fn  name of the file to be written
    """
    output_pil (im, fn, 'JPEG')

#-------------------------------------------------------------------------------
def output_pil (im, fn, format='PNG'):
    """
    Output an image to a file using PIL.

    Arguments:
       im  image to be output
       fn  name of the file to be written
    format  the format of the file to be written (default: 'PNG')
    """
    import Image
    ny, nx, nc = sizes (im)
    bim = im.astype ('B')
    if nc == 3:
        pilImage = Image.fromarray (bim, 'RGB')
    elif nc == 4:
        pilImage = Image.fromarray (bim, 'RGBA')
    else:
        pilImage = Image.fromarray (bim[:,:,0], 'L')
    if format == 'display': pilImage.show ()
    else: pilImage.save (fn, format)

#-------------------------------------------------------------------------------
def output_png (im, fn):
    """
    Output an image to a file in PNG format.

    Arguments:
    im  image to be output
    fn  name of the file to be written
    """
    output_pil (im, fn, 'PNG')

#-------------------------------------------------------------------------------
def output_pnm (im, fn, binary=True, stretch=False, biggreys=False):
    """
    Output an image in PBMPLUS format to a file or stdout.

    Arguments:
          im  image to be output
          fn  name of the file to be written
      binary  if True, output binary, rather than text, data (default: True)
     stretch  if True, contrast-stretch the image during output (default: False)
    biggreys  if True, output 16-bit pixels (default: False)
    """

    # First, make sure we know the range of the data we are to output and
    # work out the necessary scaling factor.
    if biggreys: opmax = 62235; fmt = "%6d"
    else:        opmax = max_image_value; fmt = "%4d"
    if stretch:
        lo, hi = extrema (im)
        opmin = 0
        fac = opmax / (hi - lo)
    # Open the file and write out the header.
    ny, nx, nc = sizes (im)
    if binary:
        mode = "b"
        if nc == 1:    pbmtype = "P5"
        elif nc == 3:  pbmtype = "P6"
        else:          pbmtype = "P? (%d channels as binary)" % nc
    else:
        mode = ""
        if nc == 1:    pbmtype = "P2"
        elif nc == 3:  pbmtype = "P3"
        else:          pbmtype = "P? (%d channels as ASCII)" % nc

    if fn == "-":
        f = sys.stdout
    else:
        f = open (fn, "w" + mode)
    f.write (pbmtype + "\n#CREATOR: eve.output_pnm\n%d %d\n%d\n" \
                 % (nx, ny, opmax))

    if binary:
        temp = im
        if stretch: temp = (temp - lo) * fac
        # Clip values into range opmin to opmax
        f.write (temp.astype("B"))
    else:
        for y in xrange (0, ny):
            for x in xrange (0, nx):
                for c in xrange (0, nc):
                    if stretch:
                        v = (im[y,x,c] - lo) * fac
                        if v > opmax: v = opmax
                        if v < opmin: v = opmin
                    else:
                        v = im[y,x,c]
                    if binary:
                        byte = struct.pack ("B", v)
                        f.write (byte)
                    else:
                        f.write (fmt % v)
            if not binary: f.write ("\n")
    if fn != "-": f.close ()

#-------------------------------------------------------------------------------
def pca (im):
    """
    Perform a principal component analysis of the channels of im,
    returning the eigenvalues, kernel and mean values.

    Beware: I am not yet happy that this routine is correct; in particular,
    the sum of the eigenvalues does not match the sum of the variances of
    the input images, which it should!

    Arguments:
    im  image for which the PCA is to be calculated

    This code is based on that written by Jan Erik Solem at
    http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html
    """
    # Re-arrange the data and calculate the mean of all channels of each pixel.
    ny, nx, nc = sizes (im)
    linpix = numpy.ndarray ((nc, ny*nx))
    for c in range (0, nc):
        linpix[c,:] = im[:,:,c].copy().flatten()
    aves = linpix.mean(axis=0)
    
    # Mean-zero the data, form the covariance matrix, then calculate its
    # eigen decomposition.
    for c in range (0, nc):
        linpix[c] -= aves
    covmat = numpy.dot (linpix, linpix.T) / (ny * nx)
    vals, vecs = numpy.linalg.eigh (covmat)

    # Perform Turk's & Pentland's 'compact trick' and reverse the order as we
    # want the eigenvectors and values in descending order.
    temp = numpy.dot (linpix.T, vecs).T
    vecs = temp[::-1]
    vals = numpy.sqrt (vals)[::-1]
    return vals, vecs, aves

#-------------------------------------------------------------------------------
def pca_channels (im):
    """
    Perform a principal component analysis of the channels of im,
    returning the eigenvalues, kernel and means.

    Arguments:
    im  image for which the PCA is to be calculated (modified)
    """
    ny, nx, nc = sizes (im)
    covmat, ave = covariance (im)
    vals, vecs = numpy.linalg.eigh (covmat)
    perm = numpy.argsort(-vals)  # sort in descending order of eigenvalue
    vecs = vecs[:,perm].T        # transpose gives transform kernel
    for y in xrange (0, ny):
        for x in xrange (0, nx):
            v = im[y,x,:] - ave
            im[y,x,:] = numpy.dot (vecs, v)
    # The eigenvalues need to be normalized in order to keep the total variance
    # of the transform equal to that of the input; this is not mentioned in the
    # documentation of the eigen decomposition routine.  The scale factor was
    # found by experiment.
    vals = vals[perm] / nc
    return vals, vecs, ave

#-------------------------------------------------------------------------------
def print_peaks (pos, format="%4d %4d: %.2f", intro=None, fd=sys.stdout):
    """
    Print a series of peaks out, one per line.

    Arguments:
       pos  list containing the peaks to be printed out
    format  format for (y,x) and height to be printed out
            (default: "%4d %4d: %.2f")
     intro  if supplied, this string is printed out before the peaks
        fd  file on which the output is to be written (default: sys.stdout)
    """
    if not intro is None: print intro
    for ht, y, x in pos:
        print >>fd, format % (y, x, ht)

#-------------------------------------------------------------------------------
def print_positions (pos, format="%4d %4d", intro=None, fd=sys.stdout):
    """
    Print a series of positions out, one per line.

    Arguments:
       pos  list containing the positions to be printed out
    format  format for (y,x) to be printed out (default: "%4d %4d")
     intro  if supplied, this string is printed out before the positions
        fd  file on which the output is to be written (default: sys.stdout)
    """
    if not intro is None: print intro
    for y, x in pos:
        print >>fd, format % (y, x)

#-------------------------------------------------------------------------------
def radial_profile (im, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                     ahi=math.pi):
    """
    Return an array of the rotational means at one-pixel radial spacings in an
    annular region of an image.

    Arguments:
     im  the image to be examined
     y0  the y-value of the centre of the rotation (default: centre pixel)
     x0  the x-value of the centre of the rotation (default: centre pixel)
    rlo  the inner radius of the annular region
    rhi  the outer radius of the annular region
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)
    n = int (rhi + 1.0)
    ave = numpy.zeros ((n))
    num = numpy.zeros ((n))
    # Cycle through the image.
    for y in xrange (0, ny):
        yy = (y - y0)**2
        for x in xrange (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            for c in xrange (0, nc):
                if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                    i = (r - rlo)
                    ave[i] += im[y,x,c]
                    num[i] += 1
    # Convert the sums into means.
    for i in xrange (0, n):
        if num[i] > 0: ave[i] /= num[i]
    return ave

#-------------------------------------------------------------------------------
def ramp (im):
    """
    Fill an image with a grey-scale ramp.

    Arguments:
    im  image into which the pattern is written (modified)
    """
    ny, nx, nc = sizes (im)
    im[:,:,:] = numpy.fromfunction (lambda i, j, k: i + j + k, ((ny, nx, nc)))

#-------------------------------------------------------------------------------
def reduce (im, blocksize):
    """
    Reduce the size of an image by averaging each region of
    blocksize x blocksize pixels to a single pixel, returning the result.

    Arguments:
           im  image to be reduced in size
    blocksize  factor by which the size of the image is to be reduced
    """
    ny, nx, nc = sizes (im)
    nny = ny // blocksize
    nnx = nx // blocksize
    nim = image ((nny, nnx, nc))
    for y in range (0, nny):
        ylo = y * blocksize
        yhi = ylo + blocksize
        for x in range (0, nnx):
            xlo = x * blocksize
            xhi = xlo + blocksize
            for c in range (0, nc):
                nim[y,x,c] = im[ylo:yhi,xlo:xhi,c].mean()
    return nim

#-------------------------------------------------------------------------------
def reflect_horizontally (im):
    """
    Reflect an image horizontally.

    Arguments:
    im  image to be reflected (modified)
    """
    ny, nx, nc = sizes (im)
    nx2 = nx // 2
    for y in xrange (0, ny):
        for x in xrange (0, nx2):
            t = im[y,x,:].copy()
            im[y,x,:] = im[y,nx-x-1,:].copy()
            im[y,nx-x-1,:] = t

#-------------------------------------------------------------------------------
def reflect_vertically (im):
    """
    Reflect an image vertically.

    Arguments:
    im  image to be reflected (modified)
    """
    ny, nx, nc = sizes (im)
    ny2 = ny // 2
    for y in xrange (0, ny2):
        for x in xrange (0, nx):
            t = im[y,x,:].copy()
            im[y,x,:] = im[ny-y-1,x,:].copy()
            im[ny-y-1,x,:] = t

#-------------------------------------------------------------------------------
def region (im, ylo, yhi, xlo, xhi):
    """
    Return a rectangular region of an image.

    Arguments:
     im  image from which the region is to be taken
    ylo  lower y-value (row) of the region
    yhi  higher y-value (row) of the region
    xlo  lower x-value (column) of the region
    xhi  higher x-value (column) of the region
    """
    return im[ylo:yhi,xlo:xhi,:]

#-------------------------------------------------------------------------------
def resize (im, nny, nnx, order=1):
    """
    Return im, re-sized to be of size (nny, nnx) by interpolation.

    Arguments:
       im  image to be re-sized
      nny  number of rows in the re-sized image
      nnx  number of columns in the re-sized image
    order  order of interpolating function (defult: 1)
    """
    # The following is adapted from an example in the scipy cookbook.
    import scipy.ndimage
    ny, nx, nc = sizes (im)
    yl, xl = numpy.mgrid[0:ny-1:nny*1j,0:nx-1:nnx*1j]
    coords = scipy.array ([yl, xl])
    result = image ((nny, nnx, nc))
    for c in range (0, nc):
        result[:,:,c] = scipy.ndimage.map_coordinates (im[:,:,c], coords,
                                                       order=order)
    return result

#-------------------------------------------------------------------------------
def rgb_to_hsv (im):
    """
    Convert an image from RGB space to HSV.

    This routine converts an image in which the red, green and blue
    components are in channels 0, 1 and 2 respectively to the HSV colour
    space.  The hue, saturation and value components are returned in
    channels 0, 1 and 2 respectively.  Hue lies in the range [0,359]
    while saturation and value are percentages; these are compatible
    with the popular display program 'xv'.

    Arguments:
    im  image to be converted (modified)

    This routine is adapted from code written by Frank Warmerdam
    <warmerdam@pobox.com> and Trent Hare; see
    http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/hsv_merge.py
    """
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    maxc = numpy.maximum (r, numpy.maximum(g, b))
    minc = numpy.minimum (r, numpy.minimum(g, b))
    v = maxc
    minc_eq_maxc = numpy.equal(minc,maxc)

    # Compute the difference, but reset zeros to ones to avoid divide
    # by zeros later.
    ones = numpy.ones ((r.shape[0], r.shape[1]))
    maxc_minus_minc = numpy.choose (minc_eq_maxc, (maxc-minc,ones))
    s = (maxc - minc) / numpy.maximum (ones, maxc)
    rc = (maxc - r) / maxc_minus_minc
    gc = (maxc - g) / maxc_minus_minc
    bc = (maxc - b) / maxc_minus_minc
    maxc_is_r = numpy.equal (maxc,r)
    maxc_is_g = numpy.equal (maxc,g)
    maxc_is_b = numpy.equal (maxc,b)
    h = numpy.zeros ((r.shape[0], r.shape[1]))
    h = numpy.choose (maxc_is_b, (h, 4.0 + gc - rc))
    h = numpy.choose (maxc_is_g, (h, 2.0 + rc - bc))
    h = numpy.choose (maxc_is_r, (h, bc - gc))
    im[:,:,0] = numpy.mod (h/6.0, 1.0) * 360.0
    im[:,:,1] = s * 100.0   # to be a percentage
    im[:,:,2] = v * 100.0 / max_image_value # to be a percentage

#-------------------------------------------------------------------------------
def rgb_to_mono (im):
    """
    Convert an image from RGB space to luminence (the Y of YIQ).

    This routine converts an image in which the red, green and blue
    components are in channels 0, 1 and 2 respectively to luminance,
    assuming the standard NTSC phosphor.  The result is returned in a
    new image.

    Arguments:
    im  image to be converted
    """
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    ny, nx, nc = sizes (im)
    lum = image ((ny, nx, 1))
    lum[:,:,0] = 0.299*r + 0.587*g + 0.114*b
    return lum

#-------------------------------------------------------------------------------
def rgb_to_yiq (im):
    """
    Convert an image from RGB space to YIQ.

    This routine converts an image in which the red, green and blue components
    are in channels 0, 1 and 2 respectively to the YIQ colour space, assuming
    the standard NTSC phosphor.  The Y, I and Q components are returned in
    channels 0, 1 and 2 respectively.

    Arguments:
    im  image to be converted (modified)
    """
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    im[:,:,0] = 0.299*r + 0.587*g + 0.114*b
    im[:,:,1] = 0.596*r - 0.275*g - 0.321*b
    im[:,:,2] = 0.212*r - 0.523*g + 0.311*b

#-------------------------------------------------------------------------------
def set_mean_sd (im, newmean, newsd):
    """
    Rescale the image to a given mean and sd.

    Arguments:
         im  image to be rescaled (modified)
    newmean  mean the image is to have after rescaling
      newsd  standard deviation that the image is to have after rescaling
    """
    oldmean = mean (im)
    oldsd = sd (im)
    im -= oldmean
    im /= oldsd
    im *= newsd
    im += newmean

#-------------------------------------------------------------------------------
def sd (im):
    """
    Return the standard deviation of an image.

    Arguments:
    im  image for which the standard deviation is to be found
    """
    return im.std (ddof=1)

#-------------------------------------------------------------------------------
def segment_hsv (im, hlo, hhi, slo, shi, vlo, vhi, ishsv=False):
    """
    Return a binary mask identifying pixels that fall within a region
    in HSV space.

    Arguments:
       im  image in which regions are to be found
      hlo  lowest HSV hue
      hhi  highest HSV hue
      slo  lowest HSV saturation
      shi  highest HSV saturation
      vlo  lowest HSV value
      vhi  highest HSV value
    ishsv  if True, the input image contains pixels in HSV format rather
           than RGB (default: False)
    """
    hsvim = copy (im)
    if not ishsv: rgb_to_hsv (hsvim)
    ny, nx, nc = sizes (hsvim)
    mask = image ((ny, nx, 1))
    h = hsvim[:,:,0]
    s = hsvim[:,:,1]
    v = hsvim[:,:,2]
    if hlo > hhi:
        m = ((h < hlo) | (hhi < h)) & (slo < s) & (s < shi) & \
            (vlo < v) & (v < vhi)          # we span 360 degrees
    else:
        m = ((hlo < h) & (h < hhi)) & (slo < s) & (s < shi) & \
            (vlo < v) & (v < vhi)
    mask[numpy.where (m)] = max_image_value
    return mask

#-------------------------------------------------------------------------------
def select_matches (scores, locs1, locs2, max_score_factor=5, max_matches=50):
    """Choose the matches with the best scores.

    Arguments:
              scores  list of scores from match_descriptors
               locs1  locations of features found on the first image
               locs2  locations of features found on the second image
    max_score_factor  ratio of the worst match to the best (default: 5)
         max_matches  maximum number of matches to return (default: 50)
    """
    n = len(scores)
    matches = []
    thresh  = scores[0][0] * max_score_factor
    for i in range (0, n):
        if scores[i][0] <= thresh:
            i1 = scores[i][1] # first image
            i2 = scores[i][2] # second image
            y1 = locs1[i1,0]
            x1 = locs1[i1,1]
            y2 = locs2[i2,0]
            x2 = locs2[i2,1]
            matches.append ([y1, x1, y2, x2])
            if len (matches) >= max_matches: break
        else:
            break
    return matches

#-------------------------------------------------------------------------------
def set (im, v):
    """
    Set all the pixels of an image to a value.

    Arguments:
    im  image to be set (modified)
     v  value to which the pixels are to be set
    """
    im[:,:,:] = v

#-------------------------------------------------------------------------------
def set_channel (im, c, ch):
    """
    Set a channel of an image.

    Arguments:
    im  image in which the channel is to be inserted (modified)
     c  number of the channel which is to be set
    ch  single-channel image which is to be inserted
    """
    im[:,:,c] = ch[:,:,0]

#-------------------------------------------------------------------------------
def set_region (im, yfrom, xfrom, yto, xto, v):
    """
    Set a region of an image to a constant value.

    Arguments:
     im  image in which the region is to be set (modified)
    ylo  lower y-value (row) of the region
    yhi  higher y-value (row) of the region
    xlo  lower x-value (column) of the region
    xhi  higher x-value (column) of the region
      v  value to which the region is to be set
    """
    im[yfrom:yto,xfrom:xto,:] = v

#-------------------------------------------------------------------------------
def sift (im, program='sift %i -o %o 1> /dev/null'):
    """
    Run the SIFT program on an image and return the corresponding keypoints.

    Two arrays are returned, the first giving the locations (and
    corresponding scales and orientations) of the feature points found,
    while the second contains the associated descriptors.

    The particular implementation of SIFT used is from
    http://www.vlfeat.org/, which has the advantages that it is open
    source, available on all major platforms, and its coordinate system
    matches that used by EVE.  However, the values are not identical to
    those returned by Lowe's own SIFT; see the abovementioned website
    for details.

    Arguments:
         im  the image for which the keypoints are to be found
    program  if supplied, the pathname of the SIFT program
             (default: 'sift %i -o %o 1> /dev/null')
    """
    kptfn, kptfd = sift_run (im, program)
    features = sift_keypoints (kptfn)
    os.close (kptfd)
    return features

#-------------------------------------------------------------------------------
def sift_keypoints (fn):
    """
    Return the SIFT keypoints of an image.

    Two arrays are returned, the first giving the locations (and
    corresponding scales and orientations) of the feature points found,
    while the second contains the associated descriptors.

    Arguments:
    im  name of a file containing the SIFT keypoints
    """
    import scipy.linalg
    # Read in the keypoints from the file.  We read the entire file into memory,
    # where it ends up as a list with one line of the file in each element.
    # Each line contains exactly 132 elements which give the position and
    # orientation of the feature and its descriptor; we split these out into
    # the arrays called locs and descs, normalising the latter en route.  We
    # ultimately return locs and descs.
    fd = open (fn)
    lines = fd.readlines()
    fd.close()
    lf = 128          # length of each descriptor
    nf = len (lines)  # number of features
    if nf == 0: return None, None
    locs = numpy.zeros ((nf, 4))
    descs = numpy.zeros ((nf, lf))
    for f in xrange (0, nf):
        v = lines[f].split()
        p = 0
        # row, col, scale, orientation
        locs[f,1] = float (v[p])
        locs[f,0] = float (v[p+1])
        locs[f,2] = float (v[p+2])
        locs[f,3] = float (v[p+3])
        p += 4
        for i in xrange (0, lf):
            descs[f,i] = float (v[p+i])
        descs[f] = descs[f] / scipy.linalg.norm (descs[f])
    return locs, descs

#-------------------------------------------------------------------------------
def sift_run (im, program='sift %i -o %o 1> /dev/null'):
    """
    Run the SIFT program on an image and return name of the file
    containing the corresponding keypoints.
    Arguments:
         im  the image for which the keypoints are to be found
    program  if supplied, the pathname of the SIFT program
               (default: 'sift %i -o %o 1> /dev/null')
    """
    ny, nx, nc = sizes (im)
    if nc == 1: im1 = im
    else:       im1 = mono (im)
    # Save the image to a temporary file and run SIFT on it, gathering the
    # resulting keypoints into a separate temporary file.
    infd, infn  = tempfile.mkstemp (".pgm")
    kptfd, kptfn = tempfile.mkstemp (".sift")
    output_pnm (im1, infn)
    cmd = re.sub ('%i', infn, program)
    cmd = re.sub ('%o', kptfn, cmd)
    os.system (cmd)
    os.close (infd)
    return kptfn, kptfd

#-------------------------------------------------------------------------------
def susan (im, program='susan %i %o -c 1> /dev/null'):
    """
    Process an image using the SUSAN feature point detector.

    Arguments:
         im  the image for which the keypoints are to be found
    program  if supplied, the pathname of the SIFT program
             (default: 'susan %i %o 1> /dev/null')
    """
    # SUSAN requires a single-channel image.
    ny, nx, nc = sizes (im)
    if nc == 1: im1 = im
    else:       im1 = mono (im)

    # Save the image to a temporary file, run SUSAN on it, and read in the
    # result, which we return.
    handle, infn  = tempfile.mkstemp (".pgm")
    handle, opfn = tempfile.mkstemp (".pgm")
    output_pnm (im1, infn)
    cmd = re.sub ('%i', infn, program)
    cmd = re.sub ('%o', opfn, cmd)
    os.system (cmd)
    susim = image (opfn)
    return susim

#-------------------------------------------------------------------------------
def sizes (im):
    """
    Return the dimensions of an image as a list.

    Arguments:
    im  the image whose dimensions are to be returned
    """
    return im.shape

#-------------------------------------------------------------------------------
def snr (im1, im2):
    """
    Return the signal-to-noise ratio between two images.

    Arguments:
    im1  first image to be used in calculating the SNR
    im2  first image to be used in calculating the SNR
    """
    r = correlation_coefficient (im1, im2)
    if r <= 0.0: return 0.0
    return math.sqrt (r / (1.0 - r))

#-------------------------------------------------------------------------------
def sobel (im):
    """
    Perform edge detection in im using the Sobel operator, returning the
    result.

    Arguments:
    im  image in which the edges are to be found
    """
    import scipy
    import scipy.ndimage as ndimage

    # Convert the EVE-format image into one compatible with scipy, run its
    # Sobel routine, then convert the result back into EVE format and return it.
    ny, nx, nc = sizes (im)
    if nc == 1: sci_im = im[:,:,0]
    else:       sci_im = mono(im)[:,:,0]
    grad_x = ndimage.sobel(sci_im, 0)
    grad_y = ndimage.sobel(sci_im, 1)
    grad_mag = scipy.sqrt(grad_x**2+grad_y**2)
    gm = image ((ny,nx,1))
    gm[:,:,0] = grad_mag[:,:]
    return gm

#-------------------------------------------------------------------------------
def ssd (im1, im2):
    """
    Return the sum-squared difference between two images.

    Arguments:
    im1  image form which im2 is to be subtracted
    im2  image to be subtracted from im1
    """
    return ((im1 - im2)**2).sum()

#-------------------------------------------------------------------------------
def statistics (im):
    """
    Return important statistics of an image.

    This routine returns the minimum, maximum, mean and standard deviation
    as a list.

    Arguments:
    im  image for which the statistics are to be calculated
    """
    lo, hi = extrema (im)
    ave = mean (im)
    sdev = sd (im)
    return [lo, hi, ave, sdev]

#-------------------------------------------------------------------------------
def subsample (im, inc=2):
    """
    Sub-sample an image by selecting every inc-th pixel from every inc-th line.
    The sub-sampled image is returned.

    Arguments:
     im  image to be sub-sampled
    inc  the number of pixels between sub-samples (default: 2)
    """
    ny, nx, nc = sizes (im)
    ny2 = ny // inc
    nx2 = nx // inc
    im2 = image ((ny2, nx2, nc))
    for y in xrange (0, ny2):
        for x in xrange (0, nx2):
            im2[y,x,:] = im[inc*y,inc*x,:]
    return im2

#-------------------------------------------------------------------------------
def sum (im):
    """
    Return the sum of all the values of an image.

    Arguments:
    im  image for which the sum is to be found
    """
    return im.sum()

#-------------------------------------------------------------------------------
def thong (im, scale=64.0, offset=128.0):
    """
    Fill an image with Tran Thong's zone-plate-like test pattern.

    Arguments:
       im  image to contain the pattern (modified)
     scale  maximum deviation of the pattern from the mean
    offset  mean of the resulting pattern
    """
    # Work out the centre of the region and the various fiddle factors.
    ny, nx, nc = sizes (im)
    xc = nx // 2
    yc = ny // 2
    nmax = ny
    if nx > ny: nmax = nx
    rad = 0.4 * nmax
    rad2 = rad / 2.0
    radsqd = rad * rad
    radsq4 = radsqd / 4.0
    fac = 2 * math.pi * 0.496

    # Fill the region with the pattern.
    for y in xrange (0, ny):
        yy = (y - yc) **2
        for x in xrange (0, nx):
            rsqd = (x - xc) **2 + yy
            if rsqd <= radsq4:
                v = scale * math.cos (fac*rsqd/rad) + offset
                for c in xrange (0, nc):
                    im[y,x,c] = v
            else:
                r = math.sqrt (rsqd)
                v = scale * math.cos (fac * (2*r - rsqd/rad - rad2)) + offset
                im[y,x,:] = v

#-------------------------------------------------------------------------------
def transpose (im):
    """
    Transpose an image, returning the result.

    Arguments:
    im  image to be transposed
    """
    ny, nx, nc = sizes (im)
    tr = image ((nx, ny, nc))
    return numpy.transpose (im, axes=(1, 0, 2))

#-------------------------------------------------------------------------------
def variance (im):
    """
    Return the variance of an image.

    Arguments:
    im  image for which the standard deviation is to be found
    """
    return im.var (ddof=1)

#-------------------------------------------------------------------------------
def version ():
    """
    Return the version of the Easy Vision Environment.
    """
    return timestamp[13:-1]

#-------------------------------------------------------------------------------
def version_info (prefix="  ", intro="Modules:"):
    """
    Return a string containing version information.

    Arguments:
    prefix  text to precede each line of text (default: '   ')
     intro  text to precede the output (default: 'Modules:')
    """
    import Image
    fmt = "%s%-9s %s\n" * 6
    s = intro + "\n" + fmt % (prefix, "EVE:", version(),
                              prefix, "numpy:", numpy.__version__,
                              prefix, "Image:", Image.VERSION,
                              prefix, "Python:", platform.python_version (),
                              prefix, "Compiler:", platform.python_compiler (),
                              prefix, "Build:", platform.python_build ()
                              )
    return s

#-------------------------------------------------------------------------------
def zero (im):
    """
    Set all pixels of an image to zero.

    Arguments:
    im  image to be zeroed (modified)
    """
    set (im, 0.0)

#-------------------------------------------------------------------------------
# Main program
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print "There is a separate test script to check that EVE works correctly"
    print "on your platform, available from:\n"
    print "  http://vase.essex.ac/uk/software/eve/\n"

timestamp = "Time-stamp: <2015-03-10 09:20:56 Adrian F Clark (alien@essex.ac.uk)>"

# Local Variables:
# time-stamp-line-limit: -10
# End:
#-------------------------------------------------------------------------------
# End of EVE
#-------------------------------------------------------------------------------

