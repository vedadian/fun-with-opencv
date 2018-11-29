#!/usr/bin/env python
# coding=utf8

# Copyright (c) 2018 Behrooz Vedadian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Some fun transformations applied to a sample image
"""

from __future__ import print_function, division, unicode_literals

import math
import codecs
import sys
import cv2
import numpy as np
from matplotlib import pyplot

# Make the program look alike in python versions 3 and 2
if sys.version_info < (3, 0):
    sys.stdin = codecs.getreader("utf8")(sys.stdin)
    sys.stdout = codecs.getwriter("utf8")(sys.stdout)
    sys.stderr = codecs.getwriter("utf8")(sys.stderr)

def main():
    
    def show(*images, **kwargs):
        fig = pyplot.figure() 
        if 'title' in kwargs:
            fig.canvas.set_window_title(kwargs['title']) 
        r = math.floor(math.sqrt(len(images)))
        c = math.ceil(len(images) / r)
        i = 1
        shared_axes = None
        for image in images:
            if isinstance(image, tuple):
                image, title = image
            else:
                title = ''
            if r > 1 or c > 1:
                axes = pyplot.subplot(r, c, i, sharex=shared_axes, sharey=shared_axes)
                if shared_axes is None:
                    shared_axes = axes
            if title:
                pyplot.title(title)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pyplot.imshow(image, cmap='gray')
            pyplot.xticks([])
            pyplot.yticks([])
            i += 1
        
        pyplot.show()

    image = cv2.imread('../robert_de_niro.jpg')

    def blur_play():
        simple = cv2.blur(image, (21, 21))
        gaussian = cv2.GaussianBlur(image, (21, 21), 0)
        median = cv2.medianBlur(image, 21)
        show(
            (image, "Original"),
            (simple, "Simple"),
            (gaussian, "Gaussian"),
            (median, "Median"),
            title='Blurring'
        )

    def edge_play():
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_32F, 1, 0)).mean(2)
        laplacian = np.absolute(cv2.Laplacian(image, cv2.CV_32F)).mean(2)
        canny = cv2.Canny(image, 100, 200)
        show(
            (image, "Original"),
            (sobel, "Sobel"),
            (laplacian, "Laplacian"),
            (canny, "Canny"),
            title='Finding edges'
        )


    def edge_aware_blur():
        
        edges = np.absolute(cv2.Sobel(image, cv2.CV_32F, 1, 0)).mean(2)
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        edges = edges ** 0.25
        edges = np.expand_dims(edges, 2)
        edges = np.tile(edges, (1, 1, 3))

        blurred = cv2.medianBlur(image, 21)
        final = (1 - edges) * blurred + edges * image
        final = final.astype(np.uint8)

        show(
            (image, "Original"),
            (blurred, "Just Blurred"),
            (final, "Smoothed"),
            title="Edge aware smoothing"
        )

    blur_play()
    edge_play()
    edge_aware_blur()

if __name__ == "__main__":
    main()