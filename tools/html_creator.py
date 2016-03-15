# This is an example to create a html for displaying images
# $python html_creator.py PARENT TEMPLATE
# PARENT: parent folder of the images (default $PWD)
# TEMPLATE: jinja template file
#           (default: simdat/examples/html_plots.template)
import os
import sys
from simdat.core.so import image

imtl = image.IMAGE()

args = sys.argv[1:]
if len(args) > 0:
    parent = args[0]
else:
    parent = os.getcwd()
print("Looking for images in %s" % parent)

temp = '/home/tammy/SOURCES/simdat/examples/html_plots.template'
if len(args) > 1:
    temp = args[1]

outf = parent + '/images.html'
title = 'Images'
imgs = imtl.find_images(dir_path=parent)
imgs.sort()
args = {'TITLE': 'Images',
        'imgs': imgs}
content = imtl.read_template(temp, args)
with open(outf, 'w') as f:
    print('HTML contents are written to %s' % outf)
    f.write(content)
