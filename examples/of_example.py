from simdat.openface import oftools
from simdat.core import tools

im = tools.IMAGE()
of = oftools.OpenFace()

images = im.find_images(dir_path="/tammy/viscovery/demo/images/person-1/notlike/")
of.get_reps(images, output=True)

