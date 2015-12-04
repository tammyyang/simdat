from simdat.openface import oftools
from simdat.core import tools


# Initialize OpenFace class
of = oftools.OpenFace(pfs=['openface.json', 'ml.json'])


# Get rep of images
# im = tools.IMAGE()
# images = im.find_images()
# of.get_reps(images, output=True)

# Test
inf = 'result.json'
data, target = of.read_df(inf, dtype='test')
model = of.read_model('/tammy/SOURCES/simdat/examples/output/SVC.pkl')
of.test(data, target, model)
