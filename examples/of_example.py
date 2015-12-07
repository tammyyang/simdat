from simdat.openface import oftools
from simdat.core import plot


# Initialize OpenFace class
of = oftools.OpenFace(pfs=['openface.json', 'ml.json'])
pl = plot.PLOT()


# Get rep of images
# im = tools.IMAGE()
# images = im.find_images()
# of.get_reps(images, output=True)

# inf = 'result_full.json'
# Train
# data, target, target_names = of.read_df(inf, dtype='train')
# of.run(data, target)

# Test
inf = 'result_test.json'
data, target, target_names = of.read_df(inf, dtype='test')
model = of.read_model('/tammy/viscovery/demo/SVC.pkl')
r = of.test(data, target, model, target_names=target_names)
pl.plot_matrix(r['cm'], yticks=target_names)
