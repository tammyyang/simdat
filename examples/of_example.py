from simdat.openface import oftools
from simdat.core import tools
from simdat.core import plot

im = tools.IMAGE()
pl = plot.PLOT()
of = oftools.OpenFace(pfs=['openface.json', 'ml.json'])

inf = 'result.json'

# Create representations
# images = im.find_images()
# of.get_reps(images, output=True)

# Train
# res = of.read_df(inf, dtype='train', group=False)
# mf = of.run(res['data'], res['target'])


# Test
thre = 0.4
mf = '/tammy/viscovery/demo/images/train/output/SVC.pkl'
mpf = '/tammy/viscovery/demo/images/train/output/mapping.json'
res = of.read_df(inf, dtype='test', mpf=mpf, group=True)
model = of.read_model(mf)
match = 0
for i in range(0, len(res['data'])):
    r = of.test(res['data'][i], res['target'][i], model,
                target_names=res['target_names'])
    cat = res['target'][i][0]
    found = False
    for p in range(0, len(r['prob'])):
        prob = r['prob'][p]
        vmax = max(prob)
        imax = prob.argmax()
        if vmax > thre and imax == cat:
            path = res['path'][i]
            pl.patch_rectangle_img(res['path'][i],
                                   res['pos'][i][p], new_name=None)
            found = True
    if found:
        match += 1
print('Matched rate %.2f' % (float(match)/float(len(res['data']))))
