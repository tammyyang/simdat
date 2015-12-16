import sys
import numpy as np
from simdat.openface import oftools
from simdat.core import tools
from simdat.core import plot
from simdat.core import ml

"""
This is an example to run openface and classify the results.

$python of_example ACTION METHOD/PCAMETHOD THRESHOLD/NCOMP

ACTION: action to do
    1. rep
    2. train(default)
       METHOD: method of classifier, SVC(default)/RF/Neighbors
    3. test
       METHOD: method of classifier, SVC(default)/RF/Neighbors
       THRESHOLD: throshold of testing accuracy
                  should be a float between 0-1 (default: 0.4)
    4. pca
       PCAMETHOD: method of PCA, PCA(default)/Randomized/Sparse
       NCOMP: n_components, 1 or 2 (default: 2)

Tune other parameters in openface.json and ml.json
"""

pfs = ['openface.json', 'ml.json']
mpf = '/home/tammy/viscovery/demo/db/models/train_homography/mapping.json'
im = tools.IMAGE()
io = tools.MLIO()
pl = plot.PLOT()
mltl = ml.MLTools()
of = oftools.OFTools()

args = sys.argv[1:]

def pick_images():
    root = '/home/tammy/viscovery/demo/db/'
    dbs = [root + 'train/train_homography.json',
           root + 'tests/tests_homography.json',
           root + 'tests/tests_20151216_homography.json']
    return of.pick_reps(dbs)

def _pca(df, ncomp=2, pca_method='PCA'):
    res = of.read_df(df, dtype='train', group=False, conv=False)
    p = df['class'].value_counts().idxmax()
    data = res['data']
    fname = p + '_pca.png'
    if ncomp == 1:
        pca_data = mltl.PCA(data, ncomp=1,
                            method=pca_method)
        pca_data = np.array(pca_data).T[0]
        pl.histogram(pca_data, fname=fname)
        return p, pca_data
    else:
        pca_data = mltl.PCA(data, method=pca_method)
        pca_data = np.array(pca_data).T
        pl.plot_points(pca_data[0], pca_data[1], fname=fname,
                       xmin=-1, xmax=1, ymin=-1, ymax=1)
        return p, [pca_data[0], pca_data[1]]


act = 'pick'
if len(args) > 0:
    act = args[0]
print("Action: %s" % act)

if act in ['train', 'test']:
    method = 'SVC'
    if len(args) > 1:
        method = args[1]

    if method == 'RF':
        ml = ml.RFRun(pfs=pfs)
    elif method == 'Neighbors':
        ml = ml.NeighborsRun(pfs=pfs)
    else:
        ml = ml.SVMRun(pfs=pfs)

if act == 'rep':
    of = oftools.OpenFace(pfs=pfs)
    images = im.find_images()
    of.get_reps(images, output=True)

elif act == 'pick':
    pca_method = 'PCA'
    if len(args) > 1:
        pca_method = args[1]
    print('pca_method = %s' % pca_method)
    ncomp = 2
    if len(args) > 2:
        ncomp = int(args[2])
    print('ncomp = %i' % ncomp)

    mapping = io.parse_json(mpf)
    df = pick_images()
    all_data = []
    labels = []
    for p in mapping.keys():
        _df = df[df['class'] == p]
        p, data = _pca(_df, ncomp=ncomp, pca_method=pca_method)
        all_data.append(data)
        labels.append(p)
    if ncomp == 1:
        pl.plot_1D_dists(all_data, legend=labels)
    else:
        pl.plot_classes(all_data, legend=labels)

elif act == 'train':
    root = '/tammy/viscovery/demo/db/'
    inf = root + 'train/train_homography.json'
    res = of.read_df(inf, dtype='train', group=False)
    mf = ml.run(res['data'], res['target'])

elif act == 'test':
    from datetime import date
    thre = 0.4
    if len(args) > 2:
        thre = float(args[2])
    print('Threshold applied %.2f' % thre)
    root = '/home/tammy/viscovery/demo/db/'
    mf = root + 'models/train_homography/' + method + '.pkl'
    # mf = "/tammy/viscovery/demo/20151126/full/outDir/classifier.pkl"
    inf = root + 'tests/tests_homography.json'
    print('Reading model from %s' % mf)
    print('Reading db from %s' % inf)
    print('Reading mappings from %s' % mpf)
    # res = of.read_df(inf, dtype='test', mpf=mpf, group=True)
    df = pick_images()
    res = of.read_df(df, dtype='test', mpf=mpf, group=True)
    model = ml.read_model(mf)
    # model = model[1]
    match = 0
    nwrong = 0
    today = date.today().strftime("%Y%m%d")
    new_home = '/home/tammy/viscovery/demo/images/matched/' + today
    for i in range(0, len(res['data'])):
        r1 = ml.test(res['data'][i], res['target'][i], model,
                     target_names=res['target_names'])
        cat = res['target'][i][0]
        found = False
        mis_match = False
        if r1['prob'] is None:
            for p in range(0, len(r1['predicted'])):
                if cat == r1['predicted'][p]:
                    path = res['path'][i].replace('/tammy', '/home/tammy')
                    pl.patch_rectangle_img(path, res['pos'][i][p],
                                           new_name=None)
                    found = True
        else:
            for p in range(0, len(r1['prob'])):
                prob = r1['prob'][p]
                vmax = max(prob)
                imax = prob.argmax()
                if vmax > thre:
                    if imax == cat:
                        path = res['path'][i].replace('/tammy', '/home/tammy')
                        pl.patch_rectangle_img(path, res['pos'][i][p],
                                               new_home=new_home)
                        found = True
                    else:
                        mis_match = True
        if found:
            match += 1
        if mis_match:
            nwrong += 1
    print('Matched rate %.2f' % (float(match)/float(len(res['data']))))
    print('Mis-matched rate %.2f' % (float(nwrong)/float(len(res['data']))))


