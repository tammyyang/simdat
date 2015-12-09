from simdat.openface import oftools
from simdat.core import tools

im = tools.IMAGE()
of = oftools.OpenFace(pfs=['openface.json', 'ml.json'])

inf = 'result.json'

# Create representations
# images = im.find_images()
# of.get_reps(images, output=True)

# Train
# res = of.read_df(inf, dtype='train', group=False)
# mf = of.run(res['data'], res['target'])


# Test
import cv2
import numpy as np
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
        imax = np.argmax(prob)
        if vmax > thre and imax == cat:
            path = res['path'][i]
            img = cv2.imread(path)
            pos = res['pos'][i][p]
            cv2.rectangle(img, (pos[0], pos[1]),
                          (pos[2], pos[3]), (0, 255, 0), 2)
            newname = path.replace('.jpg', '_rec.jpg')
            cv2.imwrite(newname, img)
            found = True
    if found:
        match += 1
print('Matched rate %.2f' % (float(match)/float(len(res['data']))))
