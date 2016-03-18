import os
import time
import numpy as np
import argparse
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image
from simdat.core import ml

im = image.IMAGE()

class DEMOArgs(ml.MLArgs):
    def _add_args(self):
        self._add_ml_args()
        self.weight_path = None
        self.imagenet_labels = None

class DEMO(dp_models.DPModel):
    def dpmodel_init(self):
        self.t0 = time.time()
        self.imnet = dp_models.ImageNet()
        self.pl = plot.PLOT()
        self.args = DEMOArgs(pfs=['/docker/vgg_demo.json'])
        self.model = self.VGG_16(self.args.weight_path)
        self.t0 = self.pl.print_time(self.t0, 'initiate')

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.t0 = self.pl.print_time(self.t0, 'compile model')

    def _predict(self, fimg, plot=False):
        print('Processing %s' % fimg)
        name, ext = os.path.splitext(fimg)
        img_original = im.read(fimg, size=(224, 224))
        img = img_original.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        out = self.model.predict(img)
        prob = out.ravel()
        if plot:
            self.pl.plot(out.ravel())
        result = self.imnet.find_topk(prob, fname=self.args.imagenet_labels)
        return result

    def predict(self, imgs, plot=False):
        results = {}
        for fimg in imgs:
            result = self._predict(fimg, plot=plot)
            results[os.path.basename(fimg)] = result
        self.t0 = self.pl.print_time(self.t0, 'compute for all images')
        print(results)
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Simple Keras Demo"
        )
    parser.add_argument(
        "-w", "--workdir", type=str, default="/docker/images/tests/",
        help="Working directory where the images are stored." +
             " (default: /docker/images/tests/)"
        )
    args = parser.parse_args()

    images = im.find_images(dir_path=args.workdir)
    demo = DEMO()
    demo.predict(images)

if __name__ == '__main__':
    main()

