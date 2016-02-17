import cv2
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image

models = dp_models.DPModel()
imnet = dp_models.ImageNet()
im = image.IMAGE()
pl = plot.PLOT()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'ILSVRC2012_test_00099995.JPEG'

model = models.VGG_16(weight_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

img_original = im.read(img_path, size=(224, 224))
# img_converted = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
# pl.plot_matrix(img_converted, fname='img_converted.png', norm=False,
#                show_text=False, show_axis=False)
img = img_original.transpose((2, 0, 1))
img = np.expand_dims(img, axis=0)
out = model.predict(img)
prob = out.ravel()
pl.plot(out.ravel())

imagenet_labels_filename = '/tammy/ImageNet/synset_words.txt'
results = imnet.find_topk(prob, fname=imagenet_labels_filename)
print(results)
