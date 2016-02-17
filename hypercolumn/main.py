import cv2
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image

models = dp_models.DPModel()
im = image.IMAGE()
pl = plot.PLOT()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'airportwaitingarea_0001.jpg'

model = models.VGG_16(weight_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

img_original = im.read(img_path, size=(224, 224))
# img_converted = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
# pl.plot_matrix(img_converted, fname='img_converted.png', norm=False,
#                show_text=False, show_axis=False)
img = img_original.transpose((2,0,1))
img = np.expand_dims(img, axis=0)
out = model.predict(img)
pl.plot(out.ravel())
# plt.imshow(img_converted)
# im.save(img_converted, fname='converted')
