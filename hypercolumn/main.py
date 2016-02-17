from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import image

models = dp_models.DPModel()
im = image.IMAGE()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'airportwaitingarea_0001.jpg'

model = models.VGG_16(weight_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

img = im.read(img_path, size=(224, 224))
# im = im_original.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)
# im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
# plt.imshow(im_converted)

