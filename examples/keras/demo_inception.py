from simdat.core import dp_models
dp = dp_models.DPModel()
model = dp.Inception_v3()
print(model.summary())
# model.compile('rmsprop', 'categorical_crossentropy')
