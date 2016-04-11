import os
import time
import argparse
import numpy as np
from simdat.core import dp_models
from simdat.core import image
from keras.optimizers import SGD
from simdat.core import tools
from keras.layers.core import Dense, Activation
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

simdat_im = image.IMAGE()
mdls = dp_models.DPModel()


def main():
    parser = argparse.ArgumentParser(
        description="Use Simple model to train a classifier."
        )
    subparsers = parser.add_subparsers(
        help='commands', dest='sbp_name'
    )
    parser.add_argument(
        "-p", "--path", type=str, default='.',
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "--img-rows", type=int, default=224, dest='rows',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-cols", type=int, default=224, dest='cols',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed, default: 1337."
        )

    predict_parser = subparsers.add_parser(
        "predict", help='Predict the images.'
        )
    predict_parser.add_argument(
        "--model-loc", type=str, default=os.getcwd(), dest='ofolder',
        help="Path of the folder to output or to load the model."
        )
    predict_parser.add_argument(
        "--batch-size", type=int, default=80, dest='batchsize',
        help="Size of the mini batch. Default: 80."
        )
    predict_parser.add_argument(
        "--input", type=str, default=None,
        help="Input image to be predicted, this overwrites --path option."
        )
    predict_parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Threshold applied to judge whether it is identified correctly."
        )
    predict_parser.add_argument(
        "--output-loc", type=str, dest='output_loc',
        default='/home/tammy/www/prediction.json',
        help="Path to store the prediction results."
        )

    batch_train_parser = subparsers.add_parser(
        "batch-train", help='Command to train with batches.'
        )
    batch_train_parser.add_argument(
        "-v", "--vgg-weights", type=str, dest='weights',
        default='/home/tammy/SOURCES/keras/examples/vgg16_weights.h5',
        help="Path of vgg weights"
        )
    batch_train_parser.add_argument(
        "--model-loc", type=str, default=os.getcwd(), dest='ofolder',
        help="Path of the folder to output or to load the model."
        )
    batch_train_parser.add_argument(
        "--batch-size", type=int, default=80, dest='batchsize',
        help="Size of the mini batch. Default: 80."
        )
    batch_train_parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )
    batch_train_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate of SGD, default 0.001."
        )
    batch_train_parser.add_argument(
        "--lr-decay", type=float, default=1e-6, dest='lrdecay',
        help="Decay of SGD lr, default 1e-6."
        )
    batch_train_parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum of SGD lr, default 0.9."
        )
    batch_train_parser.add_argument(
        "--size", type=int, default=10000,
        help="Size of the image batch (default: 10,000)"
        )
    group1 = batch_train_parser.add_mutually_exclusive_group()
    group1.add_argument(
        "--rc", default=False, action='store_true',
        help="Randomly crop the images (default: False)."
        )
    group1.add_argument(
        "--augmentation", default=False, action='store_true',
        help="True to use ImageDataGenerator."
        )

    train_parser = subparsers.add_parser(
        "train", help='Command to finetune the images.'
        )
    train_parser.add_argument(
        "-v", "--vgg-weights", type=str, dest='weights',
        default='/home/tammy/SOURCES/keras/examples/vgg16_weights.h5',
        help="Path of vgg weights"
        )
    train_parser.add_argument(
        "--model-loc", type=str, default=os.getcwd(), dest='ofolder',
        help="Path of the folder to output or to load the model."
        )
    train_parser.add_argument(
        "--batch-size", type=int, default=80, dest='batchsize',
        help="Size of the mini batch. Default: 80."
        )
    train_parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )
    train_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate of SGD, default 0.001."
        )
    train_parser.add_argument(
        "--lr-decay", type=float, default=1e-6, dest='lrdecay',
        help="Decay of SGD lr, default 1e-6."
        )
    train_parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum of SGD lr, default 0.9."
        )
    group = train_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rc", default=False, action='store_true',
        help="Randomly crop the images (default: False)."
        )
    group.add_argument(
        "--augmentation", default=False, action='store_true',
        help="True to use ImageDataGenerator."
        )

    crop_parser = subparsers.add_parser(
        "augmentation", help='Generate scroped images.'
    )

    t0 = time.time()
    tl = tools.DATA()

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.sbp_name in ['train', 'predict', 'batch-train']:
        path_model = os.path.join(args.ofolder, 'model.json')
        path_weights = os.path.join(args.ofolder, 'weights.h5')
        path_cls = os.path.join(args.ofolder, 'classes.json')

    if args.sbp_name == 'batch-train':
        from random import shuffle
        imgs = simdat_im.find_images(dir_path=args.path)
        classes = simdat_im.find_folders(dir_path=args.path)

        model = mdls.VGG_16(args.weights, lastFC=False)
        sgd = SGD(lr=args.lr, decay=args.lrdecay,
                  momentum=args.momentum, nesterov=True)
        print('[finetune_vgg] lr = %f, decay = %f, momentum = %f'
              % (args.lr, args.lrdecay, args.momentum))

        print('[finetune_vgg] Adding Dense(nclasses, activation=softmax).')
        model.add(Dense(len(classes), activation='softmax'))
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        t0 = tl.print_time(t0, 'compile the model to be fine tuned.')

        shuffle(imgs)
        for e in range(args.epochs):
            print("[finetune_vgg] Epoch %d/%d" % (e, args.epochs))
            for i in range(len(imgs)/args.size + 1):
                start = i*args.size
                end = ((i + 1)*args.size)
                if (i + 1)*args.size > len(imgs):
                    end = len(imgs)
                X_train, X_test, Y_train, Y_test, _c = mdls.prepare_data_train(
                    imgs[start:end], args.rows, args.cols,
                    classes=classes, rc=args.rc)
                model.fit(X_train, Y_train, batch_size=args.batchsize,
                          nb_epoch=1, show_accuracy=True, verbose=1,
                          validation_data=(X_test, Y_test))

        t0 = tl.print_time(t0, 'fit')

    elif args.sbp_name == 'train':
        tl.check_dir(args.ofolder)

        scale = True
        if args.augmentation:
            scale = False
        X_train, X_test, Y_train, Y_test, classes = mdls.prepare_data_train(
            args.path, args.rows, args.cols, rc=args.rc, scale=scale)
        tl.write_json(classes, fname=path_cls)
        nclasses = len(classes)
        t0 = tl.print_time(t0, 'prepare data')

        model = mdls.VGG_16(args.weights, lastFC=False)
        sgd = SGD(lr=args.lr, decay=args.lrdecay,
                  momentum=args.momentum, nesterov=True)
        print('[finetune_vgg] lr = %f, decay = %f, momentum = %f'
              % (args.lr, args.lrdecay, args.momentum))

        print('[finetune_vgg] Adding Dense(nclasses, activation=softmax).')
        model.add(Dense(nclasses, activation='softmax'))
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        t0 = tl.print_time(t0, 'compile the model to be fine tuned.')

        for stack in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            for l in mdls.layers[stack]:
                l.trainable = False

        if args.augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=False,
                featurewise_std_normalization=True,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False)

            datagen.fit(X_train)
            model.fit_generator(
                datagen.flow(X_train, Y_train, batch_size=args.batchsize),
                samples_per_epoch=X_train.shape[0],
                nb_epoch=args.epochs, show_accuracy=True,
                validation_data=(X_test, Y_test),
                nb_worker=1)

        else:
            model.fit(X_train, Y_train, batch_size=args.batchsize,
                      nb_epoch=args.epochs, show_accuracy=True, verbose=1,
                      validation_data=(X_test, Y_test))
        t0 = tl.print_time(t0, 'fit')
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        t0 = tl.print_time(t0, 'evaluate')

        json_string = model.to_json()
        open(path_model, 'w').write(json_string)
        model.save_weights(path_weights, overwrite=True)

    elif args.sbp_name == 'predict':
        X_test, Y_test, classes, F = mdls.prepare_data_test(
            args.path, args.rows, args.cols)
        t0 = tl.print_time(t0, 'prepare data')

        model = model_from_json(open(path_model).read())
        t0 = tl.print_time(t0, 'load model from json')

        model.load_weights(path_weights)
        t0 = tl.print_time(t0, 'load model weights')

        cls_map = tl.parse_json(path_cls)
        results = model.predict_proba(
            X_test, batch_size=args.batchsize, verbose=1)
        outputs = []
        for i in range(0, len(F)):
            _cls = results[i].argmax()
            max_prob = results[i][_cls]
            outputs.append({'input': F[i], 'max_probability': max_prob})
            if max_prob >= 0.7:
                cls = [b for b in cls_map if cls_map[b] == _cls][0]
                print('%s: %s' % (F[i], cls))
                outputs[-1]['class'] = cls
            else:
                print('Low probability (%.2f), cannot find a match' % max_prob)
                outputs[-1]['class'] = None
        tl.write_json(outputs, fname=args.output_loc)

    elif args.sbp_name == 'augmentation':
        fimgs = simdat_im.find_images(dir_path=args.path)
        for fimg in fimgs:
            imgs = simdat_im.read_and_random_crop(fimg, save=True)

    else:
        print('Wrong command.')
        parser.print_help()

if __name__ == '__main__':
    main()
