import os
import time
import argparse
import numpy as np
from random import shuffle
from simdat.core import dp_models
from simdat.core import image
from simdat.core import tools
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def add_traiining_args(train_parser):
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


def print_precision_recall(precision, recall, total):
    for item in precision:
        print('[finetune_vgg] Item %s' % item)
        if recall[item] == 0:
            p = -1
        else:
            p = float(precision[item])/float(recall[item])
        if item in total and total[item] != 0:
            r = float(precision[item])/float(total[item])
        else:
            r = -1
        print('[finetune_vgg]     precision = %.2f' % p)
        print('[finetune_vgg]     recall = %.2f' % r)


def add_prediction_args(predict_parser):
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
    predict_parser.add_argument(
        "--cm", default=False, action='store_true',
        help="Draw confusion matrix."
        )


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
        "--img-width", type=int, default=224, dest='width',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-height", type=int, default=224, dest='height',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed, default: 1337."
        )

    predict_parser = subparsers.add_parser(
        "predict", help='Predict the images.'
        )
    add_prediction_args(predict_parser)

    batch_train_parser = subparsers.add_parser(
        "batch-train", help='Command to train with batches.'
        )
    add_traiining_args(batch_train_parser)
    batch_train_parser.add_argument(
        "--size", type=int, default=5000,
        help="Size of the image batch (default: 5,000)"
        )

    finetune_parser = subparsers.add_parser(
        "train", help='Command to finetune the images.'
        )
    add_traiining_args(finetune_parser)

    crop_parser = subparsers.add_parser(
        "augmentation", help='Generate scroped images.'
    )

    t0 = time.time()
    tl = tools.DATA()
    simdat_im = image.IMAGE()
    mdls = dp_models.DPModel()

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.sbp_name in ['train', 'predict', 'batch-train']:
        tl.check_dir(args.ofolder)
        path_model = os.path.join(args.ofolder, 'model.json')
        path_weights = os.path.join(args.ofolder, 'weights.h5')
        path_cls = os.path.join(args.ofolder, 'classes.json')

    if args.sbp_name == 'batch-train':
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
            print("[finetune_vgg] Epoch %d/%d" % (e+1, args.epochs))
            for i in range(len(imgs)/args.size + 1):
                start = i*args.size
                end = ((i + 1)*args.size)
                files = imgs[start:end]
                shuffle(files)
                if (i + 1)*args.size > len(imgs):
                    end = len(imgs)
                X_train, X_test, Y_train, Y_test, _c = mdls.prepare_data_train(
                    files, args.width, args.height,
                    classes=classes, rc=args.rc)
                model.fit(X_train, Y_train, batch_size=args.batchsize,
                          nb_epoch=1, show_accuracy=True, verbose=1,
                          validation_data=(X_test, Y_test))

        t0 = tl.print_time(t0, 'fit')

        tl.write_json(classes, fname=path_cls)
        json_string = model.to_json()
        open(path_model, 'w').write(json_string)
        model.save_weights(path_weights, overwrite=True)

    elif args.sbp_name == 'train':

        scale = True
        if args.augmentation:
            scale = False
        X_train, X_test, Y_train, Y_test, classes = mdls.prepare_data_train(
            args.path, args.width, args.height, rc=args.rc, scale=scale)
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
        print('[finetune_vgg] Test score:', score[0])
        print('[finetune_vgg] Test accuracy:', score[1])
        t0 = tl.print_time(t0, 'evaluate')

        json_string = model.to_json()
        open(path_model, 'w').write(json_string)
        model.save_weights(path_weights, overwrite=True)

    elif args.sbp_name == 'predict':
        cls_map = tl.parse_json(path_cls)
        model = model_from_json(open(path_model).read())
        t0 = tl.print_time(t0, 'load model from json')

        model.load_weights(path_weights)
        t0 = tl.print_time(t0, 'load model weights')

        if args.cm:
            from simdat.core import plot
            from sklearn.metrics import confusion_matrix
            pl = plot.PLOT()

            X_test, Y_test, classes, F = mdls.prepare_data_test(
                args.path, args.width, args.height, convert_Y=False,
                y_as_str=False, classes=cls_map)
            t0 = tl.print_time(t0, 'prepare data')
            results = model.predict_classes(
                X_test, batch_size=args.batchsize, verbose=1)
            cm = confusion_matrix(Y_test, results)
            pl.plot_confusion_matrix(cm, xticks=cls_map, yticks=cls_map,
                                     xrotation=90)

        else:
            X_test, Y_test, classes, F = mdls.prepare_data_test(
                args.path, args.width, args.height)
            t0 = tl.print_time(t0, 'prepare data')

            results = model.predict_proba(
                X_test, batch_size=args.batchsize, verbose=1)
            outputs = []
            precision = dict((el, 0) for el in cls_map)
            recall = dict((el, 0) for el in cls_map)
            total = dict((el, 0) for el in classes)
            for i in range(0, len(F)):
                _cls = results[i].argmax()
                max_prob = results[i][_cls]
                outputs.append({'input': F[i], 'max_probability': max_prob})
                cls = cls_map[_cls]
                recall[cls] += 1
                total[Y_test[i]] += 1
                if max_prob >= args.threshold:
                    outputs[-1]['class'] = cls
                    if Y_test[i] == cls:
                        precision[cls] += 1
                    else:
                        print('[finetune_vgg] %s: %s (%.2f)'
                              % (F[i], cls, max_prob))
                else:
                    print('[finetune_vgg] %s: low probability (%.2f),'
                          ' cannot find a match' % (F[i], max_prob))
                    outputs[-1]['class'] = None
            tl.write_json(outputs, fname=args.output_loc)
            print_precision_recall(precision, recall, total)

    elif args.sbp_name == 'augmentation':
        fimgs = simdat_im.find_images(dir_path=args.path)
        for fimg in fimgs:
            imgs = simdat_im.read_and_random_crop(fimg, save=True)

    else:
        print('Wrong command.')
        parser.print_help()

if __name__ == '__main__':
    main()
