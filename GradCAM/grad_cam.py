import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image

from model import load_model, INPUT_SHAPE
from os import listdir
from os.path import isfile, join
from load_data import numbers_to_genes

import tensorflow as tf
from tensorflow.python.framework import ops

K_FOLD = 4
H, W = INPUT_SHAPE  # Input shape, defined by the model (model.input_shape)
batch_size = 8
N_FOLDS = 5
N_CLASSES = 5
THRESHOLD = 180


# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.

    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """

    return load_model(K_FOLD)


# ---------------------------------------------------------------------


def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W), color_mode="grayscale")
    x = image.img_to_array(x)
    if preprocess:
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == "th":
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:

        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return (
                grad * tf.cast(grad > 0.0, dtype) * tf.cast(op.inputs[0] > 0.0, dtype)
            )

    g = tf.get_default_graph()
    with g.gradient_override_map({"Relu": "GuidedBackProp"}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]

    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(
        input_model.output, np.dstack([range(images.shape[0]), classes])[0]
    )
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function(
        [input_model.input, K.learning_phase()], [layer_output, grads]
    )

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum("ijkl,il->ijk", conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def compute_saliency(
    model,
    guided_model,
    img_path,
    layer_name="block5_conv3",
    cls=-1,
    visualize=True,
    save=None,
    img_name=str(K_FOLD),
):
    """Compute saliency using all three approaches.
    -layer_name: layer to compute gradients;
    -cls: class number to localize (-1 for most probable class).
    """
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    if cls == -1:
        cls = np.argmax(predictions)

    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]
    if save is not None:
        img_name = img_name + ".jpg"
        path_gradcam = join(save, "gradcam")
        path_guided_backprop = join(save, "guided_backprop")
        path_guided_gradcam = join(save, "guided_gradcam")
        if not os.path.exists(path_gradcam):
            os.makedirs(path_gradcam)
        if not os.path.exists(path_guided_backprop):
            os.makedirs(path_guided_backprop)
        if not os.path.exists(path_guided_gradcam):
            os.makedirs(path_guided_gradcam)
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        cv2.imwrite(join(path_gradcam, img_name), np.uint8(jetcam))
        cv2.imwrite(join(path_guided_backprop, img_name), deprocess_image(gb[0]))
        # jetcam = cv2.applyColorMap(np.uint8(255 * guided_gradcam[0]), cv2.COLORMAP_JET)
        # jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        # cv2.imwrite(join(path_guided_gradcam, picure_name), deprocess_image(jetcam))

        jetcam = cv2.applyColorMap(
            np.uint8(255 * deprocess_image(guided_gradcam[0])), cv2.COLORMAP_WINTER
        )
        cv2.imwrite(join(path_guided_gradcam, img_name), jetcam)

    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title("GradCAM")
        plt.axis("off")
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap="jet", alpha=0.5)

        plt.subplot(132)
        plt.title("Guided Backprop")
        plt.axis("off")
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title("Guided GradCAM")
        plt.axis("off")
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return gradcam, gb, guided_gradcam


def compute_saliency_for_classes(layer_name="last_conv", colormap=True):
    """
    create guided Grad-Cam image for each class and fold
    :param layer_name: layer to compute gradients and GRAD-CAM;
    :param colormap: use the colormap for the result
    :return:
    """
    labels_index = {"0.0": 0, "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4}
    dir_for_gradCam = "data/gradCam"
    if not os.path.exists(dir_for_gradCam):
        os.makedirs(dir_for_gradCam)

    path_to_data = "data/embedding_2d/train"
    path_to_save = "data/gradCamForClasses"
    if not colormap:
        path_to_save = join(path_to_save, "grayscale")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    classes = listdir(path_to_data)
    print(classes)
    model = build_model()
    guided_model = build_guided_model()
    for cl in classes:
        path_to_data_of_class = join(path_to_data, cl)
        imgs = listdir(path_to_data_of_class)
        guided_gradcams = np.zeros((len(imgs), INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
        for i in range(0, len(imgs), batch_size):
            print("i=" + str(i))
            last_i = min(len(imgs), i + batch_size)
            batch_of_imgs = imgs[i:last_i]
            batch_of_imgs = load_batch_of_images(path_to_data_of_class, batch_of_imgs)
            cls = np.repeat(labels_index[cl], (last_i - i))
            gradcam = grad_cam_batch(model, batch_of_imgs, cls, layer_name)
            # gradcam = grad_cam(model, batch_of_imgs, labels_index[cl], layer_name)
            gb = guided_backprop(guided_model, batch_of_imgs, layer_name)
            guided_gradcam = gb * gradcam[..., np.newaxis]
            guided_gradcams[i:last_i] = guided_gradcam
        res = deprocess_image(np.mean(guided_gradcams, axis=0))
        if colormap:
            res = cv2.applyColorMap(np.uint8(255 * res), cv2.COLORMAP_WINTER)
        img_name = "class_" + cl + "_fold=" + str(K_FOLD) + ".jpg"
        cv2.imwrite(join(path_to_save, img_name), res)


def load_batch_of_images(path_to_data, pictures):
    shape = (len(pictures), INPUT_SHAPE[0], INPUT_SHAPE[1], 1)
    batch = np.zeros(shape, dtype=np.float32)
    for i, picture in enumerate(pictures):
        img_path = join(path_to_data, picture)
        batch[i] = load_image(img_path)
    return batch


def compute_saliency_for_imgs():
    labels_index = {"0.0": 0, "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4}
    dir_for_gradCam = "data/gradCam"
    if not os.path.exists(dir_for_gradCam):
        os.makedirs(dir_for_gradCam)

    path_to_data = "data/embedding_2d/train"
    classes = listdir(path_to_data)
    print(classes)
    model = build_model()
    guided_model = build_guided_model()
    for cl in classes:
        path_to_data_of_class = join(path_to_data, cl)
        imgs = listdir(path_to_data_of_class)
        for img in imgs:
            print(img)
            path_for_picture_folder = join(dir_for_gradCam, str(img[:-4]))
            if not os.path.exists(path_for_picture_folder):
                os.makedirs(path_for_picture_folder)
            img_path = join(path_to_data_of_class, img)
            gradcam, gb, guided_gradcam = compute_saliency(
                model,
                guided_model,
                layer_name="last_conv",
                img_path=img_path,
                cls=labels_index[cl],
                visualize=False,
                save=path_for_picture_folder,
                img_name=str(K_FOLD),
            )


def get_important_features(count_best=10):
    path_to_images = "data/gradCamForClasses/grayscale"
    path_to_images_with_clolormap = "data/gradCamForClasses"
    path_to_save = "data/importances"
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    imp_for_all_cl = np.zeros((N_CLASSES,) + INPUT_SHAPE + (1,))
    for cl in range(N_CLASSES):
        res = np.zeros(INPUT_SHAPE + (1,))
        for fold in range(N_FOLDS):
            picture_name = "class_" + str(cl) + ".0_fold=" + str(fold) + ".jpg"
            picture_path = join(path_to_images, picture_name)
            picture = load_image(picture_path, False)
            res += picture
        res /= N_FOLDS
        res /= 255.0
        imp_for_all_cl[cl] = res

        cv2.imwrite(
            join(path_to_images, "class_" + str(cl) + ".jpg"), deprocess_image(res)
        )
        cv2.imwrite(
            join(path_to_images_with_clolormap, "class_" + str(cl) + ".jpg"),
            cv2.applyColorMap(deprocess_image(res), cv2.COLORMAP_WINTER),
        )
        res = res.flatten()
        # get more important genes
        most_importence = np.argsort(res)[-count_best:]
        genes_names = numbers_to_genes(most_importence)

        # print and save results
        genes_with_importance = pd.DataFrame(
            {"names": genes_names, "imp": res[most_importence]}
        )
        print("importent features for class " + str(cl))
        print(genes_with_importance)
        plot = genes_with_importance.plot(
            "names",
            "imp",
            "barh",
            color=plt.cm.RdYlGn(np.linspace(0, 1, len(genes_names))),
            figsize=(12, 7),
            legend=False,
        )
        plot.set_ylabel("Common genes", fontsize=10)
        plot.set_xlabel("Feature importance", fontsize=10)
        # plot.get_figure().show()
        plot.get_figure().savefig(join(path_to_save, "class" + str(cl) + ".png"))

    imp_for_all_cl = np.mean(imp_for_all_cl, axis=0)
    cv2.imwrite(
        join(path_to_images, "for_all_classes" + ".jpg"),
        deprocess_image(imp_for_all_cl),
    )
    cv2.imwrite(
        join(path_to_images_with_clolormap, "for_all_classes" + ".jpg"),
        cv2.applyColorMap(deprocess_image(imp_for_all_cl), cv2.COLORMAP_WINTER),
    )

    imp_for_all_cl = imp_for_all_cl.flatten()
    # get more important genes
    most_importence = np.argsort(imp_for_all_cl)[-count_best:]
    genes_names = numbers_to_genes(most_importence)

    # print and save results
    genes_with_importance = pd.DataFrame(
        {"names": genes_names, "imp": imp_for_all_cl[most_importence]}
    )
    print("importent features for all classes ")
    print(genes_with_importance)
    plot = genes_with_importance.plot(
        "names",
        "imp",
        "barh",
        color=plt.cm.RdYlGn(np.linspace(0, 1, len(genes_names))),
        figsize=(12, 10),
        legend=False,
    )
    plot.set_ylabel("Common biomarkers", fontsize=9)
    plot.set_xlabel("Feature importance", fontsize=9)
    # plot.get_figure().show()
    plot.get_figure().savefig(join(path_to_save, "for_all_class" + ".png"))


def get_importent_areas():
    path_to_imgs = "data/gradCamForClasses/grayscale"
    path_to_save = join("data/gradCamForClasses/", "with_imp")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    n = 6
    shape = (INPUT_SHAPE[0] // n, INPUT_SHAPE[1] // n)
    step = 1
    for cl in range(N_CLASSES):
        picture_name = "class_" + str(cl) + ".jpg"
        picture_path = join(path_to_imgs, picture_name)
        picture = load_image(picture_path, False)[:, :, 0]
        importences = dict()
        for h in range(0, INPUT_SHAPE[0] - shape[0], step):
            for w in range(0, INPUT_SHAPE[1] - shape[1], step):
                importence = importance_of_part(
                    picture[h : h + shape[0], w : w + shape[1]]
                )
                importences[(h, w)] = importence
        # draw areas, the importance of which is greater then threshold
        threshold_for_importance = shape[0] * shape[1] * 2
        importences = {
            k: v for k, v in importences.items() if v > threshold_for_importance
        }

        # sort
        draw_areas = dict()
        sorted_imp = sorted(
            importences.items(), key=lambda kv: (kv[1], kv[0]), reverse=True
        )
        # delete interceptions
        for place, im in sorted_imp:
            inter = False
            for k, v in draw_areas.items():
                if is_intersect(place, k, shape):
                    inter = True
                    break
            if not inter:
                draw_areas[place] = im
        # draw
        print(os.getcwd())
        img = cv2.applyColorMap(deprocess_image(picture), cv2.COLORMAP_WINTER)
        img = cv2.resize(img, (int(img.shape[0] * 2), int(img.shape[1] * 2)))
        for k, v in draw_areas.items():
            img = cv2.circle(
                img,
                ((k[1] + shape[1] // 2) * 2, (k[0] + shape[1] // 2) * 2),
                shape[1],
                (0, 0, 255),
                2,
            )
        cv2.imwrite(join(path_to_save, picture_name), img)

        # drow on each fold
        for fold in range(N_FOLDS):
            img_name = "class_" + str(cl) + ".0_fold=" + str(fold) + ".jpg"
            img = load_image(join(path_to_imgs, img_name), False)[:, :, 0]
            img_colormap = cv2.applyColorMap(deprocess_image(img), cv2.COLORMAP_WINTER)
            img_colormap = cv2.resize(
                img_colormap, (int(img.shape[0] * 2), int(img.shape[1] * 2))
            )
            for k, v in draw_areas.items():
                importence = importance_of_part(
                    img[k[0] : k[0] + shape[0], k[1] : k[1] + shape[1]]
                )
                if importence > (v / 2):
                    img_colormap = cv2.circle(
                        img_colormap,
                        ((k[1] + shape[1] // 2) * 2, (k[0] + shape[1] // 2) * 2),
                        shape[1],
                        (0, 0, 255),
                        2,
                    )

            cv2.imwrite(join(path_to_save, img_name), img_colormap)


def draw_grid():
    n_clases = 5
    n_folds = 5
    start_x = 150
    start_y = 60
    retreat = 20

    path_to_images = join("data/gradCamForClasses/", "with_imp")
    images_names = np.array(listdir(path_to_images)[5:])
    images_names = images_names.reshape((n_clases, n_folds))
    shape = cv2.imread(join(path_to_images, images_names[0, 0]), cv2.IMREAD_COLOR).shape
    img = np.zeros(
        (
            (shape[0] + retreat) * n_folds + start_y,
            (shape[1] + retreat) * n_clases + start_x,
            shape[2],
        )
    )
    img[:, :] = 255
    # add images to grid
    for i in range(n_folds):
        for j in range(n_clases):
            pos_y = start_y + i * (shape[0] + retreat)
            pos_x = start_x + j * (shape[1] + retreat)
            li = cv2.imread(join(path_to_images, images_names[j, i]), cv2.IMREAD_COLOR)
            img[pos_y : pos_y + shape[0], pos_x : pos_x + shape[1], :] = li

    # add text to grid
    clases_names = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]
    for i in range(n_clases):
        cv2.putText(
            img=img,
            text=clases_names[i],
            org=(start_x + i * (shape[1] + retreat) + 70, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=3,
            color=(0, 0, 0),
        )
    for i in range(n_folds):
        cv2.putText(
            img=img,
            text="Fold:" + str(i + 1),
            org=(0, start_y + i * (shape[1] + retreat) + 150),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=3,
            color=(255, 150, 100),
        )
    cv2.imwrite("grid.png", img)


def is_intersect(start_point1, start_point2, shape):
    """
    :param start_point1: position of left-up corner of first rectangle
    :param start_point2: position of left-up corner of second rectangle
    :param shape: length and width of the rectangle
    :return: whether 2 rectangles is intersecting
    """
    if (
        (
            start_point1[0] <= start_point2[0] <= start_point1[0] + shape[0]
            and start_point1[1] <= start_point2[1] <= start_point1[1] + shape[1]
        )
        or (
            (
                start_point2[0] <= start_point1[0] <= start_point2[0] + shape[0]
                and start_point1[1] <= start_point2[1] <= start_point1[1] + shape[1]
            )
        )
        or (
            (
                start_point1[0] <= start_point2[0] <= start_point1[0] + shape[0]
                and start_point2[1] <= start_point1[1] <= start_point2[1] + shape[1]
            )
        )
        or (
            (
                start_point2[0] <= start_point1[0] <= start_point2[0] + shape[0]
                and start_point2[1] <= start_point1[1] <= start_point2[1] + shape[1]
            )
        )
    ):
        return True
    else:
        return False


def importance_of_part(area):
    imp = 0
    for row in area:
        for pixel in row:
            if pixel > THRESHOLD:
                imp += pixel - THRESHOLD
    return imp


if __name__ == "__main__":
    for K_FOLD in range(0, 5):
        compute_saliency_for_classes(colormap=False)
    get_important_features()
    get_importent_areas()
    draw_grid()
