import cv2
import numpy as np
from config import compiled_model_ic, output_layer_ic


def cv_image_classify(raw_image):
    '''
    Classify animal in image
    :param
    raw_image(ndarray): image object of the input image

    :return
    class_result(str): Name of the animal class
    '''
    #global imagenet_classes
    imagenet_classes = open("utils/imagenet_2012.txt").read().splitlines()
    # The MobileNet model expects images in RGB format.
    image = to_rgb(raw_image)

    # Resize to MobileNet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))

    # Reshape to model input shape.
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model_ic([input_image])[output_layer_ic]
    result_index = np.argmax(result_infer)
    # Convert the inference result to a class name.
    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes

    class_result = imagenet_classes[result_index].split()[1:]
    class_result = " ".join(class_result)
    return class_result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)