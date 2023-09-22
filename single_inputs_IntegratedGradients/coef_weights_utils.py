import sys

import numpy as np
from keras import backend as K
from keras.engine import InputLayer
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from single_inputs_IntegratedGradients.model_utils import get_layers


def predict(model, X, loss=None):
    prediction_scores = model.predict(X)

    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
    if loss == 'hinge':
        prediction = np.where(prediction_scores >= 0.0, 1., 0.)
    else:
        prediction = np.where(prediction_scores >= 0.5, 1., 0.)

    return prediction





def get_deep_explain_scores(model, X_train, y_train, target=-1, method_name='grad*input', detailed=False, **kwargs):
    # gradients_list = []
    # gradients_list_sample_level = []

    gradients_list = {}
    gradients_list_sample_level = {}

    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('input'):

            if target is None:
                output = i
            else:
                output = target

            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name=method_name)
            if gradients.ndim > 1:

                print('gradients.shape', gradients.shape)

                feature_weights = np.sum(gradients, axis=-2)

                print('feature_weights.shape', feature_weights.shape)
                print('feature_weights min max', min(feature_weights), max(feature_weights))
            else:

                feature_weights = gradients


            gradients_list[l.name] = feature_weights
            gradients_list_sample_level[l.name] = gradients
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass



def get_deep_explain_score_layer(model, X, layer_name, output_index=-1, method_name='grad*input'):
    scores = None
    import keras
    from single_inputs_IntegratedGradients.tensorflow_ import DeepExplain
    import tensorflow as tf
    ww = model.get_weights()
    with tf.Session() as sess:
        try:
            with DeepExplain(session=sess) as de:

                print(layer_name)
                model = keras.models.clone_model(model)
                model.set_weights(ww)

                x = model.get_layer(layer_name).output   #<tf.Tensor 'input_multi_1:0' shape=(?, 4000) dtype=float32>

                if type(output_index) == str:
                    y = model.get_layer(output_index).output
                else:
                    y = model.outputs[output_index]

                print(layer_name)
                print('model.inputs', model.inputs)
                print('model y', y)
                print('model x', x)
                attributions = de.explain(method_name, y, x, model.inputs[0], X)
                print('attributions', attributions.shape)
                scores = attributions
                return scores
        except:
            sess.close()
            print("Unexpected error:", sys.exc_info()[0])
            raise


