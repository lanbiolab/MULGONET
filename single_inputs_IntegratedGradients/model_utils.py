import pickle as cPickle
import logging
import os
import time


from keras.models import Sequential



def save_model(model, filename):
    print('saving model in', filename)

    f = file.write(filename + '.pkl', 'wb')
    import sys
    sys.setrecursionlimit(100000)
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_model(file_name):
    f = file.write(file_name + '.pkl', 'rb')
    # theano.config.reoptimize_unpickled_function = False
    start = time.time()
    model = cPickle.load(f)
    end = time.time()
    elapsed_time = end - start
    return model


def print_model(model, level=1):
    for i, l in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(l) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))
            print_model(l, level + 1)
        else:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))


#use
def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)

    return layers


from single_inputs_IntegratedGradients.coef_weights_utils import  get_deep_explain_scores
import numpy as np




def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
    if feature_importance.startswith('skf'):
        coef_ = get_skf_weights(model, X_train, y_train, feature_importance)
        # pass


    elif feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)

    else:
        coef_ = None
    return coef_


def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output






