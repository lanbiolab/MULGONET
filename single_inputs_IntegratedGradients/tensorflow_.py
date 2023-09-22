from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings
from collections import OrderedDict

import numpy as np
import tensorflow as tf
#from skimage.util import view_as_windows
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad

SUPPORTED_ACTIVATIONS = ['Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus']

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0




def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    if op.type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)




class AttributionMethod(object):
    """
    Attribution method base class
    """

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase=None):
        self.T = T
        self.inputs = inputs
        self.X = X
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple


    def session_run(self, T, xs):
        feed_dict = {}
        if self.has_multiple_inputs:
            print('has_multiple_inputs')
            # if len(xs) != len(self.X):
            if len(xs) != len(self.inputs):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   # % (str(len(xs)), str(len(self.X))))
                                   % (str(len(xs)), str(len(self.inputs))))
            # for k, v in zip(self.X, xs):
            for k, v in zip(self.inputs, xs):
                feed_dict[k] = np.float32(v)
        else:
            # feed_dict[self.X] = xs
            feed_dict[self.inputs] = xs   #输入的是参照xs ，开始计算第一层参照的时候，后面进来是输入数据

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0

        for key, value in feed_dict.items():
            if type(value) == np.ndarray:
                print(key, type(value), value.shape, value.dtype)
        return self.session.run(T, feed_dict)

    def _set_check_baseline(self):
        xss = self.xs  #输入的数据
        # xss= self.session_run(self.X, self.xs)
        print('xss {}, xs {}'.format(xss.shape, self.xs.shape))
        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros((1,) + xi.shape[1:]) for xi in xss]
            else:
                self.baseline = np.zeros((1,) + xss.shape[1:])
        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.xs):
                    if self.baseline[i].shape == xss[i].shape[1:]:
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                           % (self.baseline[i].shape, self.xs[i].shape[1:]))
            else:
                if self.baseline.shape == xss.shape[1:]:
                    self.baseline = np.expand_dims(self.baseline, 0)
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                       % (self.baseline.shape, self.xs.shape[1:]))

class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """

    def get_symbolic_attribution(self):
        print('hello from symbolic attribution')
        # gradients= K.gradients(self.T, self.X)
        # grad = K.function(inputs=self.inputs, outputs=gradients)
        gradients = [g for g in tf.gradients(self.T, self.X)]
        return gradients

    def run(self):
        attributions = self.get_symbolic_attribution()
        results = self.session_run(attributions, self.xs)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class GradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        print('hello from GradientXInput')
        # gradients =  [self.X*g for g in K.gradients(self.T, self.X)]
        gradients = [self.X * g for g in tf.gradients(self.T, self.X)]
        # gradients =   K.gradients(self.T, self.X)

        # gradients = tf.gradients(self.T, self.X)
        print(self.T, self.X, gradients)


        return gradients

class IntegratedGradients(GradientBasedMethod):

    def __init__(self, T, X, input, xs, session, keras_learning_phase, steps=10, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, input, xs, session, keras_learning_phase)
        self.steps = steps
        self.baseline = baseline

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        attributions = self.get_symbolic_attribution()
        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [b + (xs - b) * alpha for xs, b in zip(self.xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (self.xs - self.baseline) * alpha
            _attr = self.session_run(attributions, xs_mod)
            # print ('attributions',attributions)
            _attr = self.session_run(attributions, self.xs)
            xss = self.session_run(self.X, self.xs)
            if gradient is None:
                gradient = _attr
            else:
                gradient = [g + a for g, a in zip(gradient, _attr)]

        # layer_baseline = self.baseline
        layer_baseline = self.session_run(self.X, self.baseline)
        # xss = self.xs
        xss = self.session_run(self.X, self.xs)
        if self.has_multiple_inputs:
            results = [g * (x - b) / self.steps for g, x, b in zip(gradient, xss, layer_baseline)]
        else:
            print('self.xs {}, self.baseline  {}, gradient {} {}'.format(xss.shape, layer_baseline.shape, len(gradient),
                                                                         gradient[0].shape))
            results = [g * (x - b) / self.steps for g, x, b in zip(gradient, [xss], [layer_baseline])]

        return results[0] if not self.has_multiple_inputs else results


attribution_methods = OrderedDict({

    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),

})


@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)


class DeepExplain(object):

    def __init__(self, graph=None, session=tf.get_default_session()):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        print('graph', self.graph)
        # op = session.graph.get_operations()
        # for m in op:
        #     print (m.values())

        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def explain(self, method, T, X, inputs, xs, **kwargs):
        print('deep explain')
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        self.method = 'intgrad'

        method_class, method_flag = attribution_methods[self.method]

        print('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X, inputs, xs, self.session, self.keras_phase_placeholder, **kwargs)
        result = method.run()
        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return result

    @staticmethod
    def get_override_map():
        return dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.
        :return:
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]
