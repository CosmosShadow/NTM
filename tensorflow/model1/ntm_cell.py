# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from utils import *
from ops import *

class NTMCell(object):
    def __init__(self, input_dim, output_dim,
                 mem_size=128, mem_dim=20, controller_dim=100,
                 controller_layer_size=1, shift_range=1,
                 write_head_size=1, read_head_size=1):
        """Initialize the parameters for an NTM cell.
        Args:
            input_dim: int, The number of units in the LSTM cell
            output_dim: int, The dimensionality of the inputs into the LSTM cell
            mem_size: (optional) int, The size of memory [128]
            mem_dim: (optional) int, The dimensionality for memory [20]
            controller_dim: (optional) int, The dimensionality for controller [100]
            controller_layer_size: (optional) int, The size of controller layer [1]
        """
        # initialize configs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range = shift_range
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size

        self.depth = 0      #states的深度(长度)
        self.states = []     #记录每前进一步的时的所有状态


    def __call__(self, input_, state=None, scope=None):
        """Run one step of NTM.
            input_: batch x input_dim
            new_output: batch x output_dim
        """
        if state == None:
            _, state = self.initial_state()

        M_prev = state['M']
        read_w_list_prev = state['read_w']     # 读地址
        write_w_list_prev = state['write_w']    # 写地址
        read_list_prev = state['read']              # 读到的内容
        output_list_prev = state['output']        # LSTM每层的输出
        hidden_list_prev = state['hidden']      # LSTM每层的隐藏单元

        output_list, hidden_list = self.build_controller(input_, read_list_prev, output_list_prev, hidden_list_prev)
        last_output = output_list[-1]
        M, read_w_list, write_w_list, read_list = self.build_memory(M_prev, read_w_list_prev, write_w_list_prev, last_output)
        new_output = self.new_output(last_output)       # LSTM的输出再非线性一下

        state = {
            'M': M,
            'read_w': read_w_list,
            'write_w': write_w_list,
            'read': read_list,
            'output': output_list,
            'hidden': hidden_list,
        }

        self.depth += 1
        self.states.append(state)

        return new_output, state


    def new_output(self, output):
        with tf.variable_scope('output'):
            return tf.sigmoid(Linear(output, self.output_dim, name='output'))


    def build_controller(self, input_, read_list_prev, output_list_prev, hidden_list_prev):
        with tf.variable_scope("controller"):
            output_list = []
            hidden_list = []
            for layer_idx in xrange(self.controller_layer_size):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]

                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev] + read_list_prev, output_size = self.controller_dim, bias = True, scope = "%s_gate_%s" % (gate_name, layer_idx))
                else:
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev], output_size = self.controller_dim, bias = True, scope="%s_gate_%s" % (gate_name, layer_idx))

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hid = tf.add_n([f * h_prev, i * update])
                out = o * tf.tanh(hid)

                hidden_list.append(hid)
                output_list.append(out)

            return output_list, hidden_list


    def build_memory(self, M_prev, read_w_list_prev, write_w_list_prev, last_output):
        with tf.variable_scope("memory"):
            # 3.1 Reading
            if self.read_head_size == 1:
                read_w_prev = read_w_list_prev[0]
                read_w, read = self.build_read_head(M_prev, tf.squeeze(read_w_prev), last_output, 0)
                read_w_list = [read_w]
                read_list = [read]
            else:
                read_w_list = []
                read_list = []
                for idx in xrange(self.read_head_size):
                    read_w_prev_idx = read_w_list_prev[idx]
                    read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx, last_output, idx)      #更新读地址、读到的内容
                    read_w_list.append(read_w_idx)
                    read_list.append(read_idx)

            # 3.2 Writing
            if self.write_head_size == 1:
                write_w_prev = write_w_list_prev[0]
                write_w, write, erase = self.build_write_head(M_prev, tf.squeeze(write_w_prev), last_output, 0)
                M_erase = tf.ones([self.mem_size, self.mem_dim]) - outer_product(write_w, erase)
                M_write = outer_product(write_w, write)
                write_w_list = [write_w]
            else:
                write_w_list = []
                write_list = []
                erase_list = []
                M_erases = []
                M_writes = []

                for idx in xrange(self.write_head_size):
                    write_w_prev_idx = write_w_list_prev[idx]
                    write_w_idx, write_idx, erase_idx = self.build_write_head(M_prev, write_w_prev_idx, last_output, idx)    #更新写的址址、写内容、檫除内容
                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)
                    M_erases.append(tf.ones([self.mem_size, self.mem_dim]) - outer_product(write_w_idx, erase_idx))
                    M_writes.append(outer_product(write_w_idx, write_idx))

                M_erase = reduce(lambda x, y: x*y, M_erases)
                M_write = tf.add_n(M_writes)

            M = M_prev * M_erase + M_write

            return M, read_w_list, write_w_list, read_list


    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)


    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)


    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        scope = "read" if is_read else "write"
        with tf.variable_scope(scope):
            # 生成相关参数
            with tf.variable_scope("k"):    # Amplify or attenuate the precision
                k = tf.tanh(Linear(last_output, self.mem_dim, name='k_%s' % idx))
            with tf.variable_scope("g"):    # Interpolation gate
                g = tf.sigmoid(Linear(last_output, 1, name='g_%s' % idx))
            with tf.variable_scope("s_w"):   # shift weighting
                w = Linear(last_output, 2 * self.shift_range + 1, name='s_w_%s' % idx)
                s_w = softmax(w)
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx))
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)), tf.constant(1.0))

            # 基于内容地址
            similarity = smooth_cosine_similarity(M_prev, k)
            content_focused_w = softmax(scalar_mul(similarity, beta))

            # 融合以前的地址
            gated_w = tf.add_n([scalar_mul(content_focused_w, g), scalar_mul(w_prev, (tf.constant(1.0) - g))])

            # 卷积移动
            conv_w = circular_convolution(gated_w, s_w)

            # 锐化
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w)

            if is_read:
                read = matmul(tf.transpose(M_prev), w)
                return w, read
            else:
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx))
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return w, add, erase


    def initial_state(self, dummy_value=0.0):
        self.depth = 0
        self.states = []
        with tf.variable_scope("init_cell"):
            # always zero
            dummy = tf.Variable(tf.constant([[dummy_value]], dtype=tf.float32))

            # memory
            M_init_linear = tf.tanh(Linear(dummy, self.mem_size * self.mem_dim, name='M_init_linear'))
            M_init = tf.reshape(M_init_linear, [self.mem_size, self.mem_dim])

            # read weights
            read_w_list_init = []
            read_list_init = []
            for idx in xrange(self.read_head_size):
                read_w_idx = Linear(dummy, self.mem_size, is_range=True, squeeze=True, name='read_w_%d' % idx)
                read_w_list_init.append(softmax(read_w_idx))
                read_init_idx = Linear(dummy, self.mem_dim, squeeze=True, name='read_init_%d' % idx)
                read_list_init.append(tf.tanh(read_init_idx))

            # write weights
            write_w_list_init = []
            for idx in xrange(self.write_head_size):
                write_w_idx = Linear(dummy, self.mem_size, is_range=True, squeeze=True, name='write_w_%s' % idx)
                write_w_list_init.append(softmax(write_w_idx))

            # controller state
            output_init_list = []                     
            hidden_init_list = []                     
            for idx in xrange(self.controller_layer_size):
                output_init_idx = Linear(dummy, self.controller_dim, squeeze=True, name='output_init_%s' % idx)
                output_init_list.append(tf.tanh(output_init_idx))
                hidden_init_idx = Linear(dummy, self.controller_dim, squeeze=True, name='hidden_init_%s' % idx)
                hidden_init_list.append(tf.tanh(hidden_init_idx))

            output = tf.tanh(Linear(dummy, self.output_dim, name='new_output'))

            state = {
                'M': M_init,
                'read_w': read_w_list_init,
                'write_w': write_w_list_init,
                'read': read_list_init,
                'output': output_init_list,
                'hidden': hidden_init_list
            }

            self.depth += 1
            self.states.append(state)

            return output, state


    def get_memory(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['M']


    def get_read_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read_w']


    def get_write_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['write_w']


    def get_read_vector(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read']


    def print_read_max(self, sess):
        read_w_list = sess.run(self.get_read_weights())

        fmt = "%-4d %.4f"
        if self.read_head_size == 1:
            print(fmt % (argmax(read_w_list[0])))
        else:
            for idx in xrange(self.read_head_size):
                print(fmt % np.argmax(read_w_list[idx]))


    def print_write_max(self, sess):
        write_w_list = sess.run(self.get_write_weights())

        fmt = "%-4d %.4f"
        if self.write_head_size == 1:
            print(fmt % (argmax(write_w_list[0])))
        else:
            for idx in xrange(self.write_head_size):
                print(fmt % argmax(write_w_list[idx]))


















