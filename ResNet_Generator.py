#!/usr/bin/env python
"""
Generate the residule learning network.
Author: Yemin Shi, Modified by Li He

MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('solver_file',
                        help='Output solver.prototxt file')
    parser.add_argument('train_val_file',
                        help='Output train_val.prototxt file')
    parser.add_argument('--layer_number', type=int,nargs='*',
                        help=('Layer number for each layer stage.'),
                        default=[3, 8, 36, 3])
    parser.add_argument('-t', '--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer():
    data_layer_str = '''name: "ResNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    crop_size: 224
  }
}'''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    bias_term: false
    weight_filler {
      type: "%s"
    }
  }
}'''%(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "acc/top-1"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "acc/top-5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}'''%(bottom, bottom, bottom)
    return softmax_loss_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  batch_norm_param {
    use_global_stats: true
  }
}'''%(layer_name, bottom, top)
    return bn_layer_str

def generate_scale_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "Scale"
  bottom: "%s"
  top: "%s"
  scale_param {
    bias_term: true
  }
}'''%(layer_name, bottom, top)
    return bn_layer_str
    
def generate_train_val():
    args = parse_args()
    network_str = generate_data_layer()
    '''stage1'''
    last_top = 'data'
    network_str += generate_conv_layer(7, 64, 2, 3, 'conv1', last_top, 'conv1')
    network_str += generate_bn_layer('conv1_bn', 'conv1', 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    network_str += generate_scale_layer('conv1_scale', 'conv1', 'conv1')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1', 'pool1')
    '''stage 2'''
    last_top = 'pool1'
    network_str += generate_conv_layer(1, 256, 1, 0, 'conv2_1r', last_top, 'conv2_1r')
    network_str += generate_bn_layer('conv2_1r_bn', 'conv2_1r', 'conv2_1r')
    network_str += generate_scale_layer('conv2_1r_scale', 'conv2_1r', 'conv2_1r')
    network_str += generate_conv_layer(1, 64, 1, 0, 'conv2_1a', last_top, 'conv2_1a')
    network_str += generate_bn_layer('conv2_1a_bn', 'conv2_1a', 'conv2_1a')
    network_str += generate_activation_layer('conv2_1a_relu', 'conv2_1a', 'conv2_1a', 'ReLU')
    network_str += generate_scale_layer('conv2_1a_scale', 'conv2_1a', 'conv2_1a')
    network_str += generate_conv_layer(3, 64, 1, 1, 'conv2_1b', 'conv2_1a', 'conv2_1b')
    network_str += generate_bn_layer('conv2_1b_bn', 'conv2_1b', 'conv2_1b')
    network_str += generate_activation_layer('conv2_1b_relu', 'conv2_1b', 'conv2_1b', 'ReLU')
    network_str += generate_scale_layer('conv2_1b_scale', 'conv2_1b', 'conv2_1b')
    network_str += generate_conv_layer(1, 256, 1, 0, 'conv2_1c', 'conv2_1b', 'conv2_1c')
    network_str += generate_bn_layer('conv2_1c_bn', 'conv2_1c', 'conv2_1c')
    network_str += generate_scale_layer('conv2_1c_scale', 'conv2_1c', 'conv2_1c')
    network_str += generate_eltwise_layer('conv2_1', 'conv2_1r', 'conv2_1c', 'conv2_1')
    network_str += generate_activation_layer('conv2_1_relu', 'conv2_1', 'conv2_1', 'ReLU')
    last_top = 'conv2_1'
    last_output = 'conv2_1'
    for l in xrange(2, args.layer_number[0]+1):
        network_str += generate_conv_layer(1, 64, 1, 0, 'conv2_%da'%l, last_top, 'conv2_%da'%l)
        network_str += generate_bn_layer('conv2_%da_bn'%l, 'conv2_%da'%l, 'conv2_%da'%l)
        network_str += generate_activation_layer('conv2_%da_relu'%l, 'conv2_%da'%l, 'conv2_%da'%l, 'ReLU')
        network_str += generate_scale_layer('conv2_%da_scale'%l, 'conv2_%da'%l, 'conv2_%da'%l)
        network_str += generate_conv_layer(3, 64, 1, 1, 'conv2_%db'%l, 'conv2_%da'%l, 'conv2_%db'%l)
        network_str += generate_bn_layer('conv2_%db_bn'%l, 'conv2_%db'%l, 'conv2_%db'%l)
        network_str += generate_activation_layer('conv2_%db_relu'%l, 'conv2_%db'%l, 'conv2_%db'%l, 'ReLU')
        network_str += generate_scale_layer('conv2_%db_scale'%l, 'conv2_%db'%l, 'conv2_%db'%l)
        network_str += generate_conv_layer(1, 256, 1, 0, 'conv2_%dc'%l, 'conv2_%db'%l, 'conv2_%dc'%l)
        network_str += generate_bn_layer('conv2_%dc_bn'%l, 'conv2_%dc'%l, 'conv2_%dc'%l)
        network_str += generate_scale_layer('conv2_%dc_scale'%l, 'conv2_%dc'%l, 'conv2_%dc'%l)
        network_str += generate_eltwise_layer('conv2_%d'%l, last_top, 'conv2_%dc'%l, 'conv2_%d'%l)
        network_str += generate_activation_layer('conv2_%d_relu'%l, 'conv2_%d'%l, 'conv2_%d'%l, 'ReLU')
        last_top = 'conv2_%d'%l
    '''stage 3'''
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv3_1r', last_top, 'conv3_1r')
    network_str += generate_bn_layer('conv3_1r_bn', 'conv3_1r', 'conv3_1r')
    network_str += generate_scale_layer('conv3_1r_scale', 'conv3_1r', 'conv3_1r')
    network_str += generate_conv_layer(1, 128, 2, 0, 'conv3_1a', last_top, 'conv3_1a')
    network_str += generate_bn_layer('conv3_1a_bn', 'conv3_1a', 'conv3_1a')
    network_str += generate_activation_layer('conv3_1a_relu', 'conv3_1a', 'conv3_1a', 'ReLU')
    network_str += generate_scale_layer('conv3_1a_scale', 'conv3_1a', 'conv3_1a')
    network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_1b', 'conv3_1a', 'conv3_1b')
    network_str += generate_bn_layer('conv3_1b_bn', 'conv3_1b', 'conv3_1b')
    network_str += generate_activation_layer('conv3_1b_relu', 'conv3_1b', 'conv3_1b', 'ReLU')
    network_str += generate_scale_layer('conv3_1b_scale', 'conv3_1b', 'conv3_1b')
    network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_1c', 'conv3_1b', 'conv3_1c')
    network_str += generate_bn_layer('conv3_1c_bn', 'conv3_1c', 'conv3_1c')
    network_str += generate_scale_layer('conv3_1c_scale', 'conv3_1c', 'conv3_1c')
    network_str += generate_eltwise_layer('conv3_1', 'conv3_1r', 'conv3_1c', 'conv3_1')
    network_str += generate_activation_layer('conv3_1_relu', 'conv3_1', 'conv3_1', 'ReLU')
    last_top = 'conv3_1'
    for l in xrange(2, args.layer_number[1]+1):
        network_str += generate_conv_layer(1, 128, 1, 0, 'conv3_%da'%l, last_top, 'conv3_%da'%l)
        network_str += generate_bn_layer('conv3_%da_bn'%l, 'conv3_%da'%l, 'conv3_%da'%l)
        network_str += generate_activation_layer('conv3_%da_relu'%l, 'conv3_%da'%l, 'conv3_%da'%l, 'ReLU')
        network_str += generate_scale_layer('conv3_%da_scale'%l, 'conv3_%da'%l, 'conv3_%da'%l)
        network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_%db'%l, 'conv3_%da'%l, 'conv3_%db'%l)
        network_str += generate_bn_layer('conv3_%db_bn'%l, 'conv3_%db'%l, 'conv3_%db'%l)
        network_str += generate_activation_layer('conv3_%db_relu'%l, 'conv3_%db'%l, 'conv3_%db'%l, 'ReLU')
        network_str += generate_scale_layer('conv3_%db_scale'%l, 'conv3_%db'%l, 'conv3_%db'%l)
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_%dc'%l, 'conv3_%db'%l, 'conv3_%dc'%l)
        network_str += generate_bn_layer('conv3_%dc_bn'%l, 'conv3_%dc'%l, 'conv3_%dc'%l)
        network_str += generate_scale_layer('conv3_%dc_scale'%l, 'conv3_%dc'%l, 'conv3_%dc'%l)
        network_str += generate_eltwise_layer('conv3_%d'%l, last_top, 'conv3_%dc'%l, 'conv3_%d'%l)
        network_str += generate_activation_layer('conv3_%d_relu'%l, 'conv3_%d'%l, 'conv3_%d'%l, 'ReLU')
        last_top = 'conv3_%d'%l
    '''stage 4'''
    network_str += generate_conv_layer(1, 1024, 2, 0, 'conv4_1r', last_top, 'conv4_1r')
    network_str += generate_bn_layer('conv4_1r_bn', 'conv4_1r', 'conv4_1r')
    network_str += generate_scale_layer('conv4_1r_scale', 'conv4_1r', 'conv4_1r')
    network_str += generate_conv_layer(1, 256, 2, 0, 'conv4_1a', last_top, 'conv4_1a')
    network_str += generate_bn_layer('conv4_1a_bn', 'conv4_1a', 'conv4_1a')
    network_str += generate_activation_layer('conv4_1a_relu', 'conv4_1a', 'conv4_1a', 'ReLU')
    network_str += generate_scale_layer('conv4_1a_scale', 'conv4_1a', 'conv4_1a')
    network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_1b', 'conv4_1a', 'conv4_1b')
    network_str += generate_bn_layer('conv4_1b_bn', 'conv4_1b', 'conv4_1b')
    network_str += generate_activation_layer('conv4_1b_relu', 'conv4_1b', 'conv4_1b', 'ReLU')
    network_str += generate_scale_layer('conv4_1b_scale', 'conv4_1b', 'conv4_1b')
    network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_1c', 'conv4_1b', 'conv4_1c')
    network_str += generate_bn_layer('conv4_1c_bn', 'conv4_1c', 'conv4_1c')
    network_str += generate_scale_layer('conv4_1c_scale', 'conv4_1c', 'conv4_1c')
    network_str += generate_eltwise_layer('conv4_1', 'conv4_1r', 'conv4_1c', 'conv4_1')
    network_str += generate_activation_layer('conv4_1_relu', 'conv4_1', 'conv4_1', 'ReLU')
    last_top = 'conv4_1'
    for l in xrange(2, args.layer_number[2]+1):
        network_str += generate_conv_layer(1, 256, 1, 0, 'conv4_%da'%l, last_top, 'conv4_%da'%l)
        network_str += generate_bn_layer('conv4_%da_bn'%l, 'conv4_%da'%l, 'conv4_%da'%l)
        network_str += generate_activation_layer('conv4_%da_relu'%l, 'conv4_%da'%l, 'conv4_%da'%l, 'ReLU')
	network_str += generate_scale_layer('conv4_%da_scale'%l, 'conv4_%da'%l, 'conv4_%da'%l)
        network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_%db'%l, 'conv4_%da'%l, 'conv4_%db'%l)
        network_str += generate_bn_layer('conv4_%db_bn'%l, 'conv4_%db'%l, 'conv4_%db'%l)
        network_str += generate_activation_layer('conv4_%db_relu'%l, 'conv4_%db'%l, 'conv4_%db'%l, 'ReLU')
	network_str += generate_scale_layer('conv4_%db_scale'%l, 'conv4_%db'%l, 'conv4_%db'%l)
        network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_%dc'%l, 'conv4_%db'%l, 'conv4_%dc'%l)
        network_str += generate_bn_layer('conv4_%dc_bn'%l, 'conv4_%dc'%l, 'conv4_%dc'%l)
	network_str += generate_scale_layer('conv4_%dc_scale'%l, 'conv4_%dc'%l, 'conv4_%dc'%l)
        network_str += generate_eltwise_layer('conv4_%d'%l, last_top, 'conv4_%dc'%l, 'conv4_%d'%l)
        network_str += generate_activation_layer('conv4_%d_relu'%l, 'conv4_%d'%l, 'conv4_%d'%l, 'ReLU')
        last_top = 'conv4_%d'%l
    '''stage 5'''
    network_str += generate_conv_layer(1, 2048, 2, 0, 'conv5_1r', last_top, 'conv5_1r')
    network_str += generate_bn_layer('conv5_1r_bn', 'conv5_1r', 'conv5_1r')
    network_str += generate_activation_layer('conv5_1r_relu', 'conv5_1r', 'conv5_1r', 'ReLU')
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv5_1a', last_top, 'conv5_1a')
    network_str += generate_bn_layer('conv5_1a_bn', 'conv5_1a', 'conv5_1a')
    network_str += generate_activation_layer('conv5_1a_relu', 'conv5_1a', 'conv5_1a', 'ReLU')
    network_str += generate_scale_layer('conv5_1a_scale', 'conv5_1a', 'conv5_1a')
    network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_1b', 'conv5_1a', 'conv5_1b')
    network_str += generate_bn_layer('conv5_1b_bn', 'conv5_1b', 'conv5_1b')
    network_str += generate_activation_layer('conv5_1b_relu', 'conv5_1b', 'conv5_1b', 'ReLU')
    network_str += generate_scale_layer('conv5_1b_scale', 'conv5_1b', 'conv5_1b')
    network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_1c', 'conv5_1b', 'conv5_1c')
    network_str += generate_bn_layer('conv5_1c_bn', 'conv5_1c', 'conv5_1c')
    network_str += generate_scale_layer('conv5_1c_scale', 'conv5_1c', 'conv5_1c')
    network_str += generate_eltwise_layer('conv5_1', 'conv5_1r', 'conv5_1c', 'conv5_1')
    network_str += generate_activation_layer('conv5_1_relu', 'conv5_1', 'conv5_1', 'ReLU')
    last_top = 'conv5_1'
    for l in xrange(2, args.layer_number[3]+1):
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv5_%da'%l, last_top, 'conv5_%da'%l)
        network_str += generate_bn_layer('conv5_%da_bn'%l, 'conv5_%da'%l, 'conv5_%da'%l)
	network_str += generate_scale_layer('conv5_%da_scale'%l, 'conv5_%da'%l, 'conv5_%da'%l)
        network_str += generate_activation_layer('conv5_%da_relu'%l, 'conv5_%da'%l, 'conv5_%da'%l, 'ReLU')
        network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_%db'%l, 'conv5_%da'%l, 'conv5_%db'%l)
        network_str += generate_bn_layer('conv5_%db_bn'%l, 'conv5_%db'%l, 'conv5_%db'%l)
        network_str += generate_activation_layer('conv5_%db_relu'%l, 'conv5_%db'%l, 'conv5_%db'%l, 'ReLU')
	network_str += generate_scale_layer('conv5_%db_scale'%l, 'conv5_%db'%l, 'conv5_%db'%l)
        network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_%dc'%l, 'conv5_%db'%l, 'conv5_%dc'%l)
        network_str += generate_bn_layer('conv5_%dc_bn'%l, 'conv5_%dc'%l, 'conv5_%dc'%l)
	network_str += generate_scale_layer('conv5_%dc_scale'%l, 'conv5_%dc'%l, 'conv5_%dc'%l)
        network_str += generate_eltwise_layer('conv5_%d'%l, last_top, 'conv5_%dc'%l, 'conv5_%d'%l)
        network_str += generate_activation_layer('conv5_%d_relu'%l, 'conv5_%d'%l, 'conv5_%d'%l, 'ReLU')
        last_top = 'conv5_%d'%l
    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool2', last_top, 'pool2')
    network_str += generate_fc_layer(1000, 'fc', 'pool2', 'fc', 'gaussian')
    network_str += generate_softmax_loss('fc')
    return network_str

def generate_solver(train_val_name):
    solver_str = '''net: "%s"
test_iter: 1000
test_interval: 6000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
stepvalue: 300000
stepvalue: 500000
gamma: 0.1
max_iter: 600000
momentum: 0.9
weight_decay: 0.0001
snapshot: 6000
snapshot_prefix: "pku_resnet"
solver_mode: GPU
device_id: [0,1,6,8]'''%(train_val_name)
    return solver_str

def main():
    args = parse_args()
    solver_str = generate_solver(args.train_val_file)
    network_str = generate_train_val()
    fp = open(args.solver_file, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(args.train_val_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
