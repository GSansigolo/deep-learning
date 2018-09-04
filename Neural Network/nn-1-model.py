'''
Created on Sep 03, 2018
Author: @G_Sansigolo
'''
import tensorflow as tf

mnist = tf.keras.datasets.mnist

n_node_hl1 = 500
n_node_hl2 = 500
n_node_hl3 = 500

n_classes = 10
batch_size = 100

# height x weidth
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_norm([784, n_node_hl1])), 
                      'biases':  tf.Variable(tf.random_norm(n_node_hl1))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_norm([n_node_hl1, n_node_hl2])), 
                      'biases':  tf.Variable(tf.random_norm(n_node_hl2))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_norm([n_node_hl2, n_node_hl3])), 
                      'biases':  tf.Variable(tf.random_norm(n_node_hl3))}

    output_layer = {'weights': tf.Variable(tf.random_norm([n_node_hl3, n_classes])), 
                      'biases':  tf.Variable(tf.random_norm(n_classes))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = (tf.matmul(l3, output_layer['weights']) + output_layer['biases'])

    return output
