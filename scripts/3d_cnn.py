x_size = 120
y_size = 160
block = 20
n_classes = 2
batch_size = 2
dropout = 0.8
train_test_split = 0.5

labels_path = '../input/labels.txt'
data_path = '../input/rotation_resized/'
save_path = '../input/rotation_resized/'

import pandas as pd
import numpy as np
import os, glob, random, math
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def blocks(lis, n):
    for i in range(0, len(lis), n):
        yield lis[i:i + n]

def mean(lis):
    return sum(lis) / len(lis)

def preprocess_data(vid, labels, y_img_size, x_img_size, block=20, visualize=False):
    patient = vid[:4]
    label_df = clean_labels.loc[labels['Patient'] == int(patient)]
    label = label_df.iloc[0]['class']
    array = np.load('../input/rotation/' + vid)
    resized_array = [cv2.resize(frame, (y_img_size, x_img_size))
                     for frame in array]
    block_size = math.ceil(len(resized_array) / block)
    new_array = []
    for block_slice in blocks(resized_array, block_size):
        block_slice = list(map(mean, zip(*block_slice)))
        new_array.append(block_slice)
    if visualize:
        fig = plt.figure()
        for n, frame in enumerate(new_array):
            y = fig.add_subplot(4, 5, n + 1)
            y.imshow(frame, cmap='gray')
        plt.show()
    if label == 0: label = np.array([1,0])
    elif label == 4: label = np.array([1,0])
    return np.array(new_array), label

def name_format(name):
    if '-' in name:
        return name[:-4]
    else:
        return name[:4] + '-' + name[4:-4]


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def max_pool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1],
                            strides=[1,2,2,2,1], padding='SAME')
def make_cnn(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3,3,3,1,32])),
               'W_conv2': tf.Variable(tf.random_normal([3,3,3,32,64])),
               'W_fc': tf.Variable(tf.random_normal([386048, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.reshape(x, shape=[-1, x_size, y_size, block, 1])
    
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = max_pool3d(conv1)
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = max_pool3d(conv2)
    
    fc = tf.reshape(conv2, [-1, 386048])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, dropout)
    
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_cnn(x):
    predictions = make_cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        successful_runs = 0
        total_runs = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for data in train:
                total_runs += 1
                try:
                    x = data[0]
                    y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x:x, y:y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    print(str(e))
            print('Epoch', epoch+1, 'completed out of ', epochs, 'loss: ', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval({x:[i[0] for i in test],
                                               y:[i[1] for i in test]}))
            print('Done.')
            print('Accuracy: ', accuracy.eval({x:[i[0] for i in test],
                                               y:[i[1] for i in test]}))
            print('Number of successful runs: ', successful_runs/total_runs)
            

# canonical = clean_labels[(clean_labels['Abnormality'] == 4)
#                           | (clean_labels['Abnormality'] == 1)]

# canonical.groupby('class').count()


videos = os.listdir('../input/rotation')
labels = pd.read_table(labels_path, sep=' ',
                       names=['Patient', 'Abnormality', 'class'])

clean_labels = labels.drop_duplicates()

much_data = []
for n, vid in enumerate(videos):
    if n % 100 == 0:
        print(n)
    try:
        img_data, label = preprocess_data(vid, clean_labels, 
                                          y_img_size=y_size,
                                          x_img_size=x_size,
                                          block=block)
        much_data.append([img_data, label])
    except KeyError as e:
        print(str(e))
    file_path = save_path + '{}_{}_{}_{}.npy'.format(name_format(vid),
                                             y_size, 
                                             x_size,
                                             block)
    if not os.path.exists(file_path):
        np.save(file_path, much_data[n])


x = tf.placeholder('float')
y = tf.placeholder('float')

random.seed(2017)
random.shuffle(rota_data)
split_point = int(round(train_test_split * len(rota_data)))
train = rota_data[:split_point]
test = rota_data[split_point:]

train_cnn(x)



