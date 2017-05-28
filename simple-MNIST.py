def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot = True, reshape = False)
    
epochs_1 = 20000

model1 = tf.Graph()
with model1.as_default():
    with tf.device('/gpu:0'):
        
        W_conv1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 32], stddev = 0.1), name = "W_0_1")
        W_conv2 = tf.Variable(tf.truncated_normal(shape = [5, 5, 32, 64], stddev = 0.1), name = "W_1_1") 
        W_fc1 = tf.Variable(tf.truncated_normal(shape = [7*7*64, 1024], stddev = 0.1), name = "W_2_1")
        W_fc2 = tf.Variable(tf.truncated_normal(shape = [1024, 10], stddev = 0.1), name = "W_3_1")

        b_conv1 = tf.Variable(tf.constant(0.1, shape = [32]), name = "b_0_1")
        b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_1_1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024]), name = "b_2_1")
        b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]), name = "b_3_1")



        x_1 = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        y_1 = tf.placeholder(tf.float32, shape = [None, 10])
        #x_image = tf.reshape(x_1, [-1,28,28,1])

        layer_1_1 = conv2d(x_1, W_conv1)
        layer_1_1 = tf.add(layer_1_1, b_conv1)
        layer_1_1 = tf.nn.relu(layer_1_1)
        layer_1_1 = max_pool_2x2(layer_1_1)

        layer_2_1 = conv2d(layer_1_1, W_conv2)
        layer_2_1 = tf.add(layer_2_1, b_conv2)
        layer_2_1 = tf.nn.relu(layer_2_1)
        layer_2_1 = max_pool_2x2(layer_2_1)

        layer_3_1 = tf.reshape(layer_2_1, [-1, 7*7*64])

        layer_4_1 = tf.add(tf.matmul(layer_3_1, W_fc1), b_fc1)
        layer_4_1 = tf.nn.relu(layer_4_1)

        keep_prob = tf.placeholder(tf.float32)
        layer_4_1 = tf.nn.dropout(layer_4_1, keep_prob)

        layer_5_1 = tf.add(tf.matmul(layer_4_1, W_fc2), b_fc2)

        cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_5_1, labels = y_1))
        optimizer_1 = tf.train.AdamOptimizer(1e-4).minimize(cost_1)

        correct_prediction_1 = tf.equal(tf.argmax(layer_5_1, 1), tf.argmax(y_1, 1))
        accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))
    
        init = tf.global_variables_initializer()
        
tf.reset_default_graph()

def save1():
    save_file_1 = './train_model1.ckpt'
    saver1 = tf.train.Saver({"W_0_1": W_conv1,
                            "W_1_1": W_conv2,
                            "W_2_1": W_fc1,
                            "W_3_1": W_fc2,
                            "b_0_1": b_conv1,
                            "b_1_1": b_conv2,
                            "b_2_1": b_fc1,
                            "b_3_1": b_fc2})
    return saver1, save_file_1

def train1(tfgraph, tfepochs, tfmnist, saver, save_file):
    start1 = timer()

    with tf.Session(graph = tfgraph) as sess:
        sess.run(init)

        for epoch in range(tfepochs):
            batch = tfmnist.train.next_batch(50)
            if epoch%100 == 0:
                train_accuracy = accuracy_1.eval(feed_dict = {x_1: batch[0], y_1: batch[1], keep_prob: 1.})
                print("step %d, training accuracy %g"%(epoch, train_accuracy))

            optimizer_1.run(feed_dict = {x_1: batch[0], y_1: batch[1], keep_prob: 0.5})

        print("test accuracy %g"%accuracy_1.eval(feed_dict = {x_1: tfmnist.test.images, y_1: tfmnist.test.labels, keep_prob: 1.}))

        saver.save(sess, save_file)
        print("")
        print("trained model saved")   


    end1 = timer()
    print("time: ", end1 - start1)

#train1(model1, epochs_1, mnist, save1()[0], save1()[1])
tf.reset_default_graph()

#MODEL 1 TESTING SET ACCURACY:
with tf.Session(graph = model1) as sess:
    save1()[0].restore(sess, save1()[1])
    feed_dict = {x_1: mnist.test.images, y_1: mnist.test.labels, keep_prob: 1.}
    file_writer = tf.summary.FileWriter('./logs/3, sess.graph)
    print("Test Accuracy: ", accuracy_1.eval(feed_dict)) #~99.2% accuracy
tf.reset_default_graph()
