#a second neural network which determines if a digit is centered on a 28x28 image or not
#data for training and testing this network can be found in the data.zip folder

print(is_it_a_digit_load.shape)
print(is_it_a_digit_label_load.shape)
#digit_shift = is_it_a_digit_load / 255.
digit_shift = is_it_a_digit_load
digit_shift_label = is_it_a_digit_label_load

#split the data
shift_X_train, shift_X_test, shift_y_train, shift_y_test = train_test_split(digit_shift, digit_shift_label, test_size = 0.25, random_state = 7)
shift_X_test, shift_X_val, shift_y_test, shift_y_val = train_test_split(shift_X_test, shift_y_test, test_size = 0.5, random_state = 7)

print(shift_X_train.shape)
print(shift_X_test.shape)
print(shift_X_val.shape)

shift_X_train = shift_X_train.reshape(6750, 28,28, 1)
shift_X_test = shift_X_test.reshape(1125, 28,28, 1)
shift_X_val = shift_X_val.reshape(1125, 28,28, 1)

#network for digit/no digit
learning_rate = 0.15
epochs = 10
n_input = 784
n_classes = 2
n_hidden_layer = 200
num_examples = 6750
batch_size = 32

tf.reset_default_graph()

from time import perf_counter as timer
start1 = timer()


model2 = tf.Graph()
with model2.as_default():
    with tf.device('/gpu:0'):
        W_fc1_2 = tf.Variable(tf.random_normal([n_input, n_hidden_layer]), name = "W_0_2")
        W_fc2_2 = tf.Variable(tf.random_normal([n_hidden_layer, n_classes]), name = "W_1_2")

        b_fc1_2 = tf.Variable(tf.zeros([n_hidden_layer]), name = "b_0_2")
        b_fc2_2 = tf.Variable(tf.zeros([n_classes]), name = "b_1_2")

        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 2])

        x_flat = tf.reshape(x, [-1, n_input])

        #fully connected relu layer
        layer_1 = tf.add(tf.matmul(x_flat, W_fc1_2), b_fc1_2)
        layer_1 = tf.nn.relu(layer_1)

        #fully connected softmax output layer
        layer_2 = tf.add(tf.matmul(layer_1, W_fc2_2), b_fc2_2)
        prediction = tf.nn.softmax(layer_2)

        #loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_2, labels = y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

        #accuracy
        correct_prediction = tf.equal(tf.argmax(layer_2, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        
tf.reset_default_graph()

def save2():
    save_file_2 = './train_model2.ckpt'
    saver2 = tf.train.Saver({"W_0_2": W_fc1_2,
                            "W_1_2": W_fc2_2,
                            "b_0_2": b_fc1_2,
                            "b_1_2": b_fc2_2})
    return saver2, save_file_2

def train2(tfgraph, tfepochs, tfbatch, xtrain, ytrain, xtest, ytest, xval, yval, num_examples, saver, save_file):
    start1 = timer()
    

    with tf.Session(graph = tfgraph) as sess2:
        sess2.run(init)

        for epoch in range(tfepochs):

            shift_X_train, shift_y_train = shuffle(xtrain, ytrain)

            for offset in range(0, num_examples, tfbatch):
                end = offset + tfbatch
                batch_x, batch_y = shift_X_train[offset:end], shift_y_train[offset:end]
                sess2.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

            valid_accuracy = sess2.run(accuracy, feed_dict = {x:xval,
                                                            y:yval})
            print("Epoch: ", epoch)
            print("Validation Accuracy: ", valid_accuracy)

        saver.save(sess2, save_file)
        print("")
        print("trained model 2 saved")

        #calculate test accuracy
        test_accuracy = sess2.run(accuracy, feed_dict = {x:xtest, y:ytest})



        print("")
        print("testing accuracy: {}".format(test_accuracy))

    end1 = timer()
    print(end1 - start1)


#train2(model2, epochs, batch_size, shift_X_train, shift_y_train, shift_X_test, shift_y_test, shift_X_val, shift_y_val, num_examples, save2()[0], save2()[1])
tf.reset_default_graph()

#MODEL 2 TESTING SET ACCURACY:
with tf.Session(graph = model2) as sess:
    save2()[0].restore(sess, save2()[1])
    feed_dict = {x: shift_X_test, y: shift_y_test}
    print("Test Accuracy: ", accuracy.eval(feed_dict = feed_dict)) #gets around 98% accuracy
tf.reset_default_graph() 
