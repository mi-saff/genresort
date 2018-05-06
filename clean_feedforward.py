import tensorflow as tf
import time
import os
import math
import numpy as np
import generate_data_2
import embedding
from model import Model

class Config(object):
    threshold = 15
    embed_size = 50
    batch_size = 500
    label_size = 5
    max_epochs = 100
    early_stopping = 5

    def __init__(self, hl, lr_new):
        self.hidden_size = hl
        self.lr = lr_new
class ForwardModel(Model):
    def load_data(self):
        self.X_train = []
        self.y_train = []

        embedding_dict = embedding.word_to_embedding()

        self.X_train, self.y_train = embedding.generate_embeddings(generate_data_2.main("bigdata"), \
                embedding_dict, self.config.embed_size, self.config.threshold)
        self.X_dev, self.y_dev = embedding.generate_embeddings(generate_data_2.main("bigdata"), \
                embedding_dict, self.config.embed_size, self.config.threshold)
        self.X_test, self.y_test = embedding.generate_embeddings(generate_data_2.main("bigtest"), \
                embedding_dict, self.config.embed_size, self.config.threshold)
        self.X_test, self.y_test = generate_data_2.shuffle(self.X_test, self.y_test, 100)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, Config.embed_size])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, Config.label_size])

    def create_feed_dict(self, input_batch, label_batch=None):
        feed_dict = {
            self.input_placeholder:input_batch
        }

        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch

        return feed_dict

    def add_model(self):

        with tf.variable_scope("Layer1") as scope:
            W = tf.get_variable("W", [self.config.embed_size, self.config.hidden_size], initializer = tf.truncated_normal_initializer(), dtype=tf.float32)
            b1 = tf.get_variable("b1", [self.config.hidden_size], initializer = tf.zeros_initializer(), dtype=tf.float32)
            h = tf.tanh(tf.matmul(self.input_placeholder, W) + b1)

        with tf.variable_scope("Layer2") as scope:
            U = tf.get_variable("U", [self.config.hidden_size, self.config.label_size], initializer = tf.truncated_normal_initializer(), dtype=tf.float32)
            b2 = tf.get_variable("b2", [self.config.label_size], initializer = tf.zeros_initializer(), dtype=tf.float32)

        output = tf.matmul(h, U) + b2

        return output

    def add_loss_op(self, pred):
        stabilize = lambda x: x - tf.reduce_max(x, reduction_indices=[1], keep_dims=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=stabilize(pred), labels=self.labels_placeholder))

        return loss

    def add_training_op(self, loss):
        with tf.name_scope("train") as scope:
            opt_type = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt_type.minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        y = self.add_model()

        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_prediction = tf.equal(
                tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, input_labels):
        orig_X, orig_y = input_data, input_labels
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        x, y = generate_data_2.shuffle(orig_X, orig_y, self.config.batch_size)
        for step in range(len(x)):
            feed = self.create_feed_dict(input_batch = x, label_batch = y)
            loss, total_correct, _ = session.run(
                    [self.loss, self.correct_predictions, self.train_op],
                    feed_dict = feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

    def predict(self, session, X, y):
        losses = []
        results = []
        total_correct_examples = 0
        total_processed_examples = 0
        x, y = generate_data_2.shuffle(X, y, batch_size = self.config.batch_size)
        for step in range(len(x)):
            feed = self.create_feed_dict(input_batch = x, label_batch = y)
            loss, preds, correct = session.run(
                    [self.loss, self.predictions, self.correct_predictions], feed_dict = feed)
            losses.append(loss)
            total_processed_examples += len(x)
            total_correct_examples += correct
            predicted_indices = preds.argmax(axis = 1)
            results.extend(predicted_indices)
        return np.mean(losses), results, total_correct_examples / float(total_processed_examples)

def save_predictions(predictions, filename):
    with open(filename, "wb") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")

def run_model(hl, lr):
    config = Config(hl, lr)
    with tf.Graph().as_default():
        model = ForwardModel(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            for epoch in range(config.max_epochs):
                print "Epoch {}".format(epoch)
                start = time.time()
                train_loss, train_acc = model.run_epoch(session, model.X_train, model.y_train)
                val_loss, predictions, _ = model.predict(session, model.X_dev, model.y_dev)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Validation loss: {}'.format(val_loss)
                if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  best_val_epoch = epoch
                  saver.save(session, './weights/cleanfeedforward.weights')
                if epoch - best_val_epoch > config.early_stopping:
                  break
                print "Total time: {}".format(time.time() - start)

            saver.restore(session, './weights/cleanfeedforward.weights')
            print "Test"
            print "=-=-="
            print "Writing predictions"
            _, predictions, acc = model.predict(session, model.X_test, model.y_test)
            print "The model achieved " + str(round(acc * 100, 2)) + "% on the test set"
            print predictions, model.y_test
            with open("./test_results/clean_forward_data.txt", "a") as fp:
                fp.write("Hidden Layer Size: {}, Learning Rate: {}, Training Loss: {}, Test Acc: {}".format(hl, lr, best_val_loss, acc))
                fp.write("\n")

if __name__ == "__main__":
    numbers = [.0005, .0006, .0007, .0008, .0009, .001, .002, .003, .004, .005]
    for hl in range(10, 50, 10):
        for lr in numbers:
            run_model(hl, lr)
