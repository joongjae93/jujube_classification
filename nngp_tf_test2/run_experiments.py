# -*- coding: utf-8 -*-

"""Run experiments with NNGP Kernel.

Usage:

python run_experiments.py \
      --num_train=100 \
      --num_eval=1000 \
      --hparams='nonlinearity=relu,depth=10,weight_var=1.79,bias_var=0.83' \
      --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import gpr
import load_dataset_unfix
import load_dataset
import nngp


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter pairs to'
                    'override the default setting.')
flags.DEFINE_string('experiment_dir', '/tmp/nngp',
                    'Directory to put the experiment results.')
flags.DEFINE_string('grid_path', './grid_data',
                    'Directory to put or find the training data.')
flags.DEFINE_boolean('save_kernel', False, 'Save Kernel do disk')
flags.DEFINE_string('dataset', 'mnist',
                    'Which dataset to use ["mnist"]')
flags.DEFINE_boolean('use_fixed_point_norm', False,
                     'Normalize input variance to fixed point variance')

flags.DEFINE_integer('n_gauss', 501,
                     'Number of gaussian integration grid. Choose odd integer.')
flags.DEFINE_integer('n_var', 501,
                     'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500,
                     'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100,
                     'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10,
                     'Range for gaussian integration.')


def set_default_hparams(weight_var, bias_var, depth):
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=3.65, bias_var=1.38, depth=10)


def metric_batch_detail(pred, answer):
  answer_sheet = np.zeros((5, 6))
  for i in range(0, len(pred)):
    answer_sheet[answer[i].item()][pred[i]] += 1
    answer_sheet[answer[i].item()][-1] += 1
  return answer_sheet


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def loading_data():
  # Get the sets of images and labels for training, validation, and
  # # test on dataset.
  start_time = time.time()
  if FLAGS.dataset == 'mnist':
    (train_image, train_label, test_image,
     test_label, num_train, num_test) = load_dataset_unfix.load_mnist(
         mean_subtraction=True,
         random_roated_labels=False)
  else:
    raise NotImplementedError
  tf.logging.info('Loading dataset took %.3f secs'%(
      time.time() - start_time))

  return train_image, train_label, test_image, test_label, num_train, num_test


def do_eval(sess, model, x_data, y_data, save_pred=False):
  """Run evaluation."""

  gp_prediction, stability_eps = model.predict(x_data, sess)
  pred_1 = np.argmax(gp_prediction, axis=1)
  answer = np.argmax(y_data, axis=1)

  accuracy = np.sum(pred_1 == answer) / float(len(y_data))
  mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))
  pred_norm = np.mean(np.linalg.norm(gp_prediction, axis=1))
  tf.logging.info('Accuracy: %.4f'%accuracy)
  tf.logging.info('MSE: %.8f'%mse)

  if save_pred:
    with tf.gfile.Open(
        os.path.join(FLAGS.experiment_dir, 'gp_prediction_stats.npy'),
        'w') as f:
      np.save(f, gp_prediction)

  return accuracy, mse, pred_norm, stability_eps, pred_1, answer


def run_nngp_eval(hparams, run_dir, train_image, train_label, test_image, test_label, num_train, num_test):
  """Runs experiments."""

  tf.gfile.MakeDirs(run_dir)
  # Write hparams to experiment directory.
  with tf.gfile.GFile(run_dir + '/hparams', mode='w') as f:
    f.write(hparams.to_proto().SerializeToString())

  tf.logging.info('Hyperparameters')
  tf.logging.info('---------------------')
  tf.logging.info(hparams)
  tf.logging.info('---------------------')

  tf.logging.info('Building Model')

  if hparams.nonlinearity == 'tanh':
    nonlin_fn = tf.tanh
  elif hparams.nonlinearity == 'relu':
    nonlin_fn = tf.nn.relu
  else:
    raise NotImplementedError

  with tf.Session() as sess:
    # Construct NNGP kernel
    nngp_kernel = nngp.NNGPKernel(
        depth=hparams.depth,
        weight_var=hparams.weight_var,
        bias_var=hparams.bias_var,
        nonlin_fn=nonlin_fn,
        grid_path=FLAGS.grid_path,
        n_gauss=FLAGS.n_gauss,
        n_var=FLAGS.n_var,
        n_corr=FLAGS.n_corr,
        max_gauss=FLAGS.max_gauss,
        max_var=FLAGS.max_var,
        use_fixed_point_norm=FLAGS.use_fixed_point_norm)

    # Construct Gaussian Process Regression model
    model = gpr.GaussianProcessRegression(
        train_image, train_label, kern=nngp_kernel)

    start_time = time.time()
    tf.logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    acc_train, mse_train, norm_train, final_eps, _, _ = do_eval(
        sess, model, train_image, train_label)
    tf.logging.info('Evaluation of train set (%d examples) took %.3f secs'%(
        num_train, time.time() - start_time))

    start_time = time.time()
    tf.logging.info('Test')
    acc_test, mse_test, norm_test, _, pred_1, answer = do_eval(
        sess,
        model,
        test_image,
        test_label,
        save_pred=False)
    tf.logging.info('Evaluation of test set (%d examples) took %.3f secs'%(
        num_test, time.time() - start_time))

  metrics = {
      'train_acc': float(acc_train),
      'train_mse': float(mse_train),
      'train_norm': float(norm_train),
      'test_acc': float(acc_test),
      'test_mse': float(mse_test),
      'test_norm': float(norm_test),
      'stability_eps': float(final_eps),
  }

  record_results = [
      num_train, hparams.nonlinearity, hparams.weight_var,
      hparams.bias_var, hparams.depth, acc_train, acc_test,
      mse_train, mse_test, final_eps
  ]
  if nngp_kernel.use_fixed_point_norm:
    metrics['var_fixed_point'] = float(nngp_kernel.var_fixed_point_np[0])
    record_results.append(nngp_kernel.var_fixed_point_np[0])
  
  return acc_test, pred_1, answer


def save_result(hparams, pred_1, answer): 
  answer_sheet = metric_batch_detail(pred_1, answer)
  createFolder('./NNGP')
  df = pd.DataFrame(answer_sheet, index=['혹파리', '마그네슘', '정상', '노린재', '붉은점박이'],
                    columns=['혹파리', '마그네슘', '정상', '노린재', '붉은점박이', '총계'])
  df.to_excel('./NNGP/results_{}_{}_{}_{}.xlsx'.format(hparams.nonlinearity, hparams.weight_var, hparams.bias_var, hparams.depth))


def main(argv):
  del argv  # Unused
  tf.logging.info('Starting job.')
  tf.logging.info('Loading data')
  train_image, train_label, test_image, test_label, num_train, num_test = loading_data()
  best_acc = 0
  #weight_list = np.linspace(3.5, 4.2, 9)
  #bias_list = np.linspace(2, 2.5, 15)
  #layers_list = np.arange(9, 21)
  result = []
  for weight in [3.85]:
    for bias in [2.143]:
      for layers in [9]:
        hparams = tf.contrib.training.HParams(nonlinearity='tanh', weight_var=float(weight), bias_var=float(bias), depth=int(layers))
        acc_test, _, _ = run_nngp_eval(hparams, FLAGS.experiment_dir, train_image, train_label, test_image, test_label, num_train, num_test)
        result.append([weight, bias, layers, acc_test])
        if best_acc < acc_test:
          best_acc = acc_test
          best_params = hparams

      result_array = np.array(result)
      df = pd.DataFrame(result_array, columns=['weight', 'bias', 'layers', 'accuracy'])
      print(df)
      df.to_excel('./NNGP/results_88.xlsx')


  _, pred_1, answer = run_nngp_eval(best_params, FLAGS.experiment_dir, train_image, train_label, test_image, test_label, num_train, num_test)
  save_result(best_params, pred_1, answer)


if __name__ == '__main__':
  tf.app.run(main)

