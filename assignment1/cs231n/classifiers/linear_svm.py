import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  delta = 1.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        dW[:, j] += X[i].T
        dW[:, y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1.0
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_scores = scores[np.arange(scores.shape[0]), y]
  margins = np.maximum(0, scores - np.tile(correct_scores, (scores.shape[1], 1)).T + delta)
  margins[np.arange(scores.shape[0]), y] = 0
  loss = np.sum(margins) / num_train

  X_mul = np.zeros(margins.shape)
  X_mul[margins > 0] = 1
  incorrect_cnt = np.sum(X_mul, axis=1)
  X_mul[np.arange(num_train), y] = -incorrect_cnt
  dW = X.T.dot(X_mul) / num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
