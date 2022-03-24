# search_hyperparams.py
# ---------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import solvers
import util


def search_hyperparams(train_data, train_labels, val_data, val_labels,
                       learning_rates, momentums, batch_sizes, iterations,
                       model_class, init_param_values=None, use_bn=False):
    """
    Question 8: Evaluate various setups of hyperparameter and find the best one.

    Args:
        learning rate, momentums, batch_sizes are lists of the same length.
        The N-th elements from the lists form the N-th hyperparameter tuple.

    Returns:
        A model that corresponds to the best hyperparameter tuple, and the index
            of the best hyperparameter tuple

    Your implementation will train a model using each hyperparameter tuple and
    compares their accuracy on validation set to pick the best one.

    You must use MinibatchStochasticGradientDescentSolver.

    Useful methods:
    solver.solve(...)
    model.accuracy(...)
    """
    # Check length of inputs all the same
    hyperparams = [learning_rates, momentums, batch_sizes]
    for hyperparam in hyperparams:
        if len(hyperparam) != len(hyperparams[0]):
            raise ValueError('The hyperparameter lists need to be equal in length')
    hyperparams = zip(*hyperparams)
    # python *+VARIABLE: http://stackoverflow.com/questions/11315010/what-do-and-before-a-variable-name-mean-in-a-function-signature

    # Initialize the models
    models = []
    for learning_rate, momentum, batch_size in hyperparams:
        try:
            model = model_class(use_batchnorm=use_bn)
        except:
            model = model_class()
        if init_param_values is None:
            init_param_values = model.get_param_values()
        else:
            model.set_param_values(init_param_values)
        models.append(model)

    val_accuracies = []
    # Loop over hyperparams
    for model, (learning_rate, momentum, batch_size) in zip(models, hyperparams):
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable that uses the Mini-Batch SGD solver
        solver = solvers.MinibatchStochasticGradientDescentSolver(learning_rate, iterations, batch_size, momentum)

        # Train the model on the training set
        solver.solve(train_data, train_labels, val_data, val_labels, model)

        # Evaluate the accuracy of the model on the validation set
        model_accuracy = model.accuracy(val_data, val_labels)

        # Append the evaluated accuracy to the list of accuracies
        val_accuracies.append(model_accuracy)

    # Get the index of the best value in the list of accuracies
    best_index = val_accuracies.index(max(val_accuracies))

    # Get the model with the best accuracy from the list of models
    best_model = models[best_index]

    # Get the hyperparameters with the best accuracy from the list of hyperparameters
    best_hyperparams = hyperparams[best_index]
    ####################################################################################################################
    # util.raiseNotDefined()
    return best_model, best_hyperparams
