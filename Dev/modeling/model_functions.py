# packages for data loading, data analysis, and data preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import metrics

# packages for model evaluation
from sklearn.model_selection import KFold,learning_curve
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score,roc_curve, auc
from tqdm import tqdm

# plot learning curve
def plot_learning_curve(estimator, title, score, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    score : str
        A scoring parameter used to evaluate performance of the model.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,scoring=score,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

# plot ROC curve

def plot_roc_curve(estimator,x_train,y_train,x_test,y_test):

    """
    This function plots the ROC curve.

    Parameters
    ----------
    estimator : a single model instance
        ex : KNeighborsClassifier()
    x_train : Training X input data
    y_train : Training Y input data
    x_test : Testing X input data
    y_test : Testing y input data

    """

    estimator.fit(x_train,y_train) # train the model
    name = type(estimator).__name__
    y_pred = estimator.predict(x_test) # predict the test data

   # plot ROC curves
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, estimator.predict_proba(x_test)[:,1])
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,estimator.predict(x_test))
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, auc))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()   # Display
    return 1

# plot confusion matrix
def plot_confusion_matrix(estimator, x_train, y_train, x_test, y_test, classes,
                          normalize=False,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Parameters
    ----------
    estimator : a single model instance
        ex : KNeighborsClassifier()
    x_train : Training X input data
    y_train : Training Y input data
    x_test : Testing X input data
    y_test : Testing y input data
    classes : list like input for labeling the axis
        ex : classes= ['Buy=1','Sell=0']
    normalize : Bool that can be applied by setting `normalize=True`.
    cmap : color map type for confusion matrix. Default plt.cm.blues.
    """
    estimator.fit(x_train,y_train) # train the model
    y_pred = estimator.predict(x_test) # predict the test data
    cm = confusion_matrix(y_test,y_pred, labels=[1,0])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{type(estimator).__name__} Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# evaluate multiple models training learning curves
def eval_learning_curves(models,x,y,scoring):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve for each model passed in.

    Parameters
    ----------
    models : List of model name and instance
        ex: [('LR',LogisticRegression(max_iter=300)),
          ('KNN',KNeighborsClassifier())]
    x : Training X input Data
    x : Training Y input data
    scoring : List of scoring parameters used to evaluate your models
        ex: [accuracy, precision, recall]
    """
   # stores the name of model and results
    names =[]
    results =[]
    seed = 42
   # loop through each model, get the name, perform kflod using crossvalidate
    for score in scoring:
        for name,model in tqdm(models):
            kfold = KFold(n_splits=5,random_state=seed,shuffle=True)
            plot_learning_curve(estimator=model,X=x,y=y,cv=kfold,train_sizes=np.linspace(0.1, 1, 10),
                               n_jobs=-1,title= f"{name} {score} Learning Curve",score=score)
    return results

# evaluate multiple roc curves on training and test data
def eval_model_performance(models,x_train,y_train,x_test,y_test):

    """
    Generate 2 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve for each model passed in.

    Parameters
    ----------
    models : List of model name and instance
        ex: [('LR',LogisticRegression(max_iter=300)),
          ('KNN',KNeighborsClassifier())]
    x_train : Training X input data
    y_train : Training Y input data
    x_test : Testing X input data
    y_test : Testing y input data
    scoring : List of scoring parameters used to evaluate your models
        ex: [accuracy, precision, recall]
    """
    # create models report
    models_report = pd.DataFrame(columns = ['Model', 'Precision_score',
                                            'Recall_score','F1_score', 'Accuracy'])

    for name,model in tqdm(models):
        model.fit(x_train,y_train) # train the model
        y_pred = model.predict(x_test) # predict the test data
        t = pd.Series({
                     'Model': name,
                     'Precision_score': precision_score(y_test, y_pred,average='macro'),
                     'Recall_score': recall_score(y_test, y_pred,average='macro'),
                     'F1_score': f1_score(y_test, y_pred,average='macro'),
                     'Accuracy': accuracy_score(y_test, y_pred)}
                   )
        models_report = models_report.append(t, ignore_index = True)
    # plot multiple ROC curves
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
        # Calculate Area under the curve to display on the plot
        auc = metrics.roc_auc_score(y_test,model.predict(x_test))
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, auc))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()   # Display

    models_report
    return models_report
