
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize


def evaluate_model(history,X_test,y_test,model):
    scores = model.evaluate((X_test),y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)

def plot_conf_matrix(conf_matrix, ticks, title: str = None, xlabel: str = None, ylabel: str = None, savename: str = None):

    if title is None:
        title = 'Confusion Matrix'
    
    if xlabel is None:
        xlabel = 'CNN Model'
    
    if ylabel is None:
        ylabel = 'label'    
    
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(ticks)
    ax.yaxis.set_ticklabels(ticks)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()

def multiclass_roc_auc_score(y_test, y_pred, savename, classes = ['NOTEB','VEB', 'SVEB'], average="macro"):
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(classes): 
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, metrics.auc(fpr, tpr)))
        
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    plt.title('Multiclass ROC')
    plt.legend()
    plt.savefig(savename)
    return metrics.roc_auc_score(y_test, y_pred, average=average)

def compute_specificity(cm, class_index):
    TN = cm.sum() - cm[class_index, :].sum() - cm[:, class_index].sum() + cm[class_index, class_index]
    FP = cm[:, class_index].sum() - cm[class_index, class_index]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return specificity

def calculate_metrics(y_true, y_pred, num_classes=3):
    # Convert probabilities to class predictions if needed
    if y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # Calculate per-class accuracy, precision (PPV), recall (sensitivity), F1
    per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)  # True Positive Rate
    specificity = np.array([compute_specificity(cm, i) for i in range(num_classes)])
    ppv = precision_score(y_true, y_pred, average=None, labels=range(num_classes))  # Positive Predictive Value per class
    f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes))  # F1 Score per class
    accuracy = accuracy_score(y_true, y_pred)  # Overall accuracy
    return accuracy, per_class_accuracy, sensitivity, specificity, ppv, f1


def plot_multiclass_roc(val_y, val_pred, num_classes=3):
    # Binarize the labels
    val_y_bin = label_binarize(val_y, classes=range(num_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(val_y_bin[:, i], val_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC and AUC')
    plt.legend(loc="lower right")
    plt.show()


plot_multiclass_roc(val_y, val_pred, num_classes=3)


def bootstrap_metric(y_true, y_pred, func, n_bootstrap=10000):
    """ Bootstrap a metric function to find its confidence interval. """
    indices = np.arange(len(y_true))
    stats = []
    for _ in range(n_bootstrap):
        resample_idx = np.random.choice(indices, size=len(indices), replace=True)
        resampled_true = y_true[resample_idx]
        resampled_pred = y_pred[resample_idx]
        stat = func(resampled_true, resampled_pred)
        stats.append(stat)
    lower_bound = np.percentile(stats, 2.5)
    upper_bound = np.percentile(stats, 97.5)
    return np.mean(stats), lower_bound, upper_bound

# Calculate CI for each metric
acc_ci = bootstrap_metric(y_test, pred_test, lambda y_true, y_pred: np.mean(y_true == y_pred))
prec_ci = bootstrap_metric(y_test, pred_test, lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'))
rec_ci = bootstrap_metric(y_test, pred_test, lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'))
f1_ci = bootstrap_metric(y_test, pred_test, lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'))

print(f"Accuracy CI: {acc_ci}")
print(f"Precision CI: {prec_ci}")
print(f"Recall CI: {rec_ci}")
print(f"F1 Score CI: {f1_ci}")


def bootstrap_metric_class_wise(y_true, y_pred, labels, n_bootstrap=1000):
    """ Bootstrap class-wise metric function to find confidence intervals. """
    metrics = {'sensitivity': [], 'specificity': [], 'precision': [], 'f1_score': [], 'auc': []}
    for _ in range(n_bootstrap):
        resample_idx = resample(np.arange(len(y_true)))  # Bootstrap sample indices
        y_true_resampled = y_true[resample_idx]
        y_pred_resampled = y_pred[resample_idx]
        
        # Calculate precision, recall, f_score, support for each class
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_true_resampled, y_pred_resampled, labels=labels, average=None)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_true_resampled, y_pred_resampled, labels=labels)
        sensitivity = recall  # True Positive Rate
        specificity = []
        for i, label in enumerate(labels):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if tn + fp > 0 else 0)

        # Calculate AUC for each class
        auc_scores = [roc_auc_score(y_true_resampled == lab, y_pred_resampled == lab) for lab in labels]

        # Append results
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['precision'].append(precision)
        metrics['f1_score'].append(f_score)
        metrics['auc'].append(auc_scores)
    
    # Compute mean, lower and upper bounds for each metric
    result = {}
    for key, values in metrics.items():
        values_array = np.array(values)
        mean_metrics = np.mean(values_array, axis=0)
        lower_bounds = np.percentile(values_array, 2.5, axis=0)
        upper_bounds = np.percentile(values_array, 97.5, axis=0)
        result[key] = (mean_metrics, lower_bounds, upper_bounds)
    return result

# Example usage
labels = [0, 1, 2]  # Class labels
bootstrap_results = bootstrap_metric_class_wise(y_test, pred_test, labels, n_bootstrap=1000)

# Printing the results
print("Metrics for Each Class:")
for label in labels:
    print(f"\nClass {label}:")
    for metric in ['sensitivity', 'specificity', 'precision', 'f1_score', 'auc']:
        mean_val, lower_val, upper_val = bootstrap_results[metric]
        print(f"{metric.capitalize()}: {mean_val[label]:.3f} ({lower_val[label]:.3f}-{upper_val[label]:.3f})")



def bootstrap_auc(y_true, y_pred_proba, classes, n_bootstrap=1000):
    # Convert true labels to binary format (one-hot encoding)
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]
    auc_scores = {i: [] for i in range(n_classes)}
    indices = np.arange(len(y_true_bin))
    
    for _ in range(n_bootstrap):
        resample_idx = np.random.choice(indices, size=len(indices), replace=True)
        for i in range(n_classes):
            score = roc_auc_score(y_true_bin[resample_idx, i], y_pred_proba[resample_idx, i])
            auc_scores[i].append(score)
    
    auc_ci = {}
    for i in range(n_classes):
        auc_mean = np.mean(auc_scores[i])
        lower_bound = np.percentile(auc_scores[i], 2.5)
        upper_bound = np.percentile(auc_scores[i], 97.5)
        auc_ci[i] = (auc_mean, lower_bound, upper_bound)
    return auc_ci

# Assuming pred_label_prob1, y_test1, etc. are defined and classes list is available
classes = [0, 1, 2]  # Adjust classes as per your data
auc_ci1 = bootstrap_auc(y_test1, pred_label_prob1, classes)
auc_ci2 = bootstrap_auc(y_test2, pred_label_prob2, classes)
auc_ci3 = bootstrap_auc(y_test, pred_label_prob, classes)

# Print the AUC with CI for each class
print("AUC with CI for dataset 1:", auc_ci1)
print("AUC with CI for dataset 2:", auc_ci2)
print("AUC with CI for dataset 3:", auc_ci3)



def plot_roc_curves(predictions, true_labels, dataset_names):

    plt.figure(figsize=(10, 8))

    # Colors for each class
    colors = ['blue', 'red', 'green']
    class_labels = ['NOTEB', 'VEB', 'SVEB']

    for i, color in enumerate(colors):
        for j, (preds, labels, name) in enumerate(zip(predictions, true_labels, dataset_names)):
            # Convert labels to binary format
            labels_bin = label_binarize(labels, classes=[0, 1, 2])
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for k in range(3):
                fpr[k], tpr[k], _ = roc_curve(labels_bin[:, k], preds[:, k])
                roc_auc[k] = auc(fpr[k], tpr[k])

            # Creating label based on dataset and class
            label = f"Class {class_labels[i]} - {name} (AUC = {roc_auc[i]:.2f})"
            linestyle = ['-', '--', ':'][j]  # Different line style for each dataset
            plt.plot(fpr[i], tpr[i], color=color, linestyle=linestyle, lw=2, label=label)

    # Plot the random classifier curve
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    # Set the plot labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Three Test Datasets")
    plt.legend(loc="lower right")
    plt.show()

def one_v_rest(y_test, pred_label_prob):
    # Convert y_test to pandas Series
    y_test_series = pd.Series(y_test)

    # Plots the Probability Distributions and the ROC Curves One vs Rest
    plt.figure(figsize=(12, 8))
    bins = [i / 20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    classes = ['NOTEB', 'VEB', 'SVEB']
    cardio_sub = y_test_series.replace({0: 'NOTEB', 2: 'SVEB', 1: 'VEB'}).to_list()
    
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]

        # Prepares an auxiliary dataframe to help with the plots
        df_aux = pd.DataFrame({'class': cardio_sub, 'prob': pred_label_prob[:, i]})
        df_aux['class'] = [1 if y == c else 0 for y in df_aux['class']]
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i + 1)
        sns.histplot(x='prob', data=df_aux, hue='class', color='b', ax=ax, bins=bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i + 4)
        fpr, tpr, _ = roc_curve(df_aux['class'], df_aux['prob'])
        roc_auc_ovr[c] = auc(fpr, tpr)
        ax_bottom.plot(fpr, tpr, color='b', lw=2)
        ax_bottom.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        ax_bottom.set_title("ROC Curve OvR")
        ax_bottom.set_xlabel("False Positive Rate")
        ax_bottom.set_ylabel("True Positive Rate")

    plt.tight_layout()
    plt.savefig("onevsother.png") 
    
    return plt.show(), roc_auc_ovr



def plot_zoomed_roc(fpr, tpr, auc_ci, class_labels, dataset_names, file_name="Zoomed_AUC.png"):
    """
    Plots zoomed ROC curves with confidence intervals for multiple datasets and classes.

    Parameters:
    fpr (list): List containing FPR dictionaries for each dataset.
    tpr (list): List containing TPR dictionaries for each dataset.
    auc_ci (list): List containing AUC CI tuples for each dataset and class.
    class_labels (list): List of class labels.
    dataset_names (list): List of dataset names.
    file_name (str): Filename to save the plot.
    """
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']

    for i, color in enumerate(colors):
        for j, name in enumerate(dataset_names):
            label = f"Class {class_labels[i]} - {name} (AUC = {auc_ci[j][i][0]:.2f} CI=({auc_ci[j][i][1]:.2f}-{auc_ci[j][i][2]:.2f}))"
            linestyle = ['-', '--', ':'][j]
            plt.plot(fpr[j][i], tpr[j][i], color=color, linestyle=linestyle, lw=2, label=label)

    # Adjust axis limits to focus on the area of interest
    plt.xlim([0, 0.3])
    plt.ylim([0.8, 1])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Zoomed ROC Curves for Three Test Datasets")
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.show()
