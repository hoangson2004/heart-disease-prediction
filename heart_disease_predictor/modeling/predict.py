from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def predict(y_test, y_pred):
    print('The details for confusion matrix is =')
    print(classification_report(y_test, y_pred))


    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data=cm,
                               columns=['Predicted:0', 'Predicted:1'],
                               index=['Actual:0', 'Actual:1'])

    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
    plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()