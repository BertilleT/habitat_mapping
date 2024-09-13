import torch 
import numpy as np  
import matplotlib.pyplot as plt

def tune_alpha1(model, val_dl, device, alpha1_values, save_path='precision_recall.png'):
    model.eval()
    print('Tuning alpha1')
    
    precisions, recalls, f1s = [], [], []
    
    for alpha1 in alpha1_values:
        print(f'Alpha1: {alpha1}')
        with torch.no_grad():
            precision, recall, f1 = tune_alpha1_valid(model, val_dl, device, alpha1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
    
    # Plot Precision vs Recall
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    for i, txt in enumerate(alpha1_values):
        plt.annotate(txt, (recalls[i], precisions[i]))
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.title('Precision against Recall')
    plt.savefig(save_path)
    plt.close()

    best_alpha1 = alpha1_values[np.argmax(f1s)]
    print(f'Best alpha1: {best_alpha1}')

    # Plot F1 vs Alpha1
    plt.plot(alpha1_values, f1s)
    plt.xlabel('Alpha1')
    plt.ylabel('F1')
    plt.title('F1 against Alpha1')
    plt.axvline(x=best_alpha1, color='r', linestyle='--')
    plt.savefig('f1_alpha1.png')
    plt.close()
    
    return best_alpha1, precisions, recalls, f1s


def tune_alpha2(model, val_dl, device, alpha1, alpha2_values, save_path='precision_recall_by_class.png'):
    model.eval()
    
    precisions, recalls, f1s = [], [], []
    
    for alpha2 in alpha2_values:
        print(f'Alpha2: {alpha2}')
        with torch.no_grad():
            precision, recall, f1 = tune_alpha2_valid(model, val_dl, device, alpha1, alpha2)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
    
    # Select precision and recall for each class
    class_colors = ['#789262', '#555555', '#006400', '#00ff00', '#ff4500', '#8a2be2']
    
    for i in range(6):  # assuming 6 classes
        plt.plot([recall[i] for recall in recalls], [precision[i] for precision in precisions], color=class_colors[i])
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision against Recall by class')
    plt.savefig(save_path)
    plt.close()
    
    mean_precisions = [np.mean(precision) for precision in precisions]
    mean_recalls = [np.mean(recall) for recall in recalls]
    
    # Plot Mean Precision vs Mean Recall
    plt.plot(mean_recalls, mean_precisions)
    for i, txt in enumerate(alpha2_values):
        plt.annotate(txt, (mean_recalls[i], mean_precisions[i]))
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.title('Mean Precision against Mean Recall')
    plt.savefig('mean_precision_recall.png')
    plt.close()
    
    best_alpha2 = alpha2_values[np.argmax([np.mean(f1) for f1 in f1s])]
    
    # Plot F1 by Class
    for i in range(6):
        plt.plot(alpha2_values, [f1[i] for f1 in f1s], color=class_colors[i])
    
    plt.xlabel('Alpha2')
    plt.ylabel('F1')
    plt.title('F1 against Alpha2 by class')
    plt.savefig('f1_alpha2_by_class.png')
    plt.close()
    
    return best_alpha2, precisions, recalls, f1s
