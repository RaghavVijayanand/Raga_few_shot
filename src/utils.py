import numpy as np
import matplotlib.pyplot as plt

def plot_raga_segments(times, preds, label_map, out_path=None):
    plt.figure(figsize=(15, 2))
    for i, (start, end, label) in enumerate(zip(times[:-1], times[1:], preds)):
        plt.fill_between([start, end], 0, 1, color=f'C{label%10}', alpha=0.6, label=label_map[label] if i==0 else "")
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.title('Predicted Raga Segments')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def calculate_segment_accuracy(true_segments, pred_segments):
    correct = 0
    total = len(true_segments)
    for true, pred in zip(true_segments, pred_segments):
        if true == pred:
            correct += 1
    return correct / total

def calculate_f1_score(true_segments, pred_segments):
    from sklearn.metrics import f1_score
    return f1_score(true_segments, pred_segments, average='weighted')

def calculate_temporal_overlap_accuracy(true_times, true_segments, pred_times, pred_segments):
    true_intervals = [(t1, t2, s) for t1, t2, s in zip(true_times[:-1], true_times[1:], true_segments)]
    pred_intervals = [(t1, t2, s) for t1, t2, s in zip(pred_times[:-1], pred_times[1:], pred_segments)]
    
    overlap_correct = 0
    for true_interval in true_intervals:
        for pred_interval in pred_intervals:
            if true_interval[2] == pred_interval[2]:
                start_overlap = max(true_interval[0], pred_interval[0])
                end_overlap = min(true_interval[1], pred_interval[1])
                if start_overlap < end_overlap:
                    overlap_correct += 1
                    break
    
    return overlap_correct / len(true_intervals)
