def accuracy(pred_tokens, real_labels):
    accuracy_match = [p == t for p, t in zip(pred_tokens, real_labels)]
    accuracy = len([m for m in accuracy_match if m]) / len(real_labels)
    print(accuracy)