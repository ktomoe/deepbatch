import torch

def multiclass_acc(batch_result):
    outputs = batch_result['outputs']
    labels = batch_result['labels'].data

    _, preds = torch.max(outputs, 1)

    results = []
    corrects = torch.sum(preds == labels)

    for ii in range(5):
        index = labels == ii
        ipreds = preds[index]
        ilabels = labels[index]

        if len(ilabels) == 0:
            results.append(1.)
        else:
            corrects = torch.sum(ipreds == ilabels)
            results.append(corrects.detach().item() / len(ilabels))

    results.append(sum(results)/5)
    return results
