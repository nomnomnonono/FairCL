import numpy as np

    
def split_ssl_data(data, sens, target, num_labels, num_groups, index=None, include_lb_to_ulb=False):
    lb_data, lbs, lby, lb_idx = sample_labeled_data(data, sens, target, num_labels, num_groups, index=index)

    if include_lb_to_ulb:
        return lb_data, lbs, lby, data, sens, target
    else:
        ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx)))) #unlabeled_data index of data
        return lb_data, lbs, lby, data[ulb_idx], sens[ulb_idx], target[ulb_idx]


def sample_labeled_data(data, sens, target, num_labels, num_groups, num_classes=2, index=None):
    if index is not None:
        index = np.array(index, dtype=np.int32)
        return data[index], sens[index], index
    
    lb_data, lbs, lby, lb_idx = [], [], [], []
    samples_per_class = int((len(data) * num_labels) / (num_classes * num_groups))
    for g in range(num_groups):
        for c in range(num_classes):
            idx = np.where((sens==g)&(target==c))[0]
            idx = np.random.choice(idx, samples_per_class, False)
            lb_idx.extend(idx)
            lb_data.extend(data[idx])
            lbs.extend(sens[idx])
            lby.extend(target[idx])
        
    return np.array(lb_data), np.array(lbs), np.array(lby), np.array(lb_idx)
