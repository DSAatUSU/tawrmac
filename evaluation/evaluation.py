import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors,eval_idx_data_loader, batch_size=200,
                         negative_sampling_strategy='random'):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()

        eval_idx_data_loader_tqdm = tqdm(eval_idx_data_loader, ncols=120)

        for batch_idx, eval_data_indices in enumerate(eval_idx_data_loader_tqdm):
            eval_data_indices = eval_data_indices.numpy()

            sources_batch, destinations_batch = data.sources[eval_data_indices], \
                data.destinations[eval_data_indices]
            edge_idxs_batch = data.edge_idxs[eval_data_indices]
            timestamps_batch = data.timestamps[eval_data_indices]
            size = len(sources_batch)

            if negative_sampling_strategy == 'random':
                _, negative_destinations = negative_edge_sampler.sample(size, sources_batch, destinations_batch,
                                                                        timestamps_batch[0], timestamps_batch[-1])
                negative_sources = sources_batch
            else:
                negative_sources, negative_destinations = negative_edge_sampler.sample(size, sources_batch,
                                                                                       destinations_batch,
                                                                                       timestamps_batch[0],
                                                                                       timestamps_batch[-1])

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_sources,
                                                                  negative_destinations, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors, eval_idx_data_loader):
    pred_prob = np.zeros(len(data.sources))
    # num_instance = len(data.sources)
    # num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()

        eval_idx_data_loader_tqdm = tqdm(eval_idx_data_loader, ncols=120)

        for batch_idx, eval_data_indices in enumerate(eval_idx_data_loader_tqdm):
            eval_data_indices = eval_data_indices.numpy()

            sources_batch, destinations_batch = data.sources[eval_data_indices], \
                data.destinations[eval_data_indices]
            edge_idxs_batch = data.edge_idxs[eval_data_indices]
            timestamps_batch = data.timestamps[eval_data_indices]

            source_embedding, destination_embedding, _,_,_ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         None,
                                                                                         None,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            # pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
            pred_prob[eval_data_indices] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
