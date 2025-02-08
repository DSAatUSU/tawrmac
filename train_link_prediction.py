import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import datetime
from evaluation.evaluation import eval_edge_prediction
from model.tawrmac import TAWRMAC
from utils.utils import EarlyStopMonitor, set_random_seed, NegativeEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from utils.DataLoader import get_idx_data_loader

### Argument and global variables
parser = argparse.ArgumentParser('TAWRMAC link prediction')
parser.add_argument('-d', '--data', type=str,
                    choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'uci',
                             'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'],
                    help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=3, help='Idx for the gpu to use')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')

parser.add_argument('--negative_sample_strategy', type=str, default='random',
                    choices=['random', 'historical', 'inductive'],
                    help='strategy for the negative edge sampling')

# parser.add_argument('--use_global_memory', action='store_true',
#                     help='Whether to augment the model with a global graph memory')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--enable_walk', action='store_true',
                    help='Whether to use causal anonymous walks')
parser.add_argument('--enable_restart', action='store_true',
                    help='Whether to use walk restart')
parser.add_argument('--pick_new_neighbors', action='store_true',
                    help='Whether to pick new neighbors in restart')
parser.add_argument('--position_feat_dim', type=int, default=100, help='Dimensions of the walk embedding')
parser.add_argument('--walk_length', type=int, default=4, help='Length of walks')
parser.add_argument('--num_walk_heads', type=int, default=4, help='Number of walk heads')
parser.add_argument('--num_walks', type=int, default=10, help='Number of walk (Default was set to 10)')
parser.add_argument('--walk_emb_dim', type=int, default=172, help='Dimension of TAWR embedding.')
parser.add_argument('--fixed_time_dim', type=int, default=20, help='Fixed time dimension')
parser.add_argument('--max_input_seq_length', type=int, default=32, help='Max input sequence length for co-occurrence')
parser.add_argument('--enable_neighbor_cooc', action='store_true',
                    help='Whether to enable neighbor co-occurrence encoding')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MEMORY_DIM = args.memory_dim
NEG_SAMPLE_STRATEGY = args.negative_sample_strategy

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path(f"./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

currdate = str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))

get_checkpoint_path = lambda epoch, n_run: f'./saved_checkpoints/{args.data}-run-{n_run}-ep-{epoch}-{currdate}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
experiment_info = f'{args.data}_{args.negative_sample_strategy}'

if args.enable_walk:
    experiment_info += '_walk'
if args.enable_restart:
    experiment_info += '_restart'
if args.use_memory:
    experiment_info += '_mem'
if args.enable_neighbor_cooc:
    experiment_info += '_cooc'
if args.pick_new_neighbors:
    experiment_info += '_nn'

get_model_path = lambda n_run: f'./saved_models/{experiment_info}_run_{n_run}_{currdate}.pth'

fh = logging.FileHandler(f'log/{experiment_info}_{currdate}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

logger.info('Current Date: {}'.format(currdate))

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(DATA)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform, seed=1)

train_rand_sampler = NegativeEdgeSampler(train_data.sources, train_data.destinations)

val_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, seed=0,
                                       interact_times=full_data.timestamps,
                                       last_observed_time=train_data.timestamps[-1],
                                       negative_sample_strategy=args.negative_sample_strategy)
nn_val_rand_sampler = NegativeEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1,
                                          interact_times=new_node_val_data.timestamps,
                                          last_observed_time=train_data.timestamps[-1],
                                          negative_sample_strategy=args.negative_sample_strategy)
test_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, seed=2,
                                        interact_times=full_data.timestamps,
                                        last_observed_time=val_data.timestamps[-1],
                                        negative_sample_strategy=args.negative_sample_strategy)
nn_test_rand_sampler = NegativeEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3,
                                           interact_times=new_node_test_data.timestamps,
                                           last_observed_time=val_data.timestamps[-1],
                                           negative_sample_strategy=args.negative_sample_strategy)

# get data loaders
train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.sources))),
                                            batch_size=BATCH_SIZE, shuffle=False)
val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.sources))),
                                          batch_size=BATCH_SIZE, shuffle=False)
new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.sources))),
                                                   batch_size=BATCH_SIZE, shuffle=False)
test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.sources))),
                                           batch_size=BATCH_SIZE, shuffle=False)
new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.sources))),
                                                    batch_size=BATCH_SIZE, shuffle=False)

# Set device
# device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)



for i in range(0, args.n_runs):
    set_random_seed(i)


    results_path = f"results/{args.data}_{args.negative_sample_strategy}_run_{i}_{currdate}.pkl"
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    model = TAWRMAC(neighbor_finder=train_ngh_finder, node_features=node_features,
                    edge_features=edge_features, device=device,
                    n_layers=NUM_LAYER,
                    n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                    memory_dimension=MEMORY_DIM,
                    n_neighbors=NUM_NEIGHBORS,
                    enable_walk=args.enable_walk,
                    enable_restart=args.enable_restart,
                    pick_new_neighbors=args.pick_new_neighbors,
                    enable_neighbor_cooc=args.enable_neighbor_cooc,
                    walk_emb_dim=args.walk_emb_dim,
                    time_dim = TIME_DIM,
                    fixed_time_dim=args.fixed_time_dim,
                    max_input_seq_length=args.max_input_seq_length,
                    num_walks=args.num_walks,
                    position_feat_dim=args.position_feat_dim,
                    walk_length=args.walk_length,
                    num_walk_heads=args.num_walk_heads)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []


    early_stopper = EarlyStopMonitor(max_round=args.patience)

    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of epoch
        if USE_MEMORY:
            model.memory.__init_memory__()

        # Train using only training graph
        model.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))

        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            loss = 0
            optimizer.zero_grad()
            train_data_indices = train_data_indices.numpy()

            sources_batch, destinations_batch = train_data.sources[train_data_indices], \
                train_data.destinations[train_data_indices]
            edge_idxs_batch = train_data.edge_idxs[train_data_indices]
            timestamps_batch = train_data.timestamps[train_data_indices]

            size = len(sources_batch)

            _, negative_destinations = train_rand_sampler.sample(size, sources_batch, destinations_batch,
                                                                 timestamps_batch[0], timestamps_batch[-1])
            negative_sources = sources_batch

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            model = model.train()
            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_sources,
                                                                  negative_destinations,
                                                                  timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

            loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)


            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())


            if USE_MEMORY:
                model.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        model.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:

            train_memory_backup = model.memory.backup_memory()

        val_ap, val_auc = eval_edge_prediction(model=model,
                                               negative_edge_sampler=val_rand_sampler,
                                               data=val_data,
                                               n_neighbors=NUM_NEIGHBORS,
                                               negative_sampling_strategy=NEG_SAMPLE_STRATEGY,
                                               eval_idx_data_loader=val_idx_data_loader)
        if USE_MEMORY:
            val_memory_backup = model.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            model.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction(model=model,
                                                     negative_edge_sampler=nn_val_rand_sampler,
                                                     data=new_node_val_data,
                                                     n_neighbors=NUM_NEIGHBORS,
                                                     negative_sampling_strategy=NEG_SAMPLE_STRATEGY,
                                                     eval_idx_data_loader=new_node_val_idx_data_loader)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            model.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val auc: {:.4f}, new node val auc: {:.4f}'.format(val_auc, nn_val_auc))
        logger.info(
            'val ap: {:.4f}, new node val ap: {:.4f}'.format(val_ap, nn_val_ap))

        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch, i)
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), get_checkpoint_path(epoch, i))


    if USE_MEMORY:
        val_memory_backup = model.memory.backup_memory()

        ### Test
    model.set_neighbor_finder(full_ngh_finder)
    test_ap, test_auc = eval_edge_prediction(model=model,
                                             negative_edge_sampler=test_rand_sampler,
                                             data=test_data,
                                             n_neighbors=NUM_NEIGHBORS,
                                             negative_sampling_strategy=NEG_SAMPLE_STRATEGY,
                                             eval_idx_data_loader=test_idx_data_loader)

    if USE_MEMORY:
        model.memory.restore_memory(val_memory_backup)

    # Test on unseen nodes
    nn_test_ap, nn_test_auc = eval_edge_prediction(model=model,
                                                   negative_edge_sampler=nn_test_rand_sampler,
                                                   data=new_node_test_data,
                                                   n_neighbors=NUM_NEIGHBORS,
                                                   negative_sampling_strategy=NEG_SAMPLE_STRATEGY,
                                                   eval_idx_data_loader=new_node_test_idx_data_loader)

    logger.info(
        'Test statistics: Old nodes -- auc: {:.4f}, ap: {:.4f}'.format(test_auc, test_ap))
    logger.info(
        'Test statistics: New nodes -- auc: {:.4f}, ap: {:.4f}'.format(nn_test_auc, nn_test_ap))


    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "new_node_test_ap": nn_test_ap,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TAWRMAC model')
    if USE_MEMORY:
        # Restore memory at the end of validation
        model.memory.restore_memory(val_memory_backup)
    torch.save(model.state_dict(), get_model_path(i))
    logger.info('TAWRMAC model saved')
