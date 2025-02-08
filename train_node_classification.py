import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm
import math
import logging
import time
import sys
import argparse
import pickle
from pathlib import Path
import datetime
import torch
from utils.DataLoader import get_idx_data_loader
from model.tawrmac import TAWRMAC
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP, set_random_seed
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

### Argument and global variables
parser = argparse.ArgumentParser('TAWRMAC node classification')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=172, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=172, help='Dimensions of the time embedding')

parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

parser.add_argument('--enable_walk', action='store_true',
                    help='Whether to use causal anonymous walks')
parser.add_argument('--enable_restart', action='store_true',
                    help='Whether to use walk restart')
parser.add_argument('--pick_new_neighbors', action='store_true',
                    help='Whether to pick new neighbors in restart')
parser.add_argument('--walk_emb_dim', type=int, default=172, help='Dimension of TAWR embedding.')
parser.add_argument('--position_feat_dim', type=int, default=100, help='Dimensions of the walk embedding')
parser.add_argument('--walk_length', type=int, default=4, help='Length of walks')
parser.add_argument('--num_walk_heads', type=int, default=4, help='Number of walk heads')
parser.add_argument('--num_walks', type=int, default=10, help='Number of walk (Default was set to 10)')
parser.add_argument('--fixed_time_dim', type=int, default=20, help='Fixed time dimension')
parser.add_argument('--max_input_seq_length', type=int, default=32, help='Max input sequence length for co-occurrence')
parser.add_argument('--enable_neighbor_cooc', action='store_true',
                    help='Whether to enable neighbor co-occurrence encoding')
parser.add_argument('-model_date', type=str, help='Timestamp of the saved model',
                    default='20241206102940')
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
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./node_cl/saved_models/").mkdir(parents=True, exist_ok=True)
Path(f"./node_cl/saved_checkpoints/").mkdir(parents=True, exist_ok=True)

currdate = str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))

get_checkpoint_path = lambda \
        epoch, n_run: f'./node_cl/saved_checkpoints/{args.data}-run-{n_run}-ep-{epoch}-{currdate}-node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("node_cl_log/").mkdir(parents=True, exist_ok=True)
negative_sample_strategy = 'random'
experiment_info = f'{args.data}_{negative_sample_strategy}'

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

get_model_path = lambda n_run: f'./node_cl/saved_models/{experiment_info}_run_1_{args.model_date}.pth'
get_node_cl_model_path = lambda n_run: f'./node_cl/saved_models/{experiment_info}_run_{n_run}_{currdate}_node_cl.pth'

fh = logging.FileHandler(f'node_cl_log/{experiment_info}_{currdate}.log')
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

full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

full_ngh_finder = get_neighbor_finder(full_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
# device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.sources))),
                                            batch_size=BATCH_SIZE, shuffle=False)
val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.sources))),
                                          batch_size=BATCH_SIZE, shuffle=False)
test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.sources))),
                                           batch_size=BATCH_SIZE, shuffle=False)
for i in range(0, args.n_runs):
    set_random_seed(i)
    results_path = f"results/{args.data}_{negative_sample_strategy}_run_{i}_{currdate}.pkl"
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    model = TAWRMAC(neighbor_finder=full_ngh_finder, node_features=node_features,
                    edge_features=edge_features, device=device,
                    n_layers=NUM_LAYER,
                    n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                    memory_dimension=MEMORY_DIM,
                    memory_update_at_start=not args.memory_update_at_end,
                    n_neighbors=NUM_NEIGHBORS,
                    enable_walk=args.enable_walk,
                    enable_restart=args.enable_restart,
                    pick_new_neighbors=args.pick_new_neighbors,
                    enable_neighbor_cooc=args.enable_neighbor_cooc,
                    walk_emb_dim=args.walk_emb_dim,
                    time_dim=TIME_DIM,
                    fixed_time_dim=args.fixed_time_dim,
                    max_input_seq_length=args.max_input_seq_length,
                    num_walks=args.num_walks,
                    position_feat_dim=args.position_feat_dim,
                    walk_length=args.walk_length,
                    num_walk_heads=args.num_walk_heads)

    model = model.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.debug('Num of training instances: {}'.format(num_instance))
    logger.debug('Num of batches per epoch: {}'.format(num_batch))

    logger.info('Loading saved TAWRMAC model')
    model_path = get_model_path(i)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info('TAWRMAC models loaded')
    logger.info('Start training node classification task')

    decoder = MLP(model.get_node_embedding_dim(), drop=DROP_OUT)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    decoder = decoder.to(device)
    decoder_loss_criterion = torch.nn.BCELoss()

    val_aucs = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(args.n_epoch):
        start_epoch = time.time()

        # Initialize memory of the model at each epoch
        if USE_MEMORY:
            model.memory.__init_memory__()

        model = model.eval()
        decoder = decoder.train()
        loss = 0
        logger.info('start {} epoch'.format(epoch))

        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            sources_batch = train_data.sources[train_data_indices]
            destinations_batch = train_data.destinations[train_data_indices]
            timestamps_batch = train_data.timestamps[train_data_indices]
            edge_idxs_batch = full_data.edge_idxs[train_data_indices]
            labels_batch = train_data.labels[train_data_indices]

            size = len(sources_batch)

            decoder_optimizer.zero_grad()
            with torch.no_grad():
                source_embedding, destination_embedding, _, _, _ = model.compute_temporal_embeddings(sources_batch,
                                                                                                     destinations_batch,
                                                                                                     None,
                                                                                                     None,
                                                                                                     timestamps_batch,
                                                                                                     edge_idxs_batch,
                                                                                                     NUM_NEIGHBORS)

            labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
            pred = decoder(source_embedding).sigmoid()
            decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
            decoder_loss.backward()
            decoder_optimizer.step()
            loss += decoder_loss.item()
        train_losses.append(loss / num_batch)

        val_auc = eval_node_classification(model, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                           n_neighbors=NUM_NEIGHBORS, eval_idx_data_loader=val_idx_data_loader)

        val_aucs.append(val_auc)

        pickle.dump({
            "val_aps": val_aucs,
            "train_losses": train_losses,
            "epoch_times": [0.0],
            "new_nodes_val_aps": [],
        }, open(results_path, "wb"))

        logger.info(
            f'Epoch {epoch}: train loss: {loss / num_batch}, val auc: {val_auc}, time: {time.time() - start_epoch}')

        if args.use_validation:
            if early_stopper.early_stop_check(val_auc) or epoch == (NUM_EPOCH - 1):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch, i)
                decoder.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                decoder.eval()

                test_auc = eval_node_classification(model, decoder, test_data, full_data.edge_idxs, BATCH_SIZE,
                                                    n_neighbors=NUM_NEIGHBORS,
                                                    eval_idx_data_loader=test_idx_data_loader)
                break
            else:
                torch.save(decoder.state_dict(), get_checkpoint_path(epoch, i))
        else:
            # If we are not using a validation set, the test performance is just the performance computed
            # in the last epoch
            test_auc = val_aucs[-1]

    pickle.dump({
        "val_aps": val_aucs,
        "test_ap": test_auc,
        "train_losses": train_losses,
        "epoch_times": [0.0],
        "new_nodes_val_aps": [],
        "new_node_test_ap": 0,
    }, open(results_path, "wb"))

    logger.info(f'test auc: {test_auc}')
    logger.info('Saving model')
    torch.save(decoder.state_dict(), get_node_cl_model_path(i))
    logger.info('Node Classifier model saved')
