
import argparse


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """


    if args.data == 'wikipedia':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = False
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
        else:
            args.pick_new_neighbors = True
            args.position_feat_dim = 10
            args.walk_emb_dim = 10
            args.num_walks = 1
            args.max_input_seq_length = 128

    if args.data == 'mooc':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = False
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
        else:
            args.pick_new_neighbors = True

    if args.data == 'reddit':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
        else:
            args.pick_new_neighbors = True

    if args.data == 'lastfm':
        if args.negative_sample_strategy == 'random':
            args.pick_nlew_neighbors = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = False
        else:
            args.pick_new_neighbors = True
            args.max_input_seq_length = 4

    if args.data == 'uci':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 4
        else:
            args.pick_new_neighbors = True
            args.max_input_seq_length = 4

    if args.data == 'enron':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = False
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = False
            args.position_feat_dim = 10
            args.walk_emb_dim = 10
            args.num_walks = 1
            args.max_input_seq_length = 8
        else:
            args.pick_new_neighbors = False
            args.position_feat_dim = 10
            args.walk_emb_dim = 10
            args.num_walks = 1
            args.max_input_seq_length = 8

    if args.data == 'Contacts':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neiglhbors = False
        else:
            args.pick_new_neighbors = False

    if args.data == 'UNtrade':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 200
            args.num_walks = 20
            args.uniform = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 200
            args.num_walks = 20
            args.uniform = True
        else:
            args.pick_new_neighbors = True
            args.max_input_seq_length = 200
            args.num_walks = 20
            args.uniform = True



    if args.data == 'USLegis':
        if args.negative_sample_strategy == 'random':
            # trying to improve
            args.pick_new_neighbors = True
            args.max_input_seq_length = 200
            args.uniform = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = False
        else:
            args.pick_new_neighbors = False

    if args.data == 'CanParl':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 500
            args.num_walks = 30
            args.uniform = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 500
            args.num_walks = 30
            args.uniform = True
        else:
            args.pick_new_neighbors = True
            args.max_input_seq_length = 500
            args.num_walks = 30
            args.uniform = True

    if args.data == 'UNvote':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
            args.max_input_seq_length = 100
            args.num_walks = 20
            args.uniform = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = False
            args.max_input_seq_length = 100
            args.num_walks = 20
            args.uniform = True
        else:
            args.pick_new_neighbors = False
            args.max_input_seq_length = 100
            args.num_walks = 20
            args.uniform = True


    if args.data == 'Flights':
        if args.negative_sample_strategy == 'random':
            args.pick_new_neighbors = True
        elif args.negative_sample_strategy == 'historical':
            args.pick_new_neighbors = False
            args.num_walks = 20
            # args.fixed_time_dim = 50
        else:
            args.pick_new_neighbors = False
            args.num_walks = 20
            # args.fixed_time_dim = 50




