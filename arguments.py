import argparse, os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=os.getcwd())
    parser.add_argument("--dname", default="cora", type=str)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--model", default="model", type=str)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--train_prop", type=float, default=0.2)
    parser.add_argument("--valid_prop", type=float, default=0.2)

    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--cuda", default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--input_dropout", default=0.2, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument(
        "--num_MLP_layers", default=2, type=int
    )  # How many layers of encoder
    parser.add_argument(
        "--num_classifier_layers", default=2, type=int
    )  # How many layers of decoder
    parser.add_argument("--MLP_hidden", default=256, type=int)  # Encoder hidden units
    parser.add_argument(
        "--classifier_hidden", default=256, type=int
    )  # Decoder hidden units
    parser.add_argument("--display_step", type=int, default=-1)
    parser.add_argument("--display_analysis", action="store_true")
    # Args for SetGNN
    parser.add_argument("--aggregate", default="mean", choices=["mean", "sum"])
    parser.add_argument("--normalization", default="ln")

    parser.add_argument("--add_self_loop", action="store_false")
    parser.add_argument("--exclude_self", action="store_true")
    parser.add_argument("--input_norm", default=True, type=bool)
    parser.add_argument("--PMA", action="store_true")
    parser.add_argument("--GPR", action="store_false")  # skip all but last dec
    parser.add_argument("--LearnMask", action="store_false")
    parser.add_argument("--normtype", default="all_one")  # ['all_one','deg_half_sym']

    # Args for HCHA & HGNN
    parser.add_argument("--HCHA_symdegnorm", action="store_true")
    #     Args for HNHN
    parser.add_argument("--HNHN_alpha", default=-1.5, type=float)
    parser.add_argument("--HNHN_beta", default=-0.5, type=float)
    parser.add_argument("--HNHN_nonlinear_inbetween", default=True, type=bool)
    #     Args for UniGNN
    parser.add_argument(
        "--UniGNN_use-norm", action="store_true", help="use norm in the final layer"
    )
    parser.add_argument("--UniGNN_degV", default=0)
    parser.add_argument("--UniGNN_degE", default=0)
    # Args for HyperGCN
    parser.add_argument("--HyperGCN_fast", default=True, type=bool)
    parser.add_argument("--HyperGCN_mediators", default=True, type=bool)

    # Args for Attention
    parser.add_argument("--heads", default=8, type=int)
    # Args for Line Expansion Based model
    parser.add_argument("--LE_V", default=30, type=int)
    parser.add_argument("--LE_E", default=30, type=int)
    # Args for HWNN
    parser.add_argument("--HWNN_approx", default=False, type=bool)

    # Args for ED-HNN
    parser.add_argument("--edconv_type", default="EquivSet", type=str)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--restart_alpha", default=0.5, type=float)
    parser.add_argument("--MLP1_num_layers", default=2, type=int)
    parser.add_argument("--MLP2_num_layers", default=2, type=int)
    parser.add_argument("--MLP3_num_layers", default=2, type=int)

    # Args for synthetic dataset
    parser.add_argument("--feature_noise", default=0.6, type=float)
    # parser.add_argument('--syn_nodes', default=500, type=int)
    # parser.add_argument('--syn_classes', default=4, type=int)
    # parser.add_argument('--syn_hyperedges', default=1000, type=int)
    # parser.add_argument('--syn_degree', default=15, type=int)
    # parser.add_argument('--syn_alpha', default=5, type=int)

    parser.add_argument("--num_features", default=0, type=int)  # Placeholder
    parser.add_argument("--num_classes", default=0, type=int)  # Placeholder
    parser.add_argument("--num_nodes", default=0, type=int)  # Placeholder
    parser.add_argument("--num_hyperedges", default=0, type=int)  # Placeholder
    parser.add_argument("--num_edges", default=0, type=int)  # Placeholder

    parser.add_argument("--num_conv_layers", default=2, type=int)
    parser.add_argument("--num_MF_layers", default=2, type=int)
    parser.add_argument("--MF_hidden", default=256, type=int)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--attention", action="store_false")

    parser.add_argument("--aug", default="add_edge", type=str)

    # Args for AdE
    parser.add_argument("--backbone", default="GAT", type=str)
    parser.add_argument("--distance_method", default="euclidean", type=str)

    parser.add_argument("--alpha", default=0.1, type=float, help="HP for dc loss")
    parser.add_argument("--gamma", default=0.3, type=float)
    parser.add_argument("--E2VCB", action="store_false")
    parser.add_argument("--V2ECB", action="store_false")
    parser.add_argument("--resnet", action="store_false")
    #
    parser.set_defaults(display_step=-1)
    parser.set_defaults(display_analysis=False)
    parser.set_defaults(debug=True)

    return args
