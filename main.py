import argparse
import pprint

from neuralif.geo import main, device


def argparser():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--training_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="random")
    parser.add_argument("--loss", type=str, default="chol")
    parser.add_argument("--gradient_clipping", type=float, required=False)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--regularizer", type=float, default=0)
    
    # Model parameters
    parser.add_argument("--model", type=str, default="neuralpcg")
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--decode_nodes", action='store_true', default=False)
    parser.add_argument("--encoder_layer_norm", action='store_true', default=False)
    parser.add_argument("--mp_layer_norm", action='store_true', default=False)
    parser.add_argument("--aggregate", type=str, default="mean")
    parser.add_argument("--activation", type=str, default="relu")
    
    # NeuralIF parameters
    parser.add_argument("--skip_connections", action='store_true', default=True)
    parser.add_argument("--multi_graph", action='store_true', default=False)
    parser.add_argument("--global_features", type=int, default=0)
    parser.add_argument("--symmetric", type=int, default=0)
    parser.add_argument("--sparse", type=int, default=1)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    
    # initial logging
    print(f"Using device: {device}")
    print("Using config: ")
    pprint.pprint(vars(args))
    print()
    
    # run experiments
    main(vars(args))