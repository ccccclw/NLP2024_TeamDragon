import argparse

def parse_command_line_arguments():
    """Parse command line arguments for a training and evaluation script."""
    parser = argparse.ArgumentParser(description="Process model training and evaluation parameters.")

    # Required parameters
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints and predictions.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the input training file.")
    parser.add_argument("--dev_file", type=str, required=True,
                        help="Path to the input development file.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the input test file.")

    # Model configuration
    parser.add_argument("--model_type", type=str, default="bert", required=True,
                        help="Type of the model (e.g., bert, roberta).")
    parser.add_argument("--model_checkpoint", type=str, default="bert-large-cased", required=True,
                        help="Pretrained model checkpoint or Hugging Face model identifier.")

    # Training configuration
    parser.add_argument("--max_input_length", type=int, default=256, required=True,
                        help="Maximum length of input sequences.")
    parser.add_argument("--max_target_length", type=int, default=256, required=True,
                        help="Maximum length of target sequences.")
    parser.add_argument("--do_train", action="store_true", help="Flag to enable training.")
    parser.add_argument("--do_test", action="store_true", help="Flag to enable evaluation on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Flag to enable predictions saving.")

    # Optimization parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random number generation.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Beta1 parameter for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.98,
                        help="Beta2 parameter for Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon parameter for Adam optimizer.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training for linear learning rate warmup.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimization.")

    args = parser.parse_args()
    return args

