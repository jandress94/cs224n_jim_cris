import tensorflow as tf

# model hyperparams
tf.app.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.app.flags.DEFINE_float("learning_decay_rate", 0.8, "How much the learning rate should decay in an epoch.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1000000.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_boolean("use_drop_on_wv", True, "Whether to dropout the word vectors themselves")
tf.app.flags.DEFINE_boolean("init_c_with_q", True, "Whether to feed the question representation in when initially computing the context representation")

# printing / saving
tf.app.flags.DEFINE_string("data_dir", "data/squad/super_small", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "init_vars", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every_num_epochs", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("print_times_per_epoch", 100, "How many times to print the minibatch number per epoch of training.")
tf.app.flags.DEFINE_integer("print_times_per_validate", 10, "How many times to print the validation example number per round of validation.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "./data/squad/glove.trimmed.", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

# preprocessing
tf.app.flags.DEFINE_integer("max_context_len", 200, "The length we will clip/pad a context paragraph to")
# tf.app.flags.DEFINE_integer("question_len", 60, "The length we will clip/pad a question to")
tf.app.flags.DEFINE_integer("padding_token", 0, "The symbol that is inserted in order to pad questions / contexts to the correct length")

def get_flags():
	return tf.app.flags.FLAGS 
