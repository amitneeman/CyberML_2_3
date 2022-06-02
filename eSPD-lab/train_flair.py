from torch.optim.adam import Adam
from flair.data import Corpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair_util import get_corpus
import os

from util import getTrainArgs

args = getTrainArgs()

# 1. get the corpus
dir = './datasets/VTPAN/'
data_indicator = 'VTPAN'
# corpus: Corpus = get_corpus(args.data_dir, args.dataset_indicator)
corpus: Corpus = get_corpus(dir, data_indicator)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()
# TODO do this ourself so it doesn't have to be made
# label_dict = ["non-predator", "predator"]

# 3. initialize transformer document embeddings (many models are available)
# document_embeddings = TransformerDocumentEmbeddings('bert-base-cased', fine_tune = True)
# document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune = True)
document_embeddings = TransformerDocumentEmbeddings(args.model_indicator, fine_tune=True)
# document_embeddings = TransformerDocumentEmbeddings('albert-base-v1', fine_tune = True)

# 4. create the text classifier
classifier = TextClassifier(
    document_embeddings,
    label_dictionary=label_dict,
    # beta=.5
)

# 5. initialize the text classifier trainer with Adam optimizer
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

# 6. start the training
out_dir = os.path.join(args.run_dir, "non_quantized")
trainer.train(out_dir,
              learning_rate=3e-5,
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=3,  # terminate after 3 epochs
              # embeddings_storage_mode='gpu' # i hope this fits in memory

              # this disables selecting the best model depending on performance on the
              # dev/validation set and just selects
              # the final model after three epochs ???
              # the model with the best performance on train ???
              train_with_dev=True,
              monitor_train=True,
              monitor_test=True
              )

# 8. plot weight traces
plotter = Plotter()
plotter.plot_training_curves('%s/loss.tsv' % out_dir)
plotter.plot_weights('%s/weights.txt' % out_dir)
