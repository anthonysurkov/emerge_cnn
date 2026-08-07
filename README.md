CNN-based architecture for modeling results of the R255X-targeted EMERGe assay.

The EMERGe assay (En Masse Evaluation of RNA Guides) is a library screen
designed to permute a 10-nucleotide (10-nt) region of RNA to test for sequence-
base activity with the RNA-editing enzyme, ADAR.

Next-generation sequencing (NGS) is used to read out individual hairpins, where
the 10-nt variable region acts as a barcode sequence. Hairpins are classified by
the status of their target adenosine, with exactly `k` hairpins carrying a
guanosine in place of the original adenosine, and `n-k` hairpins carrying the
original adenosine. Together, each guide RNA screened by EMERGe returns
binomial data with a per-guide maximum likelihood estimator of `k/n`.

There are a few difficulties regarding data analysis over this assay:
(1) The dataset is heavily zero-inflated, with approximately 50% of sequences
yielding noise-floor zeros. The sequences we are interested in are within
the 99.99th percentile of guides.
This is a heavy class imbalance, which complicates
machine learning efforts due to the relative ease of the model regressing to
zero ("if I guess 0% editing for all guides, I'll be pretty correct!")

(2) The dataset is heavily overdispersed, with a variance exceeding standard
binomial expectations by several hundred times. Thankfully, this is
sequence-invariant.

(3) By contrast, coverage (`n`) is itself dependent on sequence
(heteroskedasticity with confounders) with a k-mer ridge regression estimating
an R2=0.87 between sequence identity and `n`. Moreover, coverage is
anti-aligned with editing, with top editing motifs generally having lower `n`
than non-editing motifs. Here, top editing motifs are references from both
k-mer ridge regression baseline (see: kmer/) and previous thresholded-theta
supervised clustering experiments.
This makes thresholded-theta approaches difficult, as enrichment of a
high-editing subset (for STREME-like analysis or similar) via mixture modeling
is underpowered for exactly the motifs that EMERGe hunts for. While a paired-
guide negative control screen would bypass this confounding variable, said
screen is weeks out from availability.
Therefore, a continuous-theta model is investigated.

A k-mer ridge regression with screened alpha has already been run as a baseline,
with an R2 of approximately 0.55. In keeping with the k-mer-phenotype modeling
space (predominantly regulatory genomics) a CNN-based architecture is proposed.
Though gene regulation and ADAR editing differ in important ways for the
modeling context---namely that the core proposed mechanism is not strictly
sequential, but instead secondary- and tertiary-structural for ADAR editing---
CNNs may still be able to learn complex sequence interactions that are in turn
correlated with the underlying mechanisms.
As well, other assumptions standard to the field, including position invariance,
context length, and sampling strategies, all ask for adapted algorithms.

A review of strategies used in DeepBind, DeepSEA, TF-MoDISco, and DeepLIFT
yields the proposed schema:

(1) Encoding into one-hot features.
    (n x N) sequences -> (d x n) x N one-hot matrices, where n=sequence length,
    N=number of examples, d=dimensionality, usually 4)

(2) Convolutional layers: "standard" conv-ReLU layers, with flattening at the
    end into a vector. Note that pooling (as in DeepBind etc.) is not used
    in this architecture due to position-variance and small search space.
    (d x n) x N -> (n x F), where F is the number of filters used. Kernel
    size k and filter number F will be screened for promising models.

(3) Fully-connected dual-head network: a hurdle model predicts a pi head
    for classification of whether a guide edits at all or not, and predicts a p
    head for regression of how well a guide edits.
    (n x F) -> pi, p

(4) Attribution: all synthesized models are passed through a DeepLIFT- and
    TF-MoDISco-based pipeline to obtain motif position-weight matrices.

A modular hyperparameter and architecture search is chosen. To facilitate,
a central `truth.py` will be made that sections the primary dataset
(`r255x.csv`) into train/validation/test splits (80-10-10), with representative
proportions of exact zeros and nonzero observations. Each tested model and
hyperparameter set will reference the same train/validation split, with the
final model selected by validation being evaluated on the canonical test split
exactly once.

The following Python modules implement the schema:
src/main.py - assembles and runs the baseline model
src/model.py - one-hot encoding, one conv-ReLU layer, and two dense output heads
src/losses.py - zero-inflated beta-binomial loss
src/truth.py - dataset loading and train/validation/test splits
src/training.py - model training and validation
src/eval.py - model loading and evaluation
src/metadata.py - model, training, and data metadata
src/paths.py - data path configuration
src/attr.py - attribution placeholder
src/tests/test_loss.py - loss tests

Loss will be a zero-inflated beta-binomial hurdle model.
