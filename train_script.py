import time
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd, init
from exam_model import EXAM
from utils import load_data, batch_iter, accuracy


# Input dataset/task parameters
target_task_index = 3  # ie 3 -> ag_news_csv

list_task = [
    "yelp_review_full_csv",
    "amazon_review_polarity_csv",
    "amazon_review_full_csv",
    "ag_news_csv",
    "yahoo_csv",
    "dbpedia_csv",
]
list_max_sequence_length = [1024, 256, 256, 256, 1024, 256]
list_n_classes = [5, 2, 5, 4, 10, 14]
list_vocab_size = [124273, 394385, 356312, 42783, 361926, 227863]
data_path = "data/"
task_path = list_task[target_task_index] + "/"


# Model Hyper-parameters
print_step = 500
emb_size = 128
region_size = 7
region_radius = region_size // 2
batch_size = 16
max_epoch = 5
learning_rate = 0.0001
n_classes = list_n_classes[target_task_index]
vocab_size = list_vocab_size[target_task_index]
max_sequence_length = list_max_sequence_length[target_task_index]


# Model Definition / Initialization
ctx = mx.gpu(0)
model = EXAM(
    num_classes=n_classes,
    vocabulary_size=vocab_size,
    embedding_size=emb_size,
    region_size=region_size,
    max_sequence_length=max_sequence_length,
    batch_size=batch_size,
)
SCE = mx.gluon.loss.SoftmaxCrossEntropyLoss()
model.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(
    model.collect_params(), "adam", {"learning_rate": learning_rate}
)
data_test, data_train = load_data(data_path + task_path, max_sequence_length)
best_acc, global_step, train_loss, train_acc = 0, 0, 0, 0

# Comment net.hybridize() code to get an interactive view/debugging of model layers - code run will slower
model.hybridize()


def batch_preprocess(seq, ctx, embedding_size):
    seq = np.array(seq)
    aligned_seq = np.zeros(
        (max_sequence_length - 2 * region_radius, batch_size, region_size)
    )
    for i in range(region_radius, max_sequence_length - region_radius):
        aligned_seq[i - region_radius] = seq[
            :, i - region_radius : i - region_radius + region_size
        ]
    aligned_seq = nd.array(aligned_seq, ctx)
    batch_sequence = nd.array(seq, ctx)
    trimed_seq = batch_sequence[:, region_radius : max_sequence_length - region_radius]
    mask = nd.broadcast_axes(
        nd.greater(trimed_seq, 0).reshape((batch_size, -1, 1)),
        axis=2,
        size=embedding_size,
    )
    return aligned_seq, nd.array(trimed_seq, ctx), mask


def evaluate(data, batch_size, embedding_size):
    test_loss = 0.0
    acc_test = 0.0
    cnt = 0
    for epoch_percent, batch_slots in batch_iter(data, batch_size, shuffle=False):
        batch_sequence, batch_label = zip(*batch_slots)
        batch_label = nd.array(batch_label, ctx)
        aligned_seq, trimed_seq, mask = batch_preprocess(
            batch_sequence, ctx, embedding_size
        )
        output = model(aligned_seq, trimed_seq, mask)
        loss = SCE(output, batch_label)
        acc_test += accuracy(output, batch_label, batch_size)
        test_loss += nd.mean(loss)
        cnt = cnt + 1
    return acc_test.asscalar() / cnt, test_loss.asscalar() / cnt


# Launching Training loop
ctime = time.time()
print("\nStarting training loop...\n")
for epoch in range(max_epoch):
    epoch_start_time = time.time()
    for epoch_percent, batch_slots in batch_iter(data_train, batch_size, shuffle=True):
        batch_sequence, batch_label = zip(*batch_slots)
        global_step = global_step + 1
        batch_label = nd.array(batch_label, ctx)
        aligned_seq, trimed_seq, mask = batch_preprocess(batch_sequence, ctx, emb_size)
        with autograd.record():
            output = model(aligned_seq, trimed_seq, mask)
            loss = SCE(output, batch_label)
        loss.backward()
        trainer.step(batch_size)
        # train_acc += accuracy(output, batch_label, batch_size)
        train_loss += nd.mean(loss)
        if global_step % print_step == 0:
            print(
                "Epoch %d Progress:" % (epoch + 1),
                "%.4f %% |" % epoch_percent,
                "train_loss:",
                train_loss.asscalar() / print_step,
                "| time:",
                time.time() - ctime,
            )
            train_loss = 0
            ctime = time.time()

    test_acc, test_loss = evaluate(data_test, batch_size, emb_size)
    epoch_time = time.time() - epoch_start_time

    if test_acc > best_acc:
        best_acc = test_acc
        model.save_parameters(f"model_bin/exam_{list_task[target_task_index]}.bin")
    print(
        "Epoch %d Completed" % (epoch + 1),
        "| Test Acc = %.4f | loss = %.4f " % (test_acc, test_loss),
        "| Duration:", epoch_time, "\n",
    )
