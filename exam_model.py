"""
Title   : Explicit Interaction Model towards Text Classification
Author  : Cunxiao Du, Zhaozheng Chin, Fuli Feng, Lei Zhu, Tian Gan, Liqiang Nie
Papers  : https://arxiv.org/pdf/1811.09386.pdf
Source  : https://github.com/zhaozhengChen/RegionEmbedding
          https://github.com/NonvolatileMemory/AAAI_2019_EXAM
"""
from mxnet.gluon import nn


class EXAM(nn.HybridBlock):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 128,
        region_size: int = 7,
        exam_activation: str = "relu",
        max_sequence_length: int = 256,
        batch_size: int = 32,
    ):
        super(EXAM, self).__init__()

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.region_size = region_size
        self.region_radius = self.region_size // 2
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

        with self.name_scope():
            # Same set of components initialy used in Region Embedding Word Context scenario
            self.embedding = nn.Embedding(
                self.vocabulary_size, self.region_size * self.embedding_size
            )
            self.embedding_region = nn.Embedding(
                self.vocabulary_size, self.embedding_size
            )

            self.max_pool = nn.GlobalMaxPool1D()

            self.dense = nn.Dense(self.num_classes)

            # EXAM adds 2 extra linear layers (dense1/dense2) on top of the default region embedding model_bin
            self.dense1 = nn.Dense(
                self.max_sequence_length * 2, activation=exam_activation
            )
            self.dense2 = nn.Dense(1)

    def hybrid_forward(self, F, aligned_seq, trimed_seq, mask):

        # Region embedding setup
        region_aligned_seq = aligned_seq.transpose((1, 0, 2))
        region_aligned_emb = self.embedding_region(region_aligned_seq).reshape(
            (self.batch_size, -1, self.region_size, self.embedding_size)
        )
        context_unit = self.embedding(trimed_seq).reshape(
            (self.batch_size, -1, self.region_size, self.embedding_size)
        )
        projected_emb = region_aligned_emb * context_unit

        feature = self.max_pool(
            projected_emb.transpose((0, 1, 3, 2)).reshape(
                (self.batch_size, -1, self.region_size)
            )
        ).reshape((self.batch_size, -1, self.embedding_size))
        feature = feature * mask

        # Exam - Feature interaction with classes
        feature = feature.reshape((-1, self.embedding_size))
        feature = (
            self.dense(feature)
            .reshape((self.batch_size, -1, self.num_classes))
            .transpose((0, 2, 1))
            .reshape((self.batch_size * self.num_classes, -1))
        )

        # Exam - Aggregation Layer + MLP modifications
        feature = F.expand_dims(feature, axis=1)
        residual = F.sum(feature, axis=2).reshape((self.batch_size, self.num_classes))
        res = (
            self.dense2(self.dense1(feature))
            .reshape(self.batch_size * self.num_classes, 1, -1)
            .reshape((self.batch_size, self.num_classes))
        )

        output = res + residual
        return output
