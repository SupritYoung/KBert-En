# -*- encoding:utf -*-
"""
  This script provides the acquisition method K-BERT Representation.
"""
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
from evaluation import *
import numpy as np
import pickle


class BertRepresentation(nn.Module):
    def __init__(self, model):
        super(BertRepresentation, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder

    def forward(self, src, mask, pooling="mean", pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            mask: [batch_size x seq_length]
            pooling: selected pooling method
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        # if not self.use_vm: # 默认使用 vm
        #     vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if pooling == "mean":
            output = torch.mean(output, dim=1)
        elif pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif pooling == "last":
            output = output[:, -1, :]
        elif pooling == "all":
            output = output[:, :, :]
        else:
            output = output[:, 0, :]
        return output


def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--query_path", type=str, required=False, default="datasets/query_doc_sim/querys.tsv",
                        help="Path of the querys.")
    parser.add_argument("--doc_path", type=str, required=False, default="datasets/query_doc_sim/docs.tsv",
                        help="Path of the documents.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=False, default="brain/kgs/jasist_trans.spo", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    return args


def add_knowledge_worker(params):
    '''
    将句子融合知识图谱相关知识
    '''
    # ids 和 sentences 为 (query id, query) 或 (doc id, doc)
    p_id, sentences, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    for line_id, sentence in enumerate(sentences):
        if line_id % 100 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()

        sentence = CLS_TOKEN + sentence
        tokens, pos, vm, _ = kg.add_knowledge_with_vm([sentence], add_pad=True, max_length=args.seq_length)
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0].astype("bool")

        token_ids = [vocab.get(t) for t in tokens]
        mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

        dataset.append((token_ids, mask, pos, vm))

    return dataset


def main():
    args = init_args()

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    model = BertRepresentation(model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    # Blend kg in the dataset
    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        ids = []  # doc 或 query 的 id
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.strip().split("\t")
                id, sentence = line[0], line[1]
                sentences.append(sentence)
                ids.append(id)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append(
                    (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset, ids

    # query_dataset = read_dataset(args.query_path, workers_num=args.workers_num)
    # doc_dataset = read_dataset(args.doc_path, workers_num=args.workers_num)

    for dataset_path in [args.query_path, args.doc_path]:
        dataset, ids = read_dataset(dataset_path, workers_num=args.workers_num)

        print("Trans data to tensor.")
        print("input_ids")
        input_ids = torch.LongTensor([example[0] for example in dataset])
        print("mask_ids")
        mask_ids = torch.LongTensor([example[1] for example in dataset])
        print("pos_ids")
        pos_ids = torch.LongTensor([example[2] for example in dataset])
        print("vms")
        vms = [example[3] for example in dataset]

        # doc_repr_dict = {}
        index = 0
        length = len(input_ids)
        if "querys" in dataset_path:
            save_path = "datasets/query_doc_sim/query_reprs.txt"
        else:
            save_path = "datasets/query_doc_sim/doc_reprs.txt"

        for id, input_id, mask_id, pos_id, vm in zip(ids, input_ids, mask_ids, pos_ids, vms):
            input_id, mask_id, pos_id, vm = torch.LongTensor(input_id), torch.LongTensor(mask_id), torch.LongTensor(
                pos_id), torch.LongTensor(vm),
            # 计算输出表征
            model_output = model(input_id, mask_id, pos=pos_id, vm=vm)

            print("{} representation completed: {}/{}".format(id, index, length))
            index += 1
            # doc_repr_dict[id] = model_output
            # 保存
            with open(save_path, 'a+') as f:
                f.write(str(id) + '\t' + str(model_output.detach().numpy()) + '\n')

        # 将表征存为字典保存
        # if 'doc' in dataset_path:
        #     repr_file = open(, 'wb')
        # elif 'query' in dataset_path:
        #     repr_file = open('datasets/query_doc_sim/query_reprs.txt', 'wb')

        # pickle.dump(doc_repr_dict, repr_file)
        # repr_file.close()

if __name__ == "__main__":
    main()
