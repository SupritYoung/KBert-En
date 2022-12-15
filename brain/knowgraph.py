# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
# import pkuseg
import numpy as np
from uer.utils.tokenizer import *

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, args, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        # 中文分词器
        # self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        # 英文分词器
        # TODO 改用分词
        self.tokenizer = BertTokenizer(args)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                        subj, pred, obje = subj.lower(), pred.lower(), obje.lower()
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + ' ' + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def get(self, tokens):
        '''
        查找最长 token 对应的三元组，返回多个 pred + obje 组成的 list
        '''
        # TODO 要查找所有的
        result = []
        lookup_tokens = ''
        span = 1
        for token_id, token in enumerate(tokens):
            lookup_tokens = token if token_id == 0 else lookup_tokens + ' ' + token
            if lookup_tokens in self.lookup_table.keys():
                result.clear()
                span = len(lookup_tokens.split(' '))
                for r in list(self.lookup_table[lookup_tokens]):
                    result.append(r)
        return result, span


    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        # split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        split_sent_batch = [sent.split(' ') for sent in sent_batch]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            token_id = 0
            split_sent = [x.lower() if x not in self.special_tags else x for x in split_sent]
            for token in split_sent:
                # entities = list(self.get(split_sent[token_id:]))[:max_entities]
                entities, span = self.get(split_sent[token_id:])
                token_id += span
                sent_tree.append((token, entities))

                # if token in self.special_tags:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
                # else:
                #     token_pos_idx.append(pos_idx+1)
                #     token_abs_idx.append(abs_idx+1)
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent = ent.lower()
                    ent = ent.split(' ')
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                # 绝对树上距离 id
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                # 相对树上距离 id
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                # abs_idx_src：句中 token id 组成的 list
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                # 进一步 split
                else:
                    know_sent.append(word)
                    seg += [0]
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    # add_word = list(sent_tree[i][1][j])
                    add_word = sent_tree[i][1][j].split(' ')
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            # pading 操作
            if len(know_sent) < max_length and add_pad:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

