# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses.ascii import isalnum
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from shutil import copyfile
import distutils
from distutils import util
import sys
import os
import csv
import logging
import argparse
import random
import tempfile
import subprocess
import string
import numpy as np
import torch
from dep2label.labeling import *

sys.path.append(os.path.join(os.path.dirname(__file__), "tree2labels"))

BERT_MODEL="bert_model"
OPENIA_GPT_MODEL="openai_gpt_model"
GPT2_MODEL="gpt2_model"
TRANSFORXL_MODEL="transforxl_model"
XLNET_MODEL="xlnet_model"
XLM_MODEL="xlm_modeL"
DISTILBERT_MODEL="distilbert_model"
ROBERT_MODEL="robert_model"

MODELS = {BERT_MODEL: (BertModel,       BertTokenizer,       'bert-base-uncased'),
          DISTILBERT_MODEL: (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
         }


class MTLBertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config, finetune, use_bilstms=False):
        
        super(MTLBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = len(self.num_labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.use_bilstms=use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True, 
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size*(2 if self.bidirectional_lstm else 1), 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size, 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()

    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        
        hidden_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask,out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:,idtask,:].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs
        

class MTLDistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config, finetune,use_bilstms=False):
        super(MTLDistilBertForTokenClassification, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.num_tasks = len(self.num_labels)
    
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.use_bilstms=use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True, 
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size*(2 if self.bidirectional_lstm else 1), 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size, 
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()



    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        hidden_outputs = self.distilbert(input_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask,out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:,idtask,:].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs


class ExtraConfig(object):

    def __init__(self, num_labels=None, use_bilstms=None, finetune=None):

        self.num_labels = num_labels
        self.use_bilstms = use_bilstms
        self.finetune = finetune

    def write_extra_bert_config(self, path):

        with open(path, "w") as f:
            f.write("labels_per_task=" + ",".join(map(str, self.num_labels)) + "\n")
            f.write("use_bilstms={}".format(self.use_bilstms) + "\n")
            f.write("finetune={}".format(self.finetune))

    def load_extra_bert_config(self, path_extra_config):

        with open(path_extra_config) as f:
            lines = f.readlines()
            for l in lines:

                key, value = l.strip().split("=")

                the_item = "labels_per_task"
                if the_item == key:
                    self.num_labels = list(map(int, value.split(",")))
                the_item = "use_bilstms"
                if the_item == key:
                    self.use_bilstms = distutils.util.strtobool(value)
                the_item = "finetune"
                if the_item == key:
                    self.finetune = distutils.util.strtobool(value)
        
class InputSLExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, 
                 text_a_list,
                 text_a_postags, labels=None, num_tasks=1):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the sentence
            label: (Optional) list. The labels for each token. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.text_a_list = text_a_list 
        self.text_a_postags = text_a_postags
        self.labels = labels
        self.num_tasks = num_tasks



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, position_ids, segment_ids, labels_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.position_ids = position_ids
        self.segment_ids = segment_ids
        self.labels_ids = labels_ids



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SLProcessor(DataProcessor):
    """Processor for PTB formatted as sequence labeling seq_lu file"""
    
    def __init__(self,label_split_char):
        self.label_split_char=label_split_char

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")


    def get_labels(self, data_dir):
        """See base class."""
        
        train_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        dev_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        
        train_labels = [sample.labels for sample in train_samples] 
        dev_labels = [sample.labels for sample in dev_samples]
        labels = []
        
        if self.label_split_char is None:
            tasks_range = 1
        else:
            tasks_range = len(train_labels[0])
        
        for idtask in range(tasks_range): #range(tasks):
            labels.append([])
            labels[idtask].append("[MASK_LABEL]")
            labels[idtask].append("-EOS-")
            labels[idtask].append("-BOS-")
        train_labels.extend(dev_labels)

        for label in train_labels:
            for id_label_component, sent_label_component in enumerate(label):
                for word_label in sent_label_component:
                    if word_label not in labels[id_label_component]:
                        labels[id_label_component].append(word_label)     

        return labels

    def _preprocess(self, word):
        
        ori_word = word
        
        if word == "-LRB-": 
            word = "("
        elif word == "-RRB-": 
            word = ")"
        elif "`" in word:
            word = word.replace("`", "'")
        elif "’’" in word:
            word = word.replace("’’", "'")
        elif "’" in word:
            word =word.replace("’","'")
        elif " " in word:
            word = word.replace(" ","")
        elif "\u200b" == word:
            word = word.replace("\u200b",".")
        elif "\xad" in word:
            word = word.replace("\xad", "") 
        elif "#" == word:
            word = "-"
        elif "#" in word and len(word.replace("#",""))!=0:
            word = word.replace("#","")
            
        elif "-" in word and word !="-" and len(word.replace("-",""))!=0:
            word = word.replace("-","")
            
        elif "–" in word and word !="–" and len(word.replace("–",""))!=0:
            word = word.replace("–","")
    
        if word == "":
            raise ValueError("Generating an empty word for", ori_word)
            # s.translate(None, string.punctuation)         
        return word
    

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        
        examples = []
        sentences_texts = []
        sentences_postags = []
        sentences_labels = []
        sentences_tokens = []
        sentence, sentence_postags, sentence_labels = [],[], []
        tokens = []
        
        for l in lines:
            if l != []:
                
                if l[0] in ["-EOS-","-BOS-"]:
                    tokens.append(l[0])
                    sentence_postags.append(l[-2]) 
                else:
                    tokens.append(l[0])
                    sentence.append(self._preprocess(l[0]))

                    if self.label_split_char != None:
                        values = l[-1].strip().split(self.label_split_char)
                    else:
                        values = [l[-1].strip()]

                    for idtask, value in enumerate(values):
                        try:
                            sentence_labels[idtask].append(value)                        
                        except IndexError:
                            sentence_labels.append([value])                        
                    sentence_postags.append(l[-2]) 
            else:
                
                sentences_texts.append(" ".join(sentence))
                sentences_labels.append(sentence_labels)
                sentences_postags.append(sentence_postags)
                sentences_tokens.append(tokens)
                sentence, sentence_postags, sentence_labels = [], [] ,[[] for idtask in sentence_labels]
                tokens = []

        for guid, (sent, labels) in enumerate(zip(sentences_texts, sentences_labels)):
 
            examples.append(
                InputSLExample(guid=guid, text_a=sent,
                               text_a_list=sentences_tokens[guid],
                               text_a_postags=sentences_postags[guid], 
                               labels=labels))
          
        return examples

def _word_is_ahead(word,prev_word,sub_wp_sent):
      
    for wp in sub_wp_sent:
        if wp != "[UNK]" and wp not in prev_word:
             break
       
    return word.startswith(wp) and not wp.startswith("##")
def _valid_wordpiece_indexes(sent, wp_sent): 
      
    valid_idxs = []
    chars_to_process = ""
    idx = 0
      
    wp_idx = 0
    case = -1
    
    dict_alignments = {}
    for idword, word in enumerate(sent):
        
        chars_to_process = word

        '''
        (0) The word fully matches the word piece when no index has been assigned yet, easy case.
        '''
        if word == wp_sent[wp_idx]: 
            dict_alignments[(idword,word)] = (wp_idx, wp_sent[wp_idx])
            valid_idxs.append(wp_idx)
            wp_idx += 1
            
        else:
            while chars_to_process != "":
                
                try:
                    if word.startswith(wp_sent[wp_idx]) and chars_to_process == word:
                        '''
                        (1) The wordpiece wp_sent[wp_idx] is the prefix of the original word, i.e. first word piece, 
                         we assign its index to the word
                        '''
                        case = 1
                        chars_to_process = chars_to_process[len(wp_sent[wp_idx]):]
                        dict_alignments[(idword,word)] = (wp_idx, wp_sent[wp_idx])
                        valid_idxs.append(wp_idx)
                        wp_idx += 1
                        continue
                    
                    elif not wp_sent[wp_idx].startswith("##") and chars_to_process.startswith(wp_sent[wp_idx]): 
                        '''
                        (2) To control errors in BERT tokenizer at word level. For example a token
                        that is split into to actual tokens and not two or more wordpieces
                        '''
                        case = 2
                        chars_to_process = chars_to_process[len(wp_sent[wp_idx]):]
                        wp_idx += 1           
                        continue
                
                    elif wp_sent[wp_idx].startswith("##"): 
                        
                        '''
                        (3) It is a wordpiece of the form ##[piece]. If this happens,
                        we skip word pieces until a new word is read because in this scenario
                        the original word (word) in the sentence has been assigned a wp_index already, according to (1)
                        '''
                        case = 3
                        while wp_sent[wp_idx].startswith("##"):
                            
                            chars_to_process = chars_to_process[len(wp_sent[wp_idx][2:]):]
                            wp_idx += 1
                        continue
                    
                    elif wp_sent[wp_idx] == "[UNK]":
                        '''  
                        (4) The word could not be tokenized and the BERT tokenizer  generated an [UNK]
                        This can be a problematic case: sometime an original token is split on two, and then each of those
                        generate two consecutive [UNK] symbols. This complicates a lot the alignment between words and word pieces
                        '''
                        
                        case = 4
                        
                        '''
                        We found an [UNK] when the current word still has not been assigned a wp_idx,
                        we consider that [UNK] index must be aligned with word
                        '''
                        if chars_to_process == word: 
                            
                            if _word_is_ahead(word, sent[idword-1], wp_sent[wp_idx:]):
                                wp_idx+=1
                            else:
                                chars_to_process = ""
                                dict_alignments[(idword,word)] = (wp_idx, wp_sent[wp_idx])
                                valid_idxs.append(wp_idx)
                                wp_idx += 1
    
                        else:
                            '''
                            We found an UNK, but the current word has been already assigned an wp_idx. However, 
                            there still missing chars to process from that word (for example if it was generated according to (1))
                            but we know this [UNK] should be a ##wordpiece. To correct this problem, we skip word pieces
                            until a word pieces matches the next word to assign an index to, to get back the alignment to valid
                            scenario.
                            '''                    
                            chars_to_process = ""
                            while idword + 1 < len(sent) and not sent[idword + 1].startswith(wp_sent[wp_idx]):
                                wp_idx += 1
                        
                        continue    
    
                    elif not word.startswith(wp_sent[wp_idx]) and chars_to_process == word:
                        '''
                        Some kind of unpredictable tokenization mismatching between the input samples and BERT
                        caused a mismatch in the alignment. We try to move forward to get the alignment back to
                        a valid position, iff the word has still not received any index
                        '''    
                        wp_idx += 1      
                    elif chars_to_process != word:
                        '''
                        otherwise we just move to the next word
                        ''' 
                        break
                    else:
                        raise RuntimeError("Potential infinite loop caused by the sentence" + 
                                           "Sentence: {}\n".format(list(enumerate(sent))) + 
                                           "Word piece sentence: {}\n".format(list(enumerate(wp_sent))) + 
                                           "Selected indexes: {}\n".format(list(enumerate(valid_indexes)))
                                           
                                           )
                except IndexError:
                    raise IndexError("""
An error occurred. It could be because:
(i) not all word pieces have been generated for the sentence - try to increase max_seq_length.
ii) there has been a problem during the alignment that is causing that the last nth words are not being aligned with a word piece.
                    
The sentence that is causing the problem is
sent = {}
                    
wp_sent = {}
current_alignment = {}
""".format(list(enumerate(sent)),list(enumerate(wp_sent)), dict_alignments ))
    try:
        assert(len(sent) == len(valid_idxs)) 
    except AssertionError:
        
        raise AssertionError("""
            An error ocurred. The number of selected word piece indexes does not match the length of the sentence.
            
            This is due to an unexpected problem during the alignment where not all words have been aligned.
            
            The sentence that is causing the problem is
            sent = {}
                                
            wp_sent = {}
            
            current_alignment = {}
            """.format(list(enumerate(sent)),list(enumerate(wp_sent)), dict_alignments))
        
    return valid_idxs 

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{l: i for i, l in enumerate(component_label)} for j, component_label in enumerate(label_list)]
    label_map_reverse = [{i: l for i, l in enumerate(component_label)} for j, component_label in enumerate(label_list)]
    num_tasks = len(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):

        ori_tokens_a = example.text_a.split(" ")
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        #         print ("ex_index", ex_index)
        #         print ("example", example)
        #         print ("ori_tokens_a", ori_tokens_a, len(ori_tokens_a))
        #         print ("tokens_a", tokens_a, len(tokens_a))
        #         input("NEXT")
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        ori_tokens_a = ["[CLS]"] + ori_tokens_a + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(max_seq_length))
        valid_indexes = _valid_wordpiece_indexes(ori_tokens_a, tokens)

        input_mask = [1 if idtoken in valid_indexes else 0
                      for idtoken, _ in enumerate(tokens)]

        labels_ids = [[] for i in range(num_tasks)]
        i = 0
        for idtoken, token in enumerate(tokens):

            for idtask in range(num_tasks):
                if idtoken in valid_indexes:

                    if token == "[CLS]":
                        labels_ids[idtask].append(label_map[idtask]["-BOS-"])
                    elif token == "[SEP]":
                        labels_ids[idtask].append(label_map[idtask]["-EOS-"])
                    else:
                        try:
                            label_mapped = label_map[idtask][example.labels[idtask][i]]
                            labels_ids[idtask].append(label_mapped)
                        except KeyError:
                            labels_ids[idtask].append(0)

                        if idtask == num_tasks - 1:
                            i += 1
                else:
                    try:
                        labels_ids[idtask].append(
                            label_map[idtask][example.labels[idtask][min(i, len(example.labels[idtask]) - 1)]])
                    except KeyError:
                        labels_ids[idtask].append(0)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        labels_ids = [lids + padding for lids in labels_ids]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        for l in labels_ids:
            assert len(l) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          position_ids=position_ids,
                          segment_ids=segment_ids,
                          labels_ids=labels_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels, mask):

    output = out*mask
    gold = labels*mask
    mask = list()
    o_flat = list(output.flatten())
    g_flat = list(gold.flatten())


    o_filtered, g_filtered = [], []
    
    for o,g in zip(o_flat,g_flat):
        if g !=0:
            g_filtered.append(g)
            o_filtered.append(o)

    assert len(o_filtered), len(g_filtered)
    return accuracy_score(o_filtered, g_filtered)


def evaluate(model, device, logger, processor, tokenizer, label_list, args):
    labeling = Labeler()
    processing = CoNLLPostProcessor()
    if args.do_test:
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_position_ids, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    label_map_reverse = {i: l for i, l in enumerate(label_list)}

    examples_texts = [example.text_a_list for example in eval_examples]
    examples_postags = [example.text_a_postags for example in eval_examples]
    # examples_preds = []
    examples_preds = [[] for i in range(len(label_list))]
    model.eval()

    eval_loss, eval_accuracy = [0] * len(label_list), [0] * len(label_list)
    # eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, position_ids, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        position_ids = position_ids.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            #  position_ids=position_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask)  # , input_mask)

            for idtask, task_output in enumerate(outputs):
                logits_task = task_output[0]
                logits_task = logits_task.detach().cpu().numpy()
                task_label_ids = label_ids[:, idtask, :].to('cpu').numpy()
                masks = input_mask.cpu().numpy()
                outputs = np.argmax(logits_task, axis=2)

                for prediction, mask in zip(outputs, masks):
                    examples_preds[idtask].append(
                        [label_map_reverse[idtask][element] for element, m in zip(prediction, mask)
                         if m != 0])

                for idx_out, (o, l) in enumerate(zip(outputs, task_label_ids)):
                    eval_accuracy[idtask] += accuracy(o, l, masks[idx_out])

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    # Join the preds into one single label
    new_examples_preds = []
    for idsample in range(len(examples_texts)):
        for idtask, component in enumerate(examples_preds):
            if idtask == 0:
                new_examples_preds.append(component[idsample])
            else:
                new_examples_preds[-1] = [c + "{}" + n for c, n in zip(new_examples_preds[-1], component[idsample])]

    output_file_name = args.output_dir + ".dev.outputs.txt.seq_lu" if not args.do_test else args.output_dir + ".test.outputs.txt.seq_lu"
    with open(output_file_name, "w") as temp_out:
        content = []
        for tokens, postags, preds in zip(examples_texts, examples_postags, new_examples_preds):
            content.append("\n".join(["\t".join(element) for element in zip(tokens, postags, preds)]))
        temp_out.write("\n\n".join(content))
        temp_out.write("\n\n")
    eval_accuracy = [e / nb_eval_examples for e in eval_accuracy]
    eval_accuracy = sum(eval_accuracy) / len(eval_accuracy)

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    score = eval_accuracy
    out = eval_accuracy
    output_eval_file = os.path.join(args.output_dir.rsplit("/", 1)[0], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    #lookup = process.dump_into_lookup(args.path_predicted_conllu)
    #output = open(output_file_name)
    tmp = tempfile.NamedTemporaryFile().name

    labeling.decode(output_file_name, tmp, args.encoding, args.path_gold_conllu)
    #d.decode(output,tmp, args.encoding)
    #process.merge_lookup(tmp, lookup)
    current_score = processing.evaluate_dependencies(
                    args.path_gold_conllu, tmp)
    #current_score = d.evaluate_dependencies(
    #    args.path_gold_conllu, tmp)
    score=current_score
    output_file_name=tmp
    print("******************")

    return eval_loss, eval_accuracy, score, out, output_file_name


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--transformer_model",
                        help="Specify the type of the transformer model from this list: 'bert-model', 'openai_gpt_model',"
                        "'gpt2_model','ctrl_model','transforxl_model','xlnet_model','xlm_model','distilbert_model','robert_model',"
                        "xlmroberta_model")
    
    parser.add_argument("--transformer_pretrained_model",
                        help="Specify a pretrained model to fine-tune. For example: 'bert-base-cased','bert-large-cased',"
                        "'bert-base-multilingual-cased','bert-base-german-cased','distilbert-base-cased','distilbert-base-german-cased'"
                        "Check the full list at: https://github.com/huggingface/transformers",
                        required=True)
    
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory model will be written.")
    
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                      #  required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--status",
                        type=str,
                        help="[train|test]")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action="store_true",
                        help="Whether to run eval on the test set")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument("--not_finetune", dest="not_finetune", default=False, action="store_true",
                        help="Determine where to finetune BERT (flag to True) or just the output layer (flag set to False)")
    
    parser.add_argument("--use_bilstms",
                        default=False,
                        action="store_true",
                        help="Further contextualized BERT outputs with BILSTMs")
    
    parser.add_argument("--encoding",
                        type=str,
                        help="encoding type")
    
    #Specific args to evaluate dependency parsing
    parser.add_argument("--conll_ud",
                        type=str,
                        help="Path to the CONLL UD format")

    parser.add_argument("--path_gold_conllu",
                        type=str,
                        help="Path to the gold file in conllu formart")

    parser.add_argument("--path_predicted_conllu",
                        type=str,
                        help="Path to the gold file in conllu formart")

    parser.add_argument("--label_split_char",
                        type=str,
                        help="Character used in labels to split their components",
                        default=None)
    
    parser.add_argument("--log")

    parser.add_argument("--cache_dir", default=PYTORCH_PRETRAINED_BERT_CACHE)

    args = parser.parse_args()

    processors = {"sl_tsv": SLProcessor(args.label_split_char)}

    if args.log is not None:
        f_log = open(args.log,"w")


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]#()

    num_labels = [len(task_labels) for task_labels in processor.get_labels(args.data_dir)]
    label_list = processor.get_labels(args.data_dir)
    label_reverse_map = [{i:label for i,label in enumerate(labels)}
                         for labels in label_list]
    num_tasks = len(label_list)

    if args.transformer_model in MODELS:
        model_class,tokenizer_class, pretrained_model = MODELS[args.transformer_model]
    else:
        raise KeyError("The transformer model ({}) does not exist".format(args.transformer_model))

    tokenizer = tokenizer_class.from_pretrained(args.transformer_pretrained_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    extra_config = ExtraConfig(num_labels, args.use_bilstms, not args.not_finetune)
    if args.status=="train":
        extra_config.write_extra_bert_config(os.path.join(args.model_dir) + ".extra_config")

    if args.transformer_model == BERT_MODEL:

        model = MTLBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                      # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                       num_labels=num_labels,
                                                       finetune=not args.not_finetune,
                                                       use_bilstms=args.use_bilstms)
    elif args.transformer_model == DISTILBERT_MODEL:

        model = MTLDistilBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                          # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                           num_labels=num_labels,
                                                           finetune=not args.not_finetune,
                                                           use_bilstms=args.use_bilstms)

    else: raise NotImplementedError("The selected transformer is not available for parsing as token classification")

   # print (model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
       # optimizer = BertAdam(optimizer_grouped_parameters,
       #                      lr=args.learning_rate,
       #                      warmup=args.warmup_proportion,
       #                      t_total=num_train_optimization_steps)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.labels_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_position_ids, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_dev_evalb = -sys.maxsize -1
        best_dev_evalb_disco = -sys.maxsize -1
        last_epoch= ""
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, position_ids, segment_ids, label_ids = batch

                outputs =model(input_ids=input_ids,
                            #   position_ids=position_ids,
                               token_type_ids = segment_ids,
                               attention_mask = input_mask,
                               labels=label_ids)

                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss = loss.mean()
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1


            dev_loss, dev_acc, dev_eval_score, _, path_output_file = evaluate(model, device, logger, processor,tokenizer, label_list,args)


            f_log.write("\t".join(["Epoch", str(epoch), "F-score", str(dev_eval_score)])+"\n")
            f_log.flush()
            print("Epoch: "+str(epoch))
            print("so far the best "+repr(best_dev_evalb))
            if best_dev_evalb < dev_eval_score:
                print("New best "+repr(dev_eval_score))
                last_epoch= str(epoch)
                best_dev_evalb = dev_eval_score

                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.model_dir)
              #  output_model_file = os.path.join(args.model_dir, "pytorch_model.bin")

                if args.do_train:
                    print ("Saving the best new model...")
                    torch.save(model_to_save.state_dict(), output_model_file)
            print("Last best epoch "+last_epoch)
            print("*****************************")
            model.train() #If not, following error: cudnn RNN backward can only be called in training mode

    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.model_dir)
    model_state_dict = torch.load(output_model_file)


    extra_config = ExtraConfig()
    extra_config.load_extra_bert_config(os.path.join(args.model_dir) + ".extra_config")


    if args.transformer_model == BERT_MODEL:
        model = MTLBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                              state_dict=model_state_dict,
                                                              cache_dir=args.cache_dir,
                                                              num_labels=extra_config.num_labels,
                                                              finetune=extra_config.finetune,
                                                              use_bilstms=extra_config.use_bilstms
                                                              )
    elif args.transformer_model == DISTILBERT_MODEL:
        model = MTLDistilBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                                    state_dict=model_state_dict,
                                                                    cache_dir=args.cache_dir,
                                                                    # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                                    num_labels=extra_config.num_labels,
                                                                    finetune=extra_config.finetune,
                                                                    use_bilstms=extra_config.use_bilstms
                                                                    )

    model.to(device)

    if (args.do_eval or args.do_test) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        loss, acc, eval_score, detailed_score, path_output_file = evaluate(model, device, logger, processor,
                                                                           tokenizer, label_list,
                                                                           args)


if __name__ == "__main__":
    main()
