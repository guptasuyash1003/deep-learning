from gensim.models import Word2Vec
import numpy as np
from pprint import *
from collections import OrderedDict
import pickle

def sentence_list_creation(file):
    with open(file,'r',encoding='UTF-8') as pos_file:
        lines = [line for line in pos_file.readlines()]
        # print(lines)
        empty_ind = []
        for ind, new_line_index in enumerate(lines):
            if new_line_index == '\n':
                empty_ind.append(ind)
        # print(empty_ind)

    initial_indx = 0
    sentences_list = []
    for emp_ind in empty_ind:
        sentences_list.append(lines[initial_indx:emp_ind])
        initial_indx = emp_ind + 1
    # print('Original sentence lists:\n%s\n'%sentences_list)
    print(sentences_list)
    return sentences_list

def word_tag_individual_list(file_name):
    sentences_list = sentence_list_creation(file_name)
    words = []
    tags = []
    chunks = []
    for sentence_id in sentences_list:
        word = []
        tag = []
        chunk = []
        for words_with_tag in sentence_id:
            if str(words_with_tag.split(' ')[0]).startswith('\ufeff'):
                word.append(words_with_tag.split(' ')[0].lstrip('\ufeff'))
            else:
                word.append(words_with_tag.split(' ')[0])
            tag.append(words_with_tag.split(' ')[1].rstrip('\n'))
            chunk.append(words_with_tag.split(' ')[2].rstrip('\n'))
        words.append(word)
        tags.append(tag)
        chunks.append(chunk)
    return [words, tags, chunks]

def unique_tags(tag_list):
    tag = list(set([word_tag for sent_tag in tag_list for word_tag in sent_tag]))
    return tag

def unique_chunks(chunk_list):
    chunk = list(set([word_chunk for sent_chunk in chunk_list for word_chunk in sent_chunk]))
    return chunk

def word2Vec_model(words, embidding_size):
    word_vec_model = Word2Vec(sentences=words, size=embidding_size, window=3, workers=5, min_count=1)
    # word_vec_model = Word2Vec(sentences=words, size=300, window=3, workers=5, min_count=1)
    word_vec_model.save('maithili_word2vec')
    # # print('Vocabularies size: ', len(word_vec_model.wv.vocab))
    # word_vec_model = Word2Vec.load('hindi_wordvec_38')
    # print(word_vec_model['??'])
    return word_vec_model

def tag_maping_model(tags, one_hot_size):
    tags.append('NAN')
    one_hot_vector = np.zeros(one_hot_size, dtype=np.int32)
    tag_Hot_mapping = OrderedDict()
    index_tag_list = []


    for t_index, tag in enumerate(tags):
        if tag == 'NAN':
            index_tag_list.append((0, tag))
            tag_Hot_mapping.update({'NAN':one_hot_vector})
        else:
            index_tag_list.append((t_index+1, tag))
            temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
            temp_one_hot_vec[t_index] = 1
            tag_Hot_mapping.update({tag:temp_one_hot_vec})
    index_tag_mapping = OrderedDict(index_tag_list)
    # pprint(index_tag_mapping)
    # pprint(tag_Hot_mapping)
    return [tag_Hot_mapping, index_tag_mapping]

    #     index_tag_list.append((t_index+1, tag))
    #     temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
    #     temp_one_hot_vec[t_index] = 1
    #     tag_Hot_mapping.update({tag:temp_one_hot_vec})
    # index_tag_mapping = OrderedDict(index_tag_list)
    # # pprint(index_tag_mapping)
    # # pprint(tag_Hot_mapping)
    # return [tag_Hot_mapping, index_tag_mapping]

def tag_vector(tag_Hot_mapping, tag):
    return tag_Hot_mapping[tag]

def padding_appender(word_vec, tag_vec, tc):
    size_sent = max([len(size) for size in word_vec])
    for sent in word_vec:
        while(len(sent)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            sent.append(padding)
    for tag in tag_vec:
        while(len(tag)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            tag.append(padding)
    return word_vec, tag_vec




if __name__ == '__main__':
    file_name = '/home/dimple21/PycharmProjects/coding/file_preprocessing_temp.py'
    word_list, tag_list, chunk_list = word_tag_individual_list(file_name)
    print("chunk:\n",chunk_list)

    # unique_tag = unique_tags(tag_list)
    unique_chunk = unique_chunks(chunk_list)
    # total_class = embedding_size = len(unique_tag)+1
    total_class = embedding_size = len(unique_chunk) + 1
    wv_model = word2Vec_model(word_list, embedding_size)
    chunk_Hot_mapping, index_chunk_mapping = tag_maping_model(unique_chunk, total_class)
    print(chunk_Hot_mapping)

    word_one_hot_list = []
    for word_sentence in word_list:
        sentence_word = []
        for word in word_sentence:
            sentence_word.append(wv_model[word])
        word_one_hot_list.append(sentence_word)
    #
    # tag_one_hot_list = []
    # for tag_sentence in tag_list:
    #     sentence_tag = []
    #     for tag in tag_sentence:
    #         v_tag = tag_vector(chunk_Hot_mapping, tag)
    #         sentence_tag.append(v_tag)
    #     tag_one_hot_list.append(sentence_tag)

    chunk_one_hot_list = []
    for chunk_sentence in chunk_list:
        sentence_chunk = []
        for chunk in chunk_sentence:
            v_chunk = tag_vector(chunk_Hot_mapping, chunk)
            sentence_chunk.append(v_chunk)
        chunk_one_hot_list.append(sentence_chunk)
    # pprint(chunk_one_hot_list)
    # pprint(word_one_hot_list)
    word_sent_list, chunk_sent_list = padding_appender(word_one_hot_list, chunk_one_hot_list, total_class)
    pprint(chunk_sent_list)
    print(np.shape(word_sent_list))
    print(np.shape(chunk_sent_list))
    #
    # with open('maithili.pickle','wb') as f:
    #     # pickle.dump([word_one_hot_list, tag_one_hot_list, index_tag_mapping, total_class],f)
    #     pickle.dump([word_sent_list, tag_sent_list, index_tag_mapping, total_class],f, protocol=pickle.HIGHEST_PROTOCOL)
    # print('File created sucessfully: maithili.pickle')
    #
    #from gensim.models import Word2Vec
import numpy as np
from pprint import *
from collections import OrderedDict
import pickle

def sentence_list_creation(file):
    with open(file,'r',encoding='UTF-8') as pos_file:
        lines = [line for line in pos_file.readlines()]
        # print(lines)
        empty_ind = []
        for ind, new_line_index in enumerate(lines):
            if new_line_index == '\n':
                empty_ind.append(ind)
        # print(empty_ind)

    initial_indx = 0
    sentences_list = []
    for emp_ind in empty_ind:
        sentences_list.append(lines[initial_indx:emp_ind])
        initial_indx = emp_ind + 1
    # print('Original sentence lists:\n%s\n'%sentences_list)
    print(sentences_list)
    return sentences_list

def word_tag_individual_list(file_name):
    sentences_list = sentence_list_creation(file_name)
    words = []
    tags = []
    chunks = []
    for sentence_id in sentences_list:
        word = []
        tag = []
        chunk = []
        for words_with_tag in sentence_id:
            if str(words_with_tag.split('\t')[0]).startswith('\ufeff'):
                word.append(words_with_tag.split('\t')[0].lstrip('\ufeff'))
            else:
                word.append(words_with_tag.split('\t')[0])
            tag.append(words_with_tag.split('\t')[1].rstrip('\n'))
            chunk.append(words_with_tag.split('\t')[2].rstrip('\n'))
        words.append(word)
        tags.append(tag)
        chunks.append(chunk)
    return [words, tags, chunks]

def unique_tags(tag_list):
    tag = list(set([word_tag for sent_tag in tag_list for word_tag in sent_tag]))
    return tag

def unique_chunks(chunk_list):
    chunk = list(set([word_chunk for sent_chunk in chunk_list for word_chunk in sent_chunk]))
    return chunk

def word2Vec_model(words, embidding_size):
    word_vec_model = Word2Vec(sentences=words, size=embidding_size, window=3, workers=5, min_count=1)
    # word_vec_model = Word2Vec(sentences=words, size=300, window=3, workers=5, min_count=1)
    word_vec_model.save('maithili_word2vec')
    # # print('Vocabularies size: ', len(word_vec_model.wv.vocab))
    # word_vec_model = Word2Vec.load('hindi_wordvec_38')
    # print(word_vec_model['??'])
    return word_vec_model

def tag_maping_model(tags, one_hot_size):
    tags.append('NAN')
    one_hot_vector = np.zeros(one_hot_size, dtype=np.int32)
    tag_Hot_mapping = OrderedDict()
    index_tag_list = []


    for t_index, tag in enumerate(tags):
        if tag == 'NAN':
            index_tag_list.append((0, tag))
            tag_Hot_mapping.update({'NAN':one_hot_vector})
        else:
            index_tag_list.append((t_index+1, tag))
            temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
            temp_one_hot_vec[t_index] = 1
            tag_Hot_mapping.update({tag:temp_one_hot_vec})
    index_tag_mapping = OrderedDict(index_tag_list)
    # pprint(index_tag_mapping)
    # pprint(tag_Hot_mapping)
    return [tag_Hot_mapping, index_tag_mapping]

    #     index_tag_list.append((t_index+1, tag))
    #     temp_one_hot_vec = np.zeros(one_hot_size, dtype=np.int32)
    #     temp_one_hot_vec[t_index] = 1
    #     tag_Hot_mapping.update({tag:temp_one_hot_vec})
    # index_tag_mapping = OrderedDict(index_tag_list)
    # # pprint(index_tag_mapping)
    # # pprint(tag_Hot_mapping)
    # return [tag_Hot_mapping, index_tag_mapping]

def tag_vector(tag_Hot_mapping, tag):
    return tag_Hot_mapping[tag]

def padding_appender(word_vec, tag_vec, tc):
    size_sent = max([len(size) for size in word_vec])
    for sent in word_vec:
        while(len(sent)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            sent.append(padding)
    for tag in tag_vec:
        while(len(tag)!=size_sent):
            padding = np.zeros(tc, dtype=np.int32)
            tag.append(padding)
    return word_vec, tag_vec




if __name__ == '__main__':
    file_name = '/home/suyash/PycharmProjects/Maithili_tagger/chunking_temp.txt'
    word_list, tag_list, chunk_list = word_tag_individual_list(file_name)
    print("chunk:\n",chunk_list)

    # unique_tag = unique_tags(tag_list)
    unique_chunk = unique_chunks(chunk_list)
    # total_class = embedding_size = len(unique_tag)+1
    total_class = embedding_size = len(unique_chunk) + 1
    wv_model = word2Vec_model(word_list, embedding_size)
    chunk_Hot_mapping, index_chunk_mapping = tag_maping_model(unique_chunk, total_class)
    print(chunk_Hot_mapping)

    word_one_hot_list = []
    for word_sentence in word_list:
        sentence_word = []
        for word in word_sentence:
            sentence_word.append(wv_model[word])
        word_one_hot_list.append(sentence_word)
    #
    # tag_one_hot_list = []
    # for tag_sentence in tag_list:
    #     sentence_tag = []
    #     for tag in tag_sentence:
    #         v_tag = tag_vector(chunk_Hot_mapping, tag)
    #         sentence_tag.append(v_tag)
    #     tag_one_hot_list.append(sentence_tag)

    chunk_one_hot_list = []
    for chunk_sentence in chunk_list:
        sentence_chunk = []
        for chunk in chunk_sentence:
            v_chunk = tag_vector(chunk_Hot_mapping, chunk)
            sentence_chunk.append(v_chunk)
        chunk_one_hot_list.append(sentence_chunk)
    # print(chunk_one_hot_list)
    # print(word_one_hot_list)
    word_sent_list, chunk_sent_list = padding_appender(word_one_hot_list, chunk_one_hot_list, total_class)
    print(chunk_sent_list)
    print(np.shape(word_sent_list))
    print(np.shape(chunk_sent_list))
    #
    with open('maithili.pickle','wb') as f:
        #pickle.dump([word_one_hot_list, tag_one_hot_list, index_tag_mapping, total_class],f)
        pickle.dump([word_sent_list, chunk_sent_list, index_chunk_mapping, total_class],f, protocol=pickle.HIGHEST_PROTOCOL)
    print('File created sucessfully: maithili.pickle')

    