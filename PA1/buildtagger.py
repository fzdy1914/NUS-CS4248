# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import json
import re


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    f = open(train_file, 'r')
    lines = f.readlines()
    f.close()

    emission_prob = dict()
    transition_prob = dict()
    all_words = list()
    word_tag_list = dict()
    word_occurrence = dict()
    predefined_unk_words = set()
    suffix_list = ["able", "acy", "age", "al", "ance", "ant", "ary", "ate", "dom", "ed", "ee", "en", "ence", "ent",
                   "ent", "er", "est", "ful", "fy", "hood", "ian", "ible", "ic", "ing", "ion", "ise", "ish", "ism",
                   "ist", "ity", "ive", "ize", "less", "logy", "ly", "ment", "ness", "or", "ous", "s", "ship", "sion",
                   "tion", "ty"]

    transition_prob['<s>'] = dict()
    for line in lines:
        pairs = line.strip().split(' ')
        current_tag = '<s>'
        for pair in pairs:
            if '/' in pair:
                word_tag = pair.split('/')
                word = '/'.join(word_tag[0:len(word_tag) - 1])
                tag = word_tag[-1]

                if word in word_occurrence:
                    word_occurrence[word] += 1
                else:
                    word_occurrence[word] = 1

                if word in word_tag_list:
                    if tag not in word_tag_list[word]:
                        word_tag_list[word].append(tag)
                else:
                    word_tag_list[word] = [tag]

                if tag in emission_prob:
                    if word in emission_prob[tag]:
                        emission_prob[tag][word] += 1
                    else:
                        emission_prob[tag][word] = 1
                else:
                    emission_prob[tag] = dict()
                    emission_prob[tag][word] = 1

                if tag in transition_prob[current_tag]:
                    transition_prob[current_tag][tag] += 1
                else:
                    transition_prob[current_tag][tag] = 1
                    if tag not in transition_prob:
                        transition_prob[tag] = dict()

                current_tag = tag

        if '</s>' in transition_prob[current_tag]:
            transition_prob[current_tag]['</s>'] += 1
        else:
            transition_prob[current_tag]['</s>'] = 1

    for word in word_occurrence:
        if word_occurrence[word] == 1:
            predefined_unk_words.add(word)
        all_words.append(word)

    for tag in emission_prob:
        suffix_counts = [0 for _ in range(len(suffix_list))]
        _SUM_ = 0
        _UNK_ = 0
        _UPPER_ = 0
        _LOWER_ = 0
        _TITLE_ = 0
        _OTHER_CAP_ = 0
        _HAS_NUMBER_ = 0
        _HAS_DASH_ = 0
        _HAS_SLASH_ = 0
        for word in emission_prob[tag]:
            _SUM_ += emission_prob[tag][word]
            if word in predefined_unk_words:
                _UNK_ += 1
            if word.isupper():
                _UPPER_ += emission_prob[tag][word]
            elif word.istitle():
                _TITLE_ += emission_prob[tag][word]
            elif word.islower():
                _LOWER_ += emission_prob[tag][word]
            else:
                _OTHER_CAP_ += emission_prob[tag][word]

            for i in range(len(suffix_list)):
                suffix = suffix_list[i]
                if word.lower().endswith(suffix):
                    suffix_counts[i] += emission_prob[tag][word]
            if bool(re.search(r'\d', word)):
                _HAS_NUMBER_ += emission_prob[tag][word]
            if '-' in word:
                _HAS_DASH_ += emission_prob[tag][word]
            if '/' in word:
                _HAS_SLASH_ += emission_prob[tag][word]

        emission_prob[tag]["_SUM_"] = _SUM_
        emission_prob[tag]["_UNK_"] = _UNK_
        emission_prob[tag]["_UPPER_"] = _UPPER_
        emission_prob[tag]["_TITLE_"] = _TITLE_
        emission_prob[tag]["_LOWER_"] = _LOWER_
        emission_prob[tag]["_OTHER_CAP_"] = _OTHER_CAP_
        emission_prob[tag]["_HAS_NUMBER_"] = _HAS_NUMBER_
        emission_prob[tag]["_NO_NUMBER_"] = _SUM_ - _HAS_NUMBER_
        emission_prob[tag]["_HAS_DASH_"] = _HAS_DASH_
        emission_prob[tag]["_NO_DASH_"] = _SUM_ - _HAS_DASH_
        emission_prob[tag]["_HAS_SLASH_"] = _HAS_SLASH_
        emission_prob[tag]["_NO_SLASH_"] = _SUM_ - _HAS_SLASH_

        for i in range(len(suffix_list)):
            suffix = suffix_list[i]
            emission_prob[tag]["_SUFFIX_" + suffix + "_"] = suffix_counts[i]

    for tag in transition_prob:
        _SUM_ = 0
        for next_tag in transition_prob[tag]:
            _SUM_ += transition_prob[tag][next_tag]
        transition_prob[tag]["_SUM_"] = _SUM_

    for tag in transition_prob:
        log_transition_prob_sum = math.log(transition_prob[tag]["_SUM_"])
        for next_tag in transition_prob[tag]:
            if next_tag == "_SUM_":
                transition_prob[tag]["_SUM_"] = log_transition_prob_sum
            elif transition_prob[tag][next_tag] != 0:
                transition_prob[tag][next_tag] = math.log(transition_prob[tag][next_tag]) - log_transition_prob_sum
            else:
                transition_prob[tag][next_tag] = float('-inf')

        if tag == '<s>':
            continue

        log_emission_prob_sum = math.log(emission_prob[tag]["_SUM_"])
        for word in emission_prob[tag]:
            if word == "_SUM_":
                emission_prob[tag][word] = log_emission_prob_sum
            elif emission_prob[tag][word] != 0:
                emission_prob[tag][word] = math.log(emission_prob[tag][word]) - log_emission_prob_sum
            else:
                emission_prob[tag][word] = float('-inf')

    model = {
        "emission_prob": emission_prob,
        "transition_prob": transition_prob,
        "all_words": all_words,
        "word_tag_list": word_tag_list
    }

    f = open(model_file, 'w')
    json.dump(model, f)
    f.close()

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
