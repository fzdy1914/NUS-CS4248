# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json
import re


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    f = open(test_file, "r")
    lines = f.readlines()
    f.close()

    f = open(model_file, "r")
    model = json.load(f)
    f.close()

    result = ""
    suffix_list = ["able", "acy", "age", "al", "ance", "ant", "ary", "ate", "dom", "ed", "ee", "en", "ence", "ent",
                   "ent", "er", "est", "ful", "fy", "hood", "ian", "ible", "ic", "ing", "ion", "ise", "ish", "ism",
                   "ist", "ity", "ive", "ize", "less", "logy", "ly", "ment", "ness", "or", "ous", "s", "ship", "sion",
                   "tion", "ty"]

    emission_prob = model["emission_prob"]
    transition_prob = model["transition_prob"]
    all_words = model["all_words"]
    word_tag_list = model["word_tag_list"]

    all_tags = set()
    for tag in emission_prob:
        if tag != "<s>":
            all_tags.add(tag)

    for line in lines:
        words = line.strip().split(" ")
        V = [dict() for _ in range(len(words))]

        prev_tags = ["<s>"]
        for i in range(len(words)):
            word = words[i]
            if word not in all_words and word.lower() in all_words:
                word = word.lower()
            if word not in all_words and word.upper() in all_words:
                word = word.upper()

            current_tags = list()
            if word not in all_words:
                for tag in all_tags:
                    log_prob = emission_prob[tag]["_UNK_"]

                    if word.isupper():
                        log_prob += emission_prob[tag]["_UPPER_"]
                    elif word.istitle():
                        log_prob += emission_prob[tag]["_TITLE_"]
                    elif word.islower():
                        log_prob += emission_prob[tag]["_LOWER_"]
                    else:
                        log_prob += emission_prob[tag]["_OTHER_CAP_"]

                    if bool(re.search(r"\d", word)):
                        log_prob += emission_prob[tag]["_HAS_NUMBER_"]
                    else:
                        log_prob += emission_prob[tag]["_NO_NUMBER_"]

                    if "-" in word:
                        log_prob += emission_prob[tag]["_HAS_DASH_"]
                    else:
                        log_prob += emission_prob[tag]["_NO_DASH_"]

                    if "/" in word:
                        log_prob += emission_prob[tag]["_HAS_SLASH_"]
                    else:
                        log_prob += emission_prob[tag]["_NO_SLASH_"]

                    for suffix in suffix_list:
                        if word.lower().endswith(suffix):
                            log_prob += emission_prob[tag]["_SUFFIX_" + suffix + "_"]

                    for pre_tag in prev_tags:
                        trans_prob = transition_prob[pre_tag][tag] if tag in transition_prob[pre_tag] else float("-inf")
                        total_prob = log_prob + trans_prob + V[i - 1][pre_tag][0] if i != 0 else log_prob + trans_prob
                        if tag not in V[i] or total_prob > V[i][tag][0]:
                            V[i][tag] = (total_prob, pre_tag)
                    if V[i][tag][0] != float("-inf"):
                        current_tags.append(tag)
            else:
                for tag in word_tag_list[word]:
                    log_prob = emission_prob[tag][word]
                    for pre_tag in prev_tags:
                        trans_prob = transition_prob[pre_tag][tag] if tag in transition_prob[pre_tag] else - transition_prob[pre_tag]["_SUM_"]
                        total_prob = log_prob + trans_prob + V[i - 1][pre_tag][0] if i != 0 else log_prob + trans_prob
                        if tag not in V[i] or total_prob > V[i][tag][0]:
                            V[i][tag] = (total_prob, pre_tag)

                    if V[i][tag][0] != float("-inf"):
                        current_tags.append(tag)
            prev_tags = current_tags

        final_best_pair = None
        for pre_tag in prev_tags:
            trans_prob = transition_prob[pre_tag]["</s>"] if "</s>" in transition_prob[pre_tag] else float("-inf")
            total_prob = trans_prob + V[len(words) - 1][pre_tag][0]
            if final_best_pair is None or total_prob > final_best_pair[0]:
                final_best_pair = (total_prob, pre_tag)

        current_best_pair = final_best_pair
        best_tags = ["" for _ in range(len(words))]
        for i in reversed(range(len(words))):
            best_tag = current_best_pair[1]
            best_tags[i] = best_tag
            current_best_pair = V[i][best_tag]

        for i in range(len(words)):
            result += words[i] + "/" + best_tags[i] + " "
        result += "\n"

    f = open(out_file, "w")
    f.write(result)
    f.close()

    print("Finished...")


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
