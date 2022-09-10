import json
import numpy as np
import pandas as pd

def json_to_spo(json_path, spo_path):
    with open(json_path, 'r', encoding='UTF-8') as f1:
        dict = json.load(f1)
        with open(spo_path, 'w', encoding='UTF-8') as f2:
            for k, v in dict.items():
                for l in v:
                    f2.write(l[0]+'\t'+l[1]+'\t'+l[2]+'\n')

def split_train_test(data_path, filename, ratio=0.2):
    data_df = pd.read_csv(data_path+filename, sep='\t', header=0)
    all_data = pd.DataFrame(columns=['label', 'text_a'])

    label_map = {'not_hate':0, 'implicit_hate':1, 'explicit_hate':2}
    # 将标签映射为数字
    all_data['label'] = [label_map[x] for x in data_df['class']]
    all_data['text_a'] = data_df['post']

    train_data = all_data.sample(frac=ratio)
    test_data = all_data[~all_data.index.isin(train_data.index)]
    train_data.to_csv(data_path+"train.tsv", sep='\t', index=False)
    test_data.to_csv(data_path+"test.tsv", sep='\t', index=False)


if __name__ == '__main__':
    # json_to_spo("data/triples.json", "data/triples.spo")
    split_train_test("data/implicit_hate_v1_stg1_posts/", "implicit_hate_v1_stg1_posts.tsv")