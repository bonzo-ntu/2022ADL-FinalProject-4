import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
from transformers import MT5Model, T5Tokenizer
import pickle as pkl
from pathlib import Path


def tokenized_colum_mt5(df, name, columns, tokenizer, path='./'):
    path = Path(path)
    for i, column in enumerate(columns):
        print(f'start tokenize {column}')

        e0 = df[column][0]
        if isinstance(e0, str) and e0.startswith('[') and e0.endswith(']'):
            sequence_token_stack = list()
            sequence_id_stack = list()
            for sequence in df[column].to_list():
                sequence = sequence.replace('nan', '')
                sequence = eval(sequence)
                encoded = tokenizer(sequence)
                sequence_token_stack.append([tokenizer.convert_ids_to_tokens(_) for _ in encoded.input_ids])
                sequence_id_stack.append(encoded.input_ids)
            df[f'{column}_token'] = sequence_token_stack
            df[f'{column}_id'] = sequence_id_stack
        else:
            encoded = tokenizer(df[column].to_list())
            df[f'{column}_token'] = [tokenizer.convert_ids_to_tokens(_) for _ in encoded.input_ids]
            df[f'{column}_id'] = encoded.input_ids            

        # write files
        path.mkdir(parents=True, exist_ok=True)
        i = "end" if i == len(columns) else i
        pkl.dump(df, open(path/f'{name}_tokenized_{i}_mt5.pkl', 'wb'))
        df.to_csv(path/f'{name}_tokenized_{i}_mt5.csv')
        print(f'end tokenize {column}')

    return df


def select_tokenizer(model_name='tuhailong/chinese-roberta-wwm-ext'): 
    if model_name == "mt5-small":
        # model = MT5Model.from_pretrained("google/mt5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    elif model_name == "chinese-roberta-wwm-ext":
        # model = AutoModel.from_pretrained("tuhailong/chinese-roberta-wwm-ext")
        tokenizer = AutoTokenizer.from_pretrained("tuhailong/chinese-roberta-wwm-ext")
    elif model_name in ("bert-base-chinese",  "ckiplab/bert-base-chinese-ner"):
        # model = AutoModel.from_pretrained("ckiplab/bert-base-chinese-ner")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    return tokenizer


if __name__ == "__main__":
    course_df = pd.read_csv('./data/courses.csv')
    course_chapter_items_sequence_df = pd.read_csv('course_chapter_items_sequence.csv')

    partial = course_df.loc[:, ['course_name', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'will_learn', 'required_tools', 'recommended_background', 'target_group']] 
    course_df.loc[:, ['course_name', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'will_learn', 'required_tools', 'recommended_background', 'target_group']] = partial.fillna('')

    model_names = ['mt5-small', 'chinese-roberta-wwm-ext', 'bert-base-chinese']
    for model_name in model_names:
        tokenizer = select_tokenizer(model_name)
        tokenized_colum_mt5(course_df, f'course_{model_name}', ['course_name', 'teacher_intro', 'will_learn', 'required_tools', 'recommended_background'], tokenizer, path='./tokenized/')
        tokenized_colum_mt5(course_chapter_items_sequence_df, f'course_chapter_items_sequence_{model_name}', ['chapter_item_name_seq'], tokenizer, path='./tokenized/')
