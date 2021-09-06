import argparse, re, os

from deep.gectorPredict.utils.helpers import read_lines
from deep.gectorPredict.gector.gec_model import GecBERTModel
from kernel.settings import BASE_DIR

import nltk
from nltk.tokenize.util import align_tokens

if os.path.join(BASE_DIR, "deep/nltk_data") not in nltk.data.path:
    nltk.data.path.append(os.path.join(BASE_DIR, "deep/nltk_data"))

import spacy
from spacy.symbols import ORTH

spacy_tokenizer = spacy.load(os.path.join(BASE_DIR, "deep/pt_core_news_sm"))
# add special rule for the tokenizer
special_case = [{ORTH: "às"}]
spacy_tokenizer.tokenizer.add_special_case("às", special_case)


def message(original_token, replacement):
    verb_msg = f"O verbo <marker>{original_token}</marker> não concorda com o resto da frase ou não é frequentemente utilizado neste contexto. Considere a alternativa."
    other_msg = f'A palavra <marker>{original_token}</marker> pode ter sido confundida com a palavra "{replacement}".'
    crase_msg = f"Possível erro de crase. Considere a alternativa."
    comma_msg = f"Possível erro de vírgula. Considere a alternativa."

    other_replace_condition = any(
        original_token == tok for tok in ["e", "esta", "da", "mal", "mau"]
    )
    crase_condition_1 = any(original_token == tok for tok in ["à", "às"]) and any(
        replacement == tok for tok in ["a", "as"]
    )
    crase_condition_2 = any(original_token == tok for tok in ["a", "as"]) and any(
        replacement == tok for tok in ["à", "às"]
    )
    comma_condition_1 = original_token[-1] == "," and replacement[-1] != ","
    comma_condition_2 = original_token[-1] != "," and replacement[-1] == ","

    if other_replace_condition:
        return other_msg
    elif crase_condition_1 or crase_condition_2:
        return crase_msg
    elif comma_condition_1 or comma_condition_2:
        return comma_msg

    return verb_msg


def short_message(original_token, replacement):
    verb_short_msg = f"Modifique a forma verbal"
    other_short_msg = f"Modifique a palavra"
    crase_short_msg = f"Erro de crase"
    comma_short_msg = f"Erro de vírgula"

    other_replace_condition = any(
        original_token == tok for tok in ["e", "esta", "da", "mal", "mau"]
    )
    crase_condition_1 = any(original_token == tok for tok in ["à", "às"]) and any(
        replacement == tok for tok in ["a", "as"]
    )
    crase_condition_2 = any(original_token == tok for tok in ["a", "as"]) and any(
        replacement == tok for tok in ["à", "às"]
    )
    comma_condition_1 = original_token[-1] == "," and replacement[-1] != ","
    comma_condition_2 = original_token[-1] != "," and replacement[-1] == ","

    if other_replace_condition:
        return other_short_msg
    elif crase_condition_1 or crase_condition_2:
        return crase_short_msg
    elif comma_condition_1 or comma_condition_2:
        return comma_short_msg

    return verb_short_msg


def examples(original_token, replacement):
    verb_incorrect_example = f"Minha mãe fizeram dois bolos."
    verb_correct_example = f"Minha mãe fez dois bolos."
    other_incorrect_example = f"Essa pessoa esta muito irritada."
    other_correct_example = f"Essa pessoa está muito irritada."
    crase_incorrect_example = f"Eu fui as compras no supermercado."
    crase_correct_example = f"Eu fui às compras no supermercado."
    comma_incorrect_example = f"Ele foi ao, mercado."
    comma_correct_example = f"Ele foi ao mercado."

    other_replace_condition = any(
        original_token == tok for tok in ["e", "esta", "da", "mal", "mau"]
    )
    crase_condition_1 = any(original_token == tok for tok in ["à", "às"]) and any(
        replacement == tok for tok in ["a", "as"]
    )
    crase_condition_2 = any(original_token == tok for tok in ["a", "as"]) and any(
        replacement == tok for tok in ["à", "às"]
    )
    comma_condition_1 = original_token[-1] == "," and replacement[-1] != ","
    comma_condition_2 = original_token[-1] != "," and replacement[-1] == ","

    if other_replace_condition:
        return other_incorrect_example, other_correct_example
    elif crase_condition_1 or crase_condition_2:
        return crase_incorrect_example, crase_correct_example
    elif comma_condition_1 or comma_condition_2:
        return comma_incorrect_example, comma_correct_example

    return verb_incorrect_example, verb_correct_example


def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, _, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, _, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, "w") as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + "\n")
    return cnt_corrections, [" ".join(x) for x in predictions]


# if True, the repl entry is added, otherwise it is not.
def removeFalsePositives(sent_label, tokens_in, regexp_dic):
    for key in regexp_dic:
        label_regex = regexp_dic[key][0]
        pronoun_regex = regexp_dic[key][1]
        if bool(label_regex.search(sent_label)) and all(
            not bool(pronoun_regex.search(tok)) for tok in tokens_in
        ):
            return False
    return True


def predict_for_paragraph(
    input_paragraph, model, batch_size=32, tokenizer_method="split"
):
    test_data = nltk.tokenize.sent_tokenize(input_paragraph, language="portuguese")
    split_positions = [x for x in align_tokens(test_data, input_paragraph)]

    if type(test_data) == str:
        test_data = [test_data]

    data_dic = {}
    if tokenizer_method == "split+spacy":
        data_dic["spacy"] = test_data[:]
        data_dic["split"] = test_data[:]
    elif tokenizer_method == "spacy":
        data_dic["spacy"] = test_data[:]
    elif tokenizer_method == "split":
        data_dic["split"] = test_data[:]

    repl = {}
    for method, sentences in data_dic.items():
        cnt_corrections = 0
        tokenized_sentences = []
        predictions = []
        diffs = []
        spans = []
        batch = []
        for sent in sentences:
            if method == "split":
                tokenized_sentence = sent.split()
            elif method == "spacy":
                tokenized_sentence = [str(tok) for tok in spacy_tokenizer(sent)]

            tokenized_sentences.append(tokenized_sentence)

            # getting to know where the spaces are at
            # print('here', tokenized_sentence, sent)
            sent_spans = align_tokens(tokenized_sentence, sent)
            sent_diffs = [sent_spans[0][0] - 0]
            old_span = sent_spans[0]
            for i, span in enumerate(sent_spans[1:]):
                diff = span[0] - old_span[1]
                sent_diffs.append(diff)
                old_span = span
            spans.append(sent_spans)
            diffs.append(sent_diffs)

            batch.append(tokenized_sentence)
            if len(batch) == batch_size:
                preds, labels, cnt = model.handle_batch(batch)
                predictions.extend(preds)
                cnt_corrections += cnt
                batch = []
        if batch:
            preds, labels, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt

        print("number of iterations:", cnt_corrections, " | tokenizer:", method)

        # removing the first label which is for the SENT_START token
        labels = [x[1:] for x in labels]
        # defining regex patterns for later
        re1S = re.compile(r"VM.([1][P]|[23][SP])_VM.1[S]$")
        re2S = re.compile(r"VM.([2][P]|[13][SP])_VM.2[S]$")
        re1P = re.compile(r"VM.([1][S]|[23][SP])_VM.1[P]$")
        re2P = re.compile(r"VM.([2][S]|[13][SP])_VM.2[P]$")
        reEU = re.compile(r"^[Ee][Uu]$")
        reTU = re.compile(r"^[Tt][Uu]$")
        reEUNOS = re.compile(r"^([Ee][Uu]|[Nn][óÓ][sS])$")
        reTUVOS = re.compile(r"^([Tt][Uu]|[Vv][óÓ][sS])$")
        regexp_dic = {
            "1S": [re1S, reEU],
            "2S": [re2S, reTU],
            "1P": [re1P, reEUNOS],
            "2P": [re2P, reTUVOS],
        }
        # obtain a dictionary with replacements
        for sent_pos, tokens_in, tokens_out, spaces_lengths, sent_labels in zip(
            split_positions, tokenized_sentences, predictions, diffs, labels
        ):
            # this is for the case where the label is equal to '$APPEND...' and the sentences have different lengths. Our sentences would then have to be treated differently.
            if len(tokens_in) != len(tokens_out):
                continue
            past_token_in = ""
            for i, (token_in, token_out, space_length, sent_label) in enumerate(
                zip(tokens_in, tokens_out, spaces_lengths, sent_labels)
            ):
                replace = removeFalsePositives(sent_label, tokens_in, regexp_dic)
                if i == 0:
                    pos = space_length + sent_pos[0]
                elif i > 0:
                    pos += len(past_token_in) + space_length
                if token_in != token_out:
                    length = len(token_in)
                    if replace:
                        repl[(token_in, pos)] = (pos, length, token_out, method)

                past_token_in = token_in

        # print('number of corrections:', cnt_corrections)
    return repl


def main(args):
    # get all paths
    model = GecBERTModel(
        vocab_path=args.vocab_path,
        model_paths=args.model_path,
        max_len=args.max_len,
        min_len=args.min_len,
        iterations=args.iteration_count,
        min_error_probability=args.min_error_probability,
        lowercase_tokens=args.lowercase_tokens,
        model_name=args.transformer_model,
        special_tokens_fix=args.special_tokens_fix,
        log=False,
        confidence=args.additional_confidence,
        is_ensemble=args.is_ensemble,
        weigths=args.weights,
    )

    cnt_corrections, _ = predict_for_file(
        args.input_file, args.output_file, model, batch_size=args.batch_size
    )
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == "__main__":
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", help="Path to the model file.", nargs="+", required=True
    )
    parser.add_argument(
        "--vocab_path",
        help="Path to the model file.",
        default="data/output_vocabulary",  # to use pretrained models
    )
    parser.add_argument("--input_file", help="Path to the evalset file", required=True)
    parser.add_argument("--output_file", help="Path to the output file", required=True)
    parser.add_argument(
        "--max_len",
        type=int,
        help="The max sentence length" "(all longer will be truncated)",
        default=50,
    )
    parser.add_argument(
        "--min_len",
        type=int,
        help="The minimum sentence length" "(all longer will be returned w/o changes)",
        default=3,
    )
    parser.add_argument(
        "--batch_size", type=int, help="The size of hidden unit cell.", default=128
    )
    parser.add_argument(
        "--lowercase_tokens", type=int, help="Whether to lowercase tokens.", default=0
    )
    parser.add_argument(
        "--transformer_model",
        choices=[
            "bert",
            "gpt2",
            "transformerxl",
            "xlnet",
            "distilbert",
            "roberta",
            "albert",
            "bertimbaubase",
            "bertimbaularge",
        ],
        help="Name of the transformer model.",
        default="roberta",
    )
    parser.add_argument(
        "--iteration_count",
        type=int,
        help="The number of iterations of the model.",
        default=5,
    )
    parser.add_argument(
        "--additional_confidence",
        type=float,
        help="How many probability to add to $KEEP token.",
        default=0,
    )
    parser.add_argument(
        "--min_error_probability",
        type=float,
        help="Minimum probability for each action to apply. "
        "Also, minimum error probability, as described in the paper.",
        default=0.0,
    )
    parser.add_argument(
        "--special_tokens_fix",
        type=int,
        help="Whether to fix problem with [CLS], [SEP] tokens tokenization. "
        "For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.",
        default=1,
    )
    parser.add_argument(
        "--is_ensemble", type=int, help="Whether to do ensembling.", default=0
    )
    parser.add_argument(
        "--weights", help="Used to calculate weighted average", nargs="+", default=None
    )
    args = parser.parse_args()
    main(args)
