import argparse, os

from deep.gectorPredict.utils.helpers import read_lines
from deep.gectorPredict.gector.gec_model import GecBERTModel
from kernel.settings import BASE_DIR

import nltk
from nltk.tokenize.util import align_tokens
if os.path.join(BASE_DIR, 'deep/nltk_data') not in nltk.data.path:
    nltk.data.path.append(os.path.join(BASE_DIR, 'deep/nltk_data'))



def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + '\n')
    return cnt_corrections, [" ".join(x) for x in predictions]


def predict_for_paragraph(input_paragraph, model, batch_size=32, tokenizer_method='split'):
    #sent_tokenizer = nltk.data.load(os.path.join(BASE_DIR, 'deep/nltk_data/tokenizers/punkt/portuguese.pickle'))
    test_data = nltk.tokenize.sent_tokenize(input_paragraph, language='portuguese')
    split_positions = [x for x in align_tokens(test_data, input_paragraph)]
    predictions = []; diffs = []; tokenized_sentences = []; spans = [];
    cnt_corrections = 0
    batch = []
    if type(test_data) == str:
        test_data = [test_data]
    for sent in test_data:
        if tokenizer_method == 'split':
            tokenized_sentence = sent.split()
        elif tokenizer_method == 'nltk':
            tokenized_sentence = nltk.tokenize.word_tokenize(sent, language='portuguese')
        tokenized_sentences.append(tokenized_sentence)
        print(tokenized_sentences)
        
        # knowing where the spaces are at
        sent_spans = align_tokens(tokenized_sentence, sent)
        sent_diffs = [sent_spans[0][0] - 0]; old_span = sent_spans[0];
        for i, span in enumerate(sent_spans[1:]):
            diff = span[0] - old_span[1]
            sent_diffs.append(diff)
            old_span = span
        spans.append(sent_spans)
        diffs.append(sent_diffs)
        
        batch.append(tokenized_sentence)
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    # obtain a dictionary with replacements
    repl = dict()
    for sent_pos, tokens_in, tokens_out, spaces_lengths in zip(split_positions, tokenized_sentences, predictions, diffs):
        past_token_in = ''
        for i, (token_in, token_out, space_length) in enumerate(zip(tokens_in, tokens_out, spaces_lengths)):
            if i == 0:
                pos = space_length + sent_pos[0]
            elif i > 0:
                pos += len(past_token_in) + space_length
            if token_in != token_out:
                length = len(token_in)
                repl[(token_in,pos)] = (pos, length, token_out)
            
            past_token_in = token_in

    # print('number of corrections:', cnt_corrections)

    return repl


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    cnt_corrections, _ = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert', 'bertimbaubase', 'bertimbaularge'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    args = parser.parse_args()
    main(args)
