import argparse
import string
import pickle
import logging
import pandas as pd
import numpy as np
from tokenizers import get_tokenizer_with_vocab
from fastai.text import TextClasDataBunch, text_classifier_learner, AWD_LSTM, \
                        TextLMDataBunch, awd_lstm_lm_config, language_model_learner, awd_lstm_clas_config


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="Training language detector")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Ratio of training size in trian/valid split")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=300, help="Hidden size of AWD LSTM model")
    parser.add_argument("--emb-size", type=int, default=200, help="Embeddings size of AWD LSTM model")
    parser.add_argument("--lm-lr", type=float, default=1e-2, help="Learning rate for language model encoder")
    parser.add_argument("--lm-epoch", type=int, default=10, help="Number of epochs for language model encoder")
    parser.add_argument("--lm-encoder-name", type=str, default='lm_enc', help="Filename of encoder model")
    parser.add_argument("--classifier-lr", type=float, default=5e-3, help="Learning rate for classifier")
    parser.add_argument("--classifier-epoch", type=int, default=20, help="Number of epochs for classifier")
    parser.add_argument("--model-path", type=str, required=True, help="Exported model path")
    return parser


def train_valid_split(data, train_ratio):
    mask = np.random.rand(len(data)) < train_ratio
    train_df = data[mask]
    valid_df = data[~mask]
    return train_df, valid_df


def trian_language_model(params, train_df, valid_df, tokenizer, vocab):
    logging.info("Start training language model")
    data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=valid_df, path="", tokenizer=tokenizer, vocab=vocab, bs=params.bs)
    config_lm = awd_lstm_lm_config.copy()
    config_lm['n_hid'] = params.hidden_size
    config_lm['emb_sz'] = params.emb_size
    learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained=False, config=config_lm)
    learn_lm.fit_one_cycle(params.lm_epoch, params.lm_lr)
    learn_lm.freeze_to(-2)
    learn_lm.fit_one_cycle(params.lm_epoch, slice(params.lm_lr/(2.6**4), params.lm_lr))
    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(params.lm_epoch, slice(params.lm_lr/(2.6**4), params.lm_lr))
    learn_lm.save_encoder(params.lm_encoder_name)


def train_classifier(params, train_df, valid_df, tokenizer, vocab):
    logging.info("Start training classifier")
    data = TextClasDataBunch.from_df(path='.', train_df=train_df, valid_df=valid_df,
                            tokenizer=tokenizer, vocab=vocab,bs=params.bs)
    config = awd_lstm_clas_config.copy()
    config['n_hid'] = params.hidden_size
    config['emb_sz'] = params.emb_size
    learn = text_classifier_learner(data, AWD_LSTM, pretrained=False, config=config)
    learn.load_encoder(params.lm_encoder_name)
    learn.fit_one_cycle(params.classifier_epoch, max_lr=params.classifier_lr)
    learn.freeze_to(-2)
    learn.fit_one_cycle(params.classifier_epoch, slice(params.classifier_lr/(2.6**4), params.classifier_lr))
    learn.unfreeze()
    learn.fit_one_cycle(params.classifier_epoch, slice(params.classifier_lr/(2.6**4), params.classifier_lr))
    learn.export(params.model_path)


def train(params):
    data = pd.read_csv(params.input)
    data = data[['language', 'file_body']]
    data = data.astype(str)
    tokenizer, vocab = get_tokenizer_with_vocab()
    train_df, valid_df = train_valid_split(data, train_ratio=params.train_ratio)
    trian_language_model(params, train_df, valid_df, tokenizer, vocab)
    train_classifier(params, train_df, valid_df, tokenizer, vocab)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    train(params)