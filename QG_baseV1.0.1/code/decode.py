"""Decode Seq2Seq model with beam search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqAttentionSharedEmbedding
from data_utils import read_question_generation_data, get_minibatch, read_config
from beam_search import Beam
from evaluate import get_bleu
import os
import time


class BeamSearchDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        config,
        model_weights,
        src,
        trg,
        gui,
        beam_size=3
    ):
        """Initialize model."""
        self.config = config
        self.model_weights = model_weights
        self.beam_size = beam_size

        self.output = config['data']['output']+'beam'
        self.src = src
        self.trg = trg
        self.gui = gui
        self.src_dict = src['word2id']
        self.tgt_dict = trg['word2id']
        self.gui_dict = gui['word2id']
        self._load_model()

    def _load_model(self):
        print('Loading pretrained model')
        if self.config['model']['seq2seq'] == 'vanilla':
            print('Loading Seq2Seq Vanilla model')

            self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        elif self.config['model']['seq2seq'] == 'attention':
            print('Loading Seq2Seq Attention model')

            self.model = Seq2SeqAttention(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                ctx_hidden_dim=self.config['model']['dim'],
                attention_mode='dot',
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            ).cuda()

        elif self.config['model']['seq2seq'] == 'dialog':
            self.model = Seq2SeqAttentionSharedEmbedding(
                emb_dim=self.config['model']['dim_word_src'],
                type_emb_dim=self.config['model']['dim_type_emb'],
                vocab_size=len(self.src['word2id']),
                type_size=self.config['model']["type_size"],
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                ctx_hidden_dim=self.config['model']['dim'],

                attention_mode='dot',
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src['word2id']['<pad>'],
                pad_token_trg=self.trg['word2id']['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.05
            ).cuda()

        self.model.load_state_dict(torch.load(
            open(self.model_weights)
        ))

    def get_hidden_representation(self, input_src, input_gui, input_type,):
        """Get hidden representation for a sentence."""
        src_emb = self.model.embedding(input_src)
        gui_emb = self.model.embedding(input_gui)

        self.h0_encoder, self.c0_encoder = self.model.get_state(input_src)
        self.h0_gui_encoder, self.c0_gui_encoder, self.atten_gui_h = self.model.get_gui_state(input_type)

        src_h, (src_h_t, src_c_t) = self.model.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        gui_h, (gui_h_t, gui_c_t) = self.model.gui_encoder(
            gui_emb, (self.h0_gui_encoder, self.c0_gui_encoder)
        )

        if self.model.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

            h_t_gui = torch.cat((gui_h_t[-1], gui_h_t[-2]), 1)
            c_t_gui = torch.cat((gui_c_t[-1], gui_c_t[-2]), 1)

        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

            h_t_gui = gui_h_t[-1]
            c_t_gui = gui_c_t[-1]

        h_all_t = torch.cat((h_t_gui, h_t), 1)
        c_all_t = torch.cat((c_t_gui, c_t), 1)


        return src_h,gui_h, (h_all_t, c_all_t)

    def get_init_state_decoder(self, input):
        """Get init state for decoder."""
        #decoder_init_state = nn.Tanh()(self.model.encoder2decoder_h(input))
        decoder_init_state_h = nn.Tanh()(self.model.encoder2decoder_h(input[0].squeeze(0)))  # the size haven't been changed
        decoder_init_state_c = nn.Tanh()(self.model.encoder2decoder_c(input[1].squeeze(0)))
        return decoder_init_state_h, decoder_init_state_c

    def decode_batch(self, idx):
        """Decode a minibatch."""
        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            self.src['data'], self.src_dict, idx,
            self.config['data']['batch_size'],
            self.config['data']['max_src_length'], is_gui= False, add_start=True, add_end=True
        )

        input_lines_gui, output_lines_gui, lens_gui, mask_gui, input_type_gui = get_minibatch(
            self.gui['data'], self.gui['word2id'], idx, self.config['data']['batch_size'],
            self.config['data']['max_gui_length'], is_gui=True, add_start=True, add_end=True,
            line_types=self.gui['type']
        )

        beam_size = self.beam_size

        #  (1) run the encoder on the src

        context_h, context_gui_h, (context_h_t, context_c_t) = self.get_hidden_representation(
            input_lines_src, input_lines_gui, input_type_gui
        )

        context_h = context_h.transpose(0, 1)  # Make things sequence first.
        context_gui_h = context_gui_h.transpose(0, 1)

        #  (3) run the decoder to generate sentences, using beam search

        batch_size = context_h.size(1)

        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))
        context_gui = Variable(context_gui_h.data.repeat(1, beam_size, 1))
        atten_gui = Variable(self.atten_gui_h.data.repeat(1, beam_size, 1))

        dec_states = [
            Variable(context_h_t.data.repeat(1, beam_size, 1)),
            Variable(context_c_t.data.repeat(1, beam_size, 1))
        ]

        beam = [
            Beam(beam_size, self.tgt_dict, cuda=True)
            for k in range(batch_size)
        ]

        dec_states[0],dec_states[1] = self.get_init_state_decoder(dec_states)#[0].squeeze(0))

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        #print('--------------------')
        for i in range(self.config['data']['max_trg_length']):

            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.model.embedding(Variable(input).transpose(1, 0))

            #print(trg_emb.size())
            #print(context.size())
            #print(context_gui.size())
            #print(atten_gui.size())
            #print("=======================")
            trg_h, (trg_h_t, trg_c_t) = self.model.decoder(
                trg_emb,
                (dec_states[0].squeeze(0), dec_states[1].squeeze(0)),
                (context, context_gui, atten_gui.squeeze(0))
            )

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            out = F.softmax(self.model.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.model.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context)
            context_gui = update_active(context_gui)
            atten_gui = update_active(atten_gui)


            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores

    def translate(self):
        """Translate the whole dataset."""
        trg_preds = []
        trg_gold = []
        output_res = open(self.output,'w')
        for j in xrange(
            0, len(self.src['data']),
            self.config['data']['batch_size']
        ):
            """Decode a single minibatch."""
            print('Decoding %d out of %d ' % (j, len(self.src['data'])))
            hypotheses, scores = decoder.decode_batch(j)
            all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            all_preds = [
                ' '.join([trg['id2word'][x] for x in hyp[:-1]])
                for hyp in all_hyp_inds
            ]



            # Get target minibatch
            input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
                get_minibatch(
                    self.trg['data'], self.tgt_dict, j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_trg_length'],
                    is_gui=False, add_start=True, add_end=True
                )
            )

            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
            all_gold = [
                ' '.join([trg['id2word'][x] for x in hyp[:-1]])
                for hyp in all_gold_inds
            ]

            trg_preds += all_preds
            trg_gold += all_gold

        output_res.writelines('\n'.join(trg_preds))
        bleu_score = get_bleu(trg_preds, trg_gold)
        output_res.close()
        print('BLEU : %.5f ' % (bleu_score))


class GreedyDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        config,
        model_weights,
        src,
        trg,
        gui,
        beam_size=1
    ):
        """Initialize model."""
        self.config = config
        self.model_weights = model_weights
        self.beam_size = beam_size

        self.output = config['data']['output']
        self.src = src
        self.trg = trg
        self.gui = gui
        self.src_dict = src['word2id']
        self.tgt_dict = trg['word2id']
        self.gui_dict = gui['word2id']
        self._load_model()

    def _load_model(self):
        print('Loading pretrained model')
        if self.config['model']['seq2seq'] == 'vanilla':
            print('Loading Seq2Seq Vanilla model')

            self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.
            ).cuda()

        elif self.config['model']['seq2seq'] == 'dialog':

            self.model = Seq2SeqAttentionSharedEmbedding(
                emb_dim=self.config['model']['dim_word_src'],
                type_emb_dim=self.config['model']['dim_type_emb'],
                vocab_size=len(self.src['word2id']),
                type_size=self.config['model']["type_size"],
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                ctx_hidden_dim=self.config['model']['dim'],

                attention_mode='dot',
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                pad_token_src=self.src['word2id']['<pad>'],
                pad_token_trg=self.trg['word2id']['<pad>'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.05
            ).cuda()

        self.model.load_state_dict(torch.load(
            open(self.model_weights)
        ))

    def decode_minibatch(
        self,
        input_lines_src,
        input_lines_trg,
        input_lines_gui,
        input_type_gui,
        output_lines_trg_gold
    ):
        """Decode a minibatch."""
        for i in xrange(self.config['data']['max_trg_length']):

            decoder_logit = self.model(input_lines_src, input_lines_trg, input_lines_gui, input_type_gui)
            word_probs = self.model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        return input_lines_trg

    def translate(self):
        """Evaluate model."""
        preds = []
        ground_truths = []
        out_put = open(self.output,'w')

        for j in xrange(0, len(self.src['data']),self.config['data']['batch_size']):

            print('Decoding : %d out of %d ' % (j, len(self.src['data'])))
            # Get source minibatch
            input_lines_src, output_lines_src, lens_src, mask_src = (
                get_minibatch(
                    self.src['data'], self.src['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_src_length'],
                    is_gui=False, add_start=True, add_end=True
                )
            )
            #if input_lines_src.size(0) != self.config['data']['batch_size']:
            #    break

            input_lines_src = Variable(input_lines_src.data, volatile=True)
            output_lines_src = Variable(output_lines_src.data, volatile=True)
            mask_src = Variable(mask_src.data, volatile=True)

            # Get target minibatch
            input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
                get_minibatch(
                    self.trg['data'], self.trg['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_trg_length'],
                    is_gui=False, add_start=True, add_end=True
                )
            )

            input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
            output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
            mask_src = Variable(mask_src.data, volatile=True)

            input_lines_gui, output_lines_gui, lens_gui, mask_gui, input_type_gui = get_minibatch(
                self.gui['data'], self.gui['word2id'], j, self.config['data']['batch_size'],
                self.config['data']['max_gui_length'], is_gui=True, add_start=True, add_end=True, line_types=self.gui['type']
            )

            input_lines_gui = Variable(input_lines_gui.data, volatile=True)
            output_lines_gui = Variable(output_lines_gui.data, volatile=True)
            input_type_gui = Variable(input_type_gui.data, volatile=True)
            mask_gui = Variable(mask_gui.data, volatile=True)


            # Initialize target with <s> for every sentence
            input_lines_trg = Variable(torch.LongTensor(
                [
                    [trg['word2id']['<s>']]
                    for i in xrange(input_lines_src.size(0))
                ]
            ), volatile=True).cuda()

            # Decode a minibatch greedily __TODO__ add beam search decoding
            input_lines_trg = self.decode_minibatch(
                input_lines_src, input_lines_trg,
                input_lines_gui,
                input_type_gui,
                output_lines_trg_gold
            )

            # Copy minibatch outputs to cpu and convert ids to words
            input_lines_trg = input_lines_trg.data.cpu().numpy()
            input_lines_trg = [
                [self.trg['id2word'][x] for x in line]
                for line in input_lines_trg
            ]

            # Do the same for gold sentences
            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            output_lines_trg_gold = [
                [self.trg['id2word'][x] for x in line]
                for line in output_lines_trg_gold
            ]

            # Process outputs
            for sentence_pred, sentence_real, sentence_real_src in zip(
                input_lines_trg,
                output_lines_trg_gold,
                output_lines_src
            ):
                '''
                while True:
                    if '<unk>' in sentence_pred:
                        sentence_pred = sentence_pred.remove('<unk>')

                    else:
                        break
                '''
                if '</s>' in sentence_pred:
                    index = sentence_pred.index('</s>')
                else:
                    index = len(sentence_pred)

                preds.append(sentence_pred[:index + 1])

                out_put.writelines(' '.join(sentence_pred[1:index]) + '\n')
                print('Predicted : %s ' % (' '.join(sentence_pred[1:index])))


                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                else:
                    index = len(sentence_real)

                ground_truths.append(['<s>'] + sentence_real[:index + 1])

                print('-----------------------------------------------')
                print('Real : %s ' % (' '.join(sentence_real[:index])))

                print('===============================================')

            #print(preds)

        bleu_score = get_bleu(preds, ground_truths)
        print('BLEU score : %.5f ' % (bleu_score))
        out_put.close()

if __name__ == '__main__':

    model_config = '/home/huwenp/project/Question_generate/ourQG_s2s_base_2guilayer/dialog/experiment/params/config_dialog_test.json'
    model_weights = '/home/huwenp/project/Question_generate/ourQG_s2s_base_2guilayer/dialog/experiment/model/model_chat__src_en__trg_en__attention_attention__dim_600__emb_dim_300__optimizer_adam__n_layers_src_1__n_layers_trg_1__bidir_True__epoch_13.model'

    config = read_config(model_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['management']['gpu_index']

    time.sleep(2)

    src, trg, gui= read_question_generation_data(config, 'train')

    src_test, trg_test, gui_test = read_question_generation_data(config, 'test')


    src_test['word2id'] = src['word2id']
    src_test['id2word'] = src['id2word']

    trg_test['word2id'] = trg['word2id']
    trg_test['id2word'] = trg['id2word']

    gui_test['word2id'] = gui['word2id']
    gui_test['id2word'] = gui['id2word']

    # decoder = BeamSearchDecoder(config, model_weights, src_test, trg_test)
    # decoder.translate()

    if config['model']['decode'] == 'greedy':
        decoder = GreedyDecoder(config, model_weights, src_test, trg_test, gui_test)
        decoder.translate()
    else:
        decoder = BeamSearchDecoder(config, model_weights, src_test, trg_test, gui_test, 1)
        decoder.translate()
    '''
    allHyp, allScores = decoder.decode_batch(0)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
    all_preds = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_hyp_inds]

    input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
        get_minibatch(
            trg['data'], trg['word2id'], 0,
            80,
            50,
            add_start=True, add_end=True
        )
    )

    output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
    all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
    all_gold = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_gold_inds]

    for hyp, gt in zip(all_preds, all_gold):
        print hyp, len(hyp.split())
        print '-------------------------------------------------'
        print gt
        print '================================================='
    '''
