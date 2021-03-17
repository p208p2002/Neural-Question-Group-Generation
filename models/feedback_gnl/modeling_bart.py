import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers.models.bart.modeling_bart import (
    BART_INPUTS_DOCSTRING,
    BART_START_DOCSTRING,
    # Seq2SeqLMOutput,
    _CONFIG_FOR_DOC,
    BART_GENERATION_EXAMPLE,
    shift_tokens_right,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    add_end_docstrings,
    BartPretrainedModel,
    BartForConditionalGeneration,
    BartModel,
    BartConfig,
    BartEncoder,
    BartDecoder,
    CrossEntropyLoss
)
from dataclasses import dataclass
from utils import NegativeCElLoss
from .argparser import get_args
args = get_args()

class CustomBartForConditionalGeneration(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config._vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
        #
        decoder_labels=None,
        n_decoder_inputs=None,
        n_decoder_attention_mask=None,
        n_decoder_labels=None
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs[0])
        
        # negative
        if n_decoder_labels is not None:
            n_decoder_inputs = n_decoder_inputs.view(n_decoder_inputs.shape[0]*n_decoder_inputs.shape[1],n_decoder_inputs.shape[-1])
            n_decoder_attention_mask = n_decoder_attention_mask.view(*n_decoder_inputs.shape)
            n_decoder_labels = n_decoder_labels.view(n_decoder_labels.shape[0]*n_decoder_labels.shape[1],n_decoder_labels.shape[-1])
            
            encoder_hidden_states = encoder_outputs[0]
            encoder_dim_size = encoder_hidden_states.shape[-1]
            encoder_length = encoder_hidden_states.shape[-2]
            new_encoder_hidden_states_size = [n_decoder_inputs.shape[0],encoder_length,encoder_dim_size]
            encoder_hidden_states = encoder_hidden_states.expand(*new_encoder_hidden_states_size)
            attention_mask = attention_mask.expand(*new_encoder_hidden_states_size[:-1])
        
            # negate decoder outputs
            n_decoder_outputs = self.decoder(
                input_ids=n_decoder_inputs,
                attention_mask=n_decoder_attention_mask,
                
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            n_lm_logits = self.lm_head(n_decoder_outputs[0])
        
        # loss
        loss = torch.tensor([0.0], requires_grad=True).to(input_ids.device)
        n_loss = torch.tensor([0.0], requires_grad=True).to(input_ids.device)

        loss_fct = CrossEntropyLoss()
        n_loss_fct = NegativeCElLoss(alpha=args.alpha)
        
        if decoder_labels is not None:
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), decoder_labels.view(-1))
        if n_decoder_labels is not None:
            n_decoder_labels = torch.where(decoder_labels == n_decoder_labels,torch.LongTensor([-100]).to(input_ids.device),n_decoder_labels)
            n_loss = n_loss_fct(n_lm_logits.view(-1, self.config.vocab_size), n_decoder_labels.view(-1))
                        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        else:
            return Seq2SeqLMOutput(
                n_loss=n_loss,
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
            
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    n_loss: Optional[torch.FloatTensor] = None 
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None