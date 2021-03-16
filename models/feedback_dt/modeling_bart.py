import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import (
    BART_INPUTS_DOCSTRING,
    BART_START_DOCSTRING,
    Seq2SeqLMOutput,
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
from utils import NegativeCElLoss

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
        masked_lm_loss = torch.tensor([0.0], requires_grad=True).to(input_ids.device)

        loss_fct = CrossEntropyLoss()
        n_loss_fct = NegativeCElLoss()
        
        if decoder_labels is not None:
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), decoder_labels.view(-1))
            masked_lm_loss = masked_lm_loss + loss
        if n_decoder_labels is not None:
            n_loss = n_loss_fct(n_lm_logits.view(-1, self.config.vocab_size), n_decoder_labels.view(-1))
            masked_lm_loss = masked_lm_loss + n_loss
                
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )