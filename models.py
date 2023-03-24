
import copy

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5EncoderModel, T5PreTrainedModel

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput


class SimplePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dropout(first_token_tensor)
        pooled_output = self.dense(pooled_output)
        pooled_output = F.relu(pooled_output)
        
        return pooled_output

class SimplePoolerMultiLayers(nn.Module):
    def __init__(self, config, num_layers=3):
        super().__init__()
        self.num_layers=num_layers
        self.dense_layers = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for i in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        pooled_output = hidden_states[:, 0]

        for layer in self.dense_layers:
            pooled_output = layer(pooled_output)
            pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output

class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
        divisor = mask.sum(dim=1).view(-1, 1).float()
        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = self.dropout(sentence_sums)
        pooled_output = self.dense(pooled_output)
        pooled_output = F.relu(pooled_output)
        
        return pooled_output


class MeanPoolerMultiLayers(nn.Module):
    def __init__(self, config, num_layers=3):
        super().__init__()
        self.num_layers=num_layers
        self.dense_layers = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for i in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
        divisor = mask.sum(dim=1).view(-1, 1).float()
        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = sentence_sums
        for layer in self.dense_layers:
            pooled_output = layer(pooled_output)
            pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output


class T5EncoderForSequenceClassificationSimple(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationSimple, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class T5EncoderForSequenceClassificationSimpleMultiLayers(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationSimpleMultiLayers, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = SimplePoolerMultiLayers(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class T5EncoderForSequenceClassificationMean(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationMean, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class T5EncoderForSequenceClassificationMeanMultiLayers(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationMeanMultiLayers, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = MeanPoolerMultiLayers(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_model(model_type, pretrained_model, label_list):
    num_labels = len(label_list)
    id2label = {idx:key for idx, key in enumerate(label_list)}
    label2id = {key:idx for idx, key in enumerate(label_list)}
    if model_type == 'T5-simple':
        return T5EncoderForSequenceClassificationSimple.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    elif model_type == "T5-simple-m":
        return T5EncoderForSequenceClassificationSimpleMultiLayers.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    elif model_type == 'T5-mean':
        return T5EncoderForSequenceClassificationMean.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    elif model_type == 'T5-mean-m':
        return T5EncoderForSequenceClassificationMeanMultiLayers.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    elif model_type == 'seq-classification':
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    elif model_type == 'seq2seq':
        return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
    else:
        raise ValueError('invalid model_type: {}'.format(model_type))

def get_model_class_by_name(model_class):
    if model_class == 'T5EncoderForSequenceClassificationSimple':
        return T5EncoderForSequenceClassificationSimple
    elif model_class == 'T5EncoderForSequenceClassificationSimpleMultiLayers':
        return T5EncoderForSequenceClassificationSimpleMultiLayers
    elif model_class == 'T5EncoderForSequenceClassificationMean':
        return T5EncoderForSequenceClassificationMean
    elif model_class == 'T5EncoderForSequenceClassificationMeanMultiLayers':
        return T5EncoderForSequenceClassificationMeanMultiLayers
    elif model_class == 'AutoModelForSequenceClassification':
        return AutoModelForSequenceClassification
    elif model_class == 'AutoModelForSeq2SeqLM':
        return AutoModelForSeq2SeqLM
    else:
        raise ValueError('invalid model class: {}'.format(model_class))


if __name__ == "__main__":
    pass
    #model = T5EncoderForSequenceClassification.from_pretrained('KETI-AIR/ke-t5-base', num_labels=NUM_EMOTION)
    #model = model.from_pretrained('KETI-AIR/ke-t5-base', num_labels=NUM_EMOTION)
    
    
    #print(model)
    #model = AutoModelForSequenceClassification.from_pretrained('KETI-AIR/ke-t5-base', num_labels=NUM_EMOTION)



