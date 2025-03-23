from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn

class T5Reranker(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.t5 = base_model
        self.classifier = torch.nn.Linear(self.t5.config.d_model, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        # Explicit encoder call
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use first token embedding for classification
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return {"loss": loss, "logits": logits}
