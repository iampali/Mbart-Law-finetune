from environment_variables import checkpoint, model_save_path
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from moe_adaptor import SparseMOE
import torch.nn as nn

def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer

def init_model(device: str = 'cuda', get_lora_model: bool = True):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map=device,
        trust_remote_code=True,
        dtype=torch.bfloat16
    )

    if get_lora_model:
        for param in model.parameters():
            param.requires_grad = False
            
        for i in range(len(model.model.encoder.layers)):
            if (i+1) % 4 == 0:
                original_layer = model.model.encoder.layers[i]
                model.model.encoder.layers[i] = EncoderLayerWithMoE(original_layer)

        for i in range(len(model.model.decoder.layers)):
            if (i+1) % 4 == 0:
                original_layer = model.model.decoder.layers[i]
                model.model.decoder.layers[i] = DecoderLayerWithMoE(original_layer)

    return model

class EncoderLayerWithMoE(nn.Module):
    def __init__(self, original_layer, num_experts=4, top_k=2):
        super().__init__()
        self.original_layer = original_layer
        hidden_size = original_layer.fc2.out_features
        
        layer_device = original_layer.fc2.weight.device
        layer_dtype = original_layer.fc2.weight.dtype
        
        self.moe = SparseMOE(hidden_size, num_experts, top_k).to(device=layer_device, dtype=layer_dtype)

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        hidden_states = outputs[0]
        
        hidden_states = self.moe(hidden_states)

        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return (hidden_states,)
    

class DecoderLayerWithMoE(nn.Module):
    def __init__(self, original_layer, num_experts=4, top_k=2):
        super().__init__()
        self.original_layer = original_layer
        hidden_size = original_layer.fc2.out_features
        
        layer_device = original_layer.fc2.weight.device
        layer_dtype = original_layer.fc2.weight.dtype
        
        self.moe = SparseMOE(hidden_size, num_experts, top_k).to(device=layer_device, dtype=layer_dtype)

    def forward(self, *args, **kwargs):
        # Run original layer
        outputs = self.original_layer(*args, **kwargs)
        hidden_states = outputs[0]
        
        hidden_states = self.moe(hidden_states)

        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return (hidden_states,)