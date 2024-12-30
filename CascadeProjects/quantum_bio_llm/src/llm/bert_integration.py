import torch
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Any, Tuple
import numpy as np

class QuantumBertIntegration:
    def __init__(self, config: Dict[str, Any]):
        """
        Integración de BERT con el sistema cuántico-bioinspirado.
        
        Args:
            config: Configuración del sistema
        """
        self.model_name = config.get("bert_model", "bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.max_length = config.get("max_length", 512)
        
    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica texto usando BERT.
        
        Args:
            text: Texto a codificar
            
        Returns:
            Tuple con embeddings y atención
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, outputs.attentions
        
    def decode_embeddings(self, embeddings: torch.Tensor) -> List[str]:
        """
        Decodifica embeddings a tokens.
        
        Args:
            embeddings: Tensor de embeddings
            
        Returns:
            Lista de tokens decodificados
        """
        # Encontrar los tokens más cercanos usando similitud coseno
        embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        vocab_embeddings = self.model.embeddings.word_embeddings.weight
        vocab_norm = vocab_embeddings / vocab_embeddings.norm(dim=-1, keepdim=True)
        
        similarity = torch.matmul(embeddings_norm, vocab_norm.t())
        token_ids = similarity.argmax(dim=-1)
        
        return self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        
    def quantum_attention(self, attention_weights: torch.Tensor, quantum_state: np.ndarray) -> torch.Tensor:
        """
        Modifica los pesos de atención usando el estado cuántico.
        
        Args:
            attention_weights: Pesos de atención originales
            quantum_state: Estado cuántico actual
            
        Returns:
            Pesos de atención modificados
        """
        # Convertir estado cuántico a tensor y asegurar dimensiones correctas
        quantum_tensor = torch.from_numpy(quantum_state).float()
        
        # Asegurar que el tensor cuántico tenga el tamaño correcto (256)
        if quantum_tensor.shape[0] != 256:
            quantum_tensor = torch.nn.functional.interpolate(
                quantum_tensor.view(1, 1, -1),
                size=256,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Normalizar y reshape para matching
        quantum_tensor = quantum_tensor.view(-1, 1)
        quantum_weights = torch.nn.functional.softmax(quantum_tensor, dim=0)
        
        # Redimensionar attention_weights si es necesario
        if attention_weights.shape[-1] != 256:
            attention_weights = torch.nn.functional.interpolate(
                attention_weights.view(1, 1, -1),
                size=256,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Modificar pesos de atención
        modified_attention = attention_weights * quantum_weights
        return torch.nn.functional.normalize(modified_attention, p=1, dim=-1)
        
    def process_with_quantum(self, text: str, quantum_state: np.ndarray) -> Dict[str, Any]:
        """
        Procesa texto usando BERT y el estado cuántico.
        
        Args:
            text: Texto a procesar
            quantum_state: Estado cuántico actual
            
        Returns:
            Dict con resultados del procesamiento
        """
        # Codificar texto
        embeddings, attention = self.encode_text(text)
        
        # Reducir dimensionalidad de los embeddings usando mean pooling
        # Primero promediamos sobre todos los tokens
        pooled_embeddings = embeddings.mean(dim=1)  # [1, 768]
        
        # Luego redimensionamos al tamaño esperado usando interpolación lineal
        target_size = 256  # Tamaño esperado por el sistema cuántico
        resized_embeddings = torch.nn.functional.interpolate(
            pooled_embeddings.unsqueeze(0),  # Añadir dimensión de batch [1, 1, 768]
            size=target_size,
            mode='linear',
            align_corners=False
        ).squeeze(0)  # [1, 256]
        
        # Aplicar modificación cuántica
        quantum_modified_attention = self.quantum_attention(attention[-1], quantum_state)
        
        # Modificar embeddings usando atención cuántica
        modified_embeddings = resized_embeddings * quantum_modified_attention.mean(dim=1, keepdim=True)
        
        # Decodificar resultados
        tokens = self.decode_embeddings(modified_embeddings)
        
        return {
            'embeddings': modified_embeddings.detach().numpy(),
            'attention': quantum_modified_attention.detach().numpy(),
            'tokens': tokens,
            'quantum_influence': np.mean(np.abs(quantum_state))
        }
