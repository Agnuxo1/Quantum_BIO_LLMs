import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class PlasticityModule(nn.Module):
    def __init__(self, size: int, learning_rate: float = 0.01):
        """
        Módulo de plasticidad sináptica.
        
        Args:
            size: Tamaño del módulo
            learning_rate: Tasa de aprendizaje
        """
        super().__init__()
        self.size = size
        self.learning_rate = learning_rate
        self.plasticity = nn.Parameter(torch.ones(size))
        self.adaptation_history = []
        self.last_update = None
        
    def adapt(self, activity: torch.Tensor) -> None:
        """
        Adapta la plasticidad basada en la actividad.
        
        Args:
            activity: Tensor de actividad neuronal
        """
        plasticity_change = self.stdp_curve(activity)
        self.plasticity.data += self.learning_rate * plasticity_change
        
        # Registrar adaptación
        current_time = datetime.now().isoformat()
        self.adaptation_history.append({
            'timestamp': current_time,
            'mean_plasticity': self.plasticity.detach().mean().item(),
            'max_change': plasticity_change.abs().max().item()
        })
        self.last_update = current_time
        
    def stdp_curve(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Implementa la curva STDP (Spike-Timing-Dependent Plasticity).
        
        Args:
            activity: Tensor de actividad
            
        Returns:
            Cambios de plasticidad
        """
        time_diff = torch.linspace(-5, 5, self.size).to(activity.device)
        return torch.exp(-torch.abs(time_diff)) * torch.sign(time_diff)

    def get_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado del módulo.
        
        Returns:
            Dict con el estado
        """
        return {
            'size': self.size,
            'learning_rate': self.learning_rate,
            'current_plasticity': self.plasticity.detach().tolist(),
            'adaptation_history': self.adaptation_history,
            'last_update': self.last_update
        }

class BioInspiredLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adaptation_rate: float = 0.01):
        """
        Capa neural bioinspirada.
        
        Args:
            input_dim: Dimensión de entrada
            output_dim: Dimensión de salida
            adaptation_rate: Tasa de adaptación
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adaptation_rate = adaptation_rate
        
        # Componentes principales
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.plasticity = PlasticityModule(output_dim, adaptation_rate)
        
        # Memoria bacteriana
        self.bacterial_memory = torch.zeros(output_dim)
        self.memory_decay = 0.99
        self.memory_history = []
        
        # Estado de adaptación
        self.adaptation_state = {
            'activity_history': [],
            'weight_changes': [],
            'plasticity_values': []
        }
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'forward_passes': 0,
            'adaptations': 0,
            'total_activity': 0,
            'peak_activity': 0
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso forward con adaptación dinámica.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        # Aplicar pesos con plasticidad
        adapted_weights = self.weights * self.plasticity.plasticity.unsqueeze(0)
        output = torch.matmul(x, adapted_weights) + self.bias
        
        # Activación no lineal con memoria bacteriana
        output = torch.tanh(output + self.bacterial_memory)
        
        # Actualizar memoria bacteriana
        self._update_bacterial_memory(output)
        
        # Registrar actividad y métricas
        self._record_activity(output)
        self.performance_metrics['forward_passes'] += 1
        self.performance_metrics['total_activity'] += output.abs().mean().item()
        self.performance_metrics['peak_activity'] = max(
            self.performance_metrics['peak_activity'],
            output.abs().max().item()
        )
        
        return output
        
    def adapt(self, feedback: torch.Tensor) -> None:
        """
        Adapta la capa basada en retroalimentación.
        
        Args:
            feedback: Tensor de retroalimentación
        """
        self.plasticity.adapt(feedback)
        self._update_adaptation_state()
        self.performance_metrics['adaptations'] += 1
        
    def _update_bacterial_memory(self, output: torch.Tensor) -> None:
        """
        Actualiza la memoria bacteriana.
        
        Args:
            output: Tensor de salida actual
        """
        self.bacterial_memory = self.memory_decay * self.bacterial_memory + \
                              (1 - self.memory_decay) * output.detach()
                              
        self.memory_history.append({
            'timestamp': datetime.now().isoformat(),
            'mean_memory': self.bacterial_memory.mean().item(),
            'max_memory': self.bacterial_memory.max().item()
        })
        
    def _record_activity(self, activity: torch.Tensor) -> None:
        """
        Registra la actividad neuronal.
        
        Args:
            activity: Tensor de actividad
        """
        self.adaptation_state['activity_history'].append({
            'timestamp': datetime.now().isoformat(),
            'mean_activity': activity.detach().mean().item(),
            'max_activity': activity.detach().max().item()
        })
        
    def _update_adaptation_state(self) -> None:
        """Actualiza el estado de adaptación"""
        current_time = datetime.now().isoformat()
        
        self.adaptation_state['weight_changes'].append({
            'timestamp': current_time,
            'magnitude': self.weights.grad.detach().norm().item() if self.weights.grad is not None else 0
        })
        
        self.adaptation_state['plasticity_values'].append({
            'timestamp': current_time,
            'mean_plasticity': self.plasticity.plasticity.detach().mean().item()
        })
        
    def get_adaptation_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado de adaptación actual.
        
        Returns:
            Dict con el estado de adaptación
        """
        return {
            'layer_info': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'adaptation_rate': self.adaptation_rate
            },
            'weights': {
                'mean': self.weights.detach().mean().item(),
                'std': self.weights.detach().std().item(),
                'shape': list(self.weights.shape)
            },
            'plasticity': self.plasticity.get_state(),
            'bacterial_memory': {
                'current': self.bacterial_memory.tolist(),
                'decay_rate': self.memory_decay,
                'history': self.memory_history[-10:]  # últimos 10 registros
            },
            'adaptation_history': {
                'activity': self.adaptation_state['activity_history'][-10:],
                'weight_changes': self.adaptation_state['weight_changes'][-10:],
                'plasticity': self.adaptation_state['plasticity_values'][-10:]
            },
            'performance': self.performance_metrics
        }

class BioInspiredNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], adaptation_rate: float = 0.01):
        """
        Red neural bioinspirada.
        
        Args:
            layer_sizes: Lista con tamaños de capas
            adaptation_rate: Tasa de adaptación
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.adaptation_rate = adaptation_rate
        
        # Crear capas
        self.layers = nn.ModuleList([
            BioInspiredLayer(layer_sizes[i], layer_sizes[i+1], adaptation_rate)
            for i in range(len(layer_sizes)-1)
        ])
        
        # Métricas de red
        self.network_metrics = {
            'created_at': datetime.now().isoformat(),
            'forward_passes': 0,
            'total_adaptations': 0,
            'layer_performances': []
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso forward a través de todas las capas.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        for layer in self.layers:
            x = layer(x)
        self.network_metrics['forward_passes'] += 1
        return x
        
    def adapt_all(self, feedback: torch.Tensor) -> None:
        """
        Adapta todas las capas.
        
        Args:
            feedback: Tensor de retroalimentación
        """
        for layer in self.layers:
            layer.adapt(feedback)
        self.network_metrics['total_adaptations'] += 1
            
    def get_network_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo de la red.
        
        Returns:
            Dict con el estado de la red
        """
        layer_states = [layer.get_adaptation_state() for layer in self.layers]
        
        # Calcular métricas agregadas
        total_params = sum(p.numel() for p in self.parameters())
        mean_plasticity = np.mean([
            layer.plasticity.plasticity.detach().mean().item()
            for layer in self.layers
        ])
        
        return {
            'architecture': {
                'layer_sizes': self.layer_sizes,
                'adaptation_rate': self.adaptation_rate,
                'total_parameters': total_params
            },
            'layer_states': layer_states,
            'network_metrics': {
                **self.network_metrics,
                'mean_network_plasticity': mean_plasticity,
                'last_update': datetime.now().isoformat()
            },
            'performance_summary': {
                'total_forward_passes': self.network_metrics['forward_passes'],
                'total_adaptations': self.network_metrics['total_adaptations'],
                'mean_layer_activity': np.mean([
                    state['performance']['total_activity'] / max(1, state['performance']['forward_passes'])
                    for state in layer_states
                ])
            }
        }