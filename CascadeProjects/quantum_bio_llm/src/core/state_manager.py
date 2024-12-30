import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class StateManager:
    def __init__(self):
        """Inicializa el gestor de estados del sistema"""
        self.quantum_state = None
        self.neural_state = None
        self.memory_state = None
        self.system_metrics = {
            'start_time': datetime.now().isoformat(),
            'updates': 0,
            'adaptations': 0,
            'state_changes': []
        }
        self.history = []
        
    def update_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Actualiza el estado del sistema.
        
        Args:
            state_dict: Diccionario con estados a actualizar
        """
        timestamp = datetime.now().isoformat()
        
        if 'quantum_state' in state_dict:
            self.quantum_state = state_dict['quantum_state']
            
        if 'neural_state' in state_dict:
            self.neural_state = state_dict['neural_state']
            
        if 'memory_state' in state_dict:
            self.memory_state = state_dict['memory_state']
            
        self.system_metrics['updates'] += 1
        self.system_metrics['state_changes'].append({
            'timestamp': timestamp,
            'components_updated': list(state_dict.keys())
        })
        
        # Registrar en historial
        self.history.append({
            'timestamp': timestamp,
            'type': 'state_update',
            'components': list(state_dict.keys()),
            'metrics': self._calculate_metrics()
        })
        
    def record_adaptation(self, adaptation_info: Dict[str, Any]) -> None:
        """
        Registra una adaptación del sistema.
        
        Args:
            adaptation_info: Información sobre la adaptación
        """
        self.system_metrics['adaptations'] += 1
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'adaptation',
            'info': adaptation_info,
            'metrics': self._calculate_metrics()
        })
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Dict con el estado del sistema
        """
        current_time = datetime.now().isoformat()
        
        return {
            'timestamp': current_time,
            'components_status': {
                'quantum': self._get_component_status(self.quantum_state),
                'neural': self._get_component_status(self.neural_state),
                'memory': self._get_component_status(self.memory_state)
            },
            'metrics': self._calculate_metrics(),
            'history_summary': self._get_history_summary()
        }
        
    def _get_component_status(self, state: Any) -> Dict[str, Any]:
        """
        Obtiene el estado de un componente.
        
        Args:
            state: Estado del componente
            
        Returns:
            Dict con el estado del componente
        """
        if state is None:
            return {'status': 'not_initialized'}
            
        if isinstance(state, (np.ndarray, torch.Tensor)):
            return {
                'status': 'active',
                'type': type(state).__name__,
                'shape': state.shape if hasattr(state, 'shape') else None
            }
            
        if isinstance(state, dict):
            return {
                'status': 'active',
                'type': 'dict',
                'keys': list(state.keys())
            }
            
        return {
            'status': 'active',
            'type': type(state).__name__
        }
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas del sistema.
        
        Returns:
            Dict con métricas
        """
        return {
            'uptime_seconds': (
                datetime.fromisoformat(datetime.now().isoformat()) - 
                datetime.fromisoformat(self.system_metrics['start_time'])
            ).total_seconds(),
            'total_updates': self.system_metrics['updates'],
            'total_adaptations': self.system_metrics['adaptations'],
            'last_changes': self.system_metrics['state_changes'][-5:]  # últimos 5 cambios
        }
        
    def _get_history_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del historial.
        
        Returns:
            Dict con resumen del historial
        """
        if not self.history:
            return {'status': 'no_history'}
            
        return {
            'total_events': len(self.history),
            'last_event': self.history[-1],
            'event_types': self._count_event_types(),
            'recent_events': self.history[-5:]  # últimos 5 eventos
        }
        
    def _count_event_types(self) -> Dict[str, int]:
        """
        Cuenta los tipos de eventos en el historial.
        
        Returns:
            Dict con conteo de eventos
        """
        counts = {}
        for event in self.history:
            event_type = event['type']
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
        
    def clear_history(self) -> None:
        """Limpia el historial del sistema"""
        self.history.clear()
        self.system_metrics['state_changes'].clear()
        print("Historial del sistema limpiado")