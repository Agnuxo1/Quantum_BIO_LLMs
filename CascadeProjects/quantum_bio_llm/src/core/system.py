import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from ..quantum.simulator import QuantumSimulator
from ..neural.bio_layer import BioInspiredNetwork
from ..memory.holographic import HolographicMemory
from .state_manager import StateManager
import json
from datetime import datetime

class QuantumBioSystem:
    def __init__(self, config: Dict[str, Any]):
        """
        Sistema principal que integra todos los componentes.
        
        Args:
            config: Diccionario con la configuración del sistema
        """
        # Validar configuración
        self._validate_config(config)
        
        # Inicializar componentes
        self.quantum_sim = QuantumSimulator(
            n_qubits=config.get("n_qubits", 8),
            coherence_time=config.get("coherence_time", 100)
        )
        
        self.neural_network = BioInspiredNetwork(
            layer_sizes=config.get("layer_sizes", [64, 32, 16]),
            adaptation_rate=config.get("adaptation_rate", 0.01)
        )
        
        self.memory = HolographicMemory(
            dimensions=config.get("memory_dimensions", (64, 64)),
            num_shards=config.get("num_memory_shards", 4)
        )
        
        self.state_manager = StateManager()
        self.config = config
        self.initialization_time = datetime.now().isoformat()
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Valida la configuración del sistema.
        
        Args:
            config: Configuración a validar
            
        Raises:
            ValueError: Si la configuración es inválida
        """
        required_keys = ["n_qubits", "layer_sizes", "memory_dimensions"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta parámetro requerido: {key}")
                
        if config["n_qubits"] < 1:
            raise ValueError("n_qubits debe ser positivo")
            
        if len(config["layer_sizes"]) < 2:
            raise ValueError("layer_sizes debe tener al menos 2 capas")
            
        if len(config["memory_dimensions"]) != 2:
            raise ValueError("memory_dimensions debe ser una tupla de 2 elementos")
            
    def process_data(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Procesa datos a través del sistema completo.
        
        Args:
            input_data: Array de entrada
            
        Returns:
            Dict con resultados del procesamiento
        """
        # Validar entrada
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
            
        # Preparar datos
        input_tensor = torch.from_numpy(input_data).float()
        
        # Paso 1: Simulación cuántica
        quantum_state = self.quantum_sim.evolve_state(input_data)
        
        # Paso 2: Procesamiento neural
        neural_output = self.neural_network(input_tensor)
        
        # Paso 3: Almacenamiento en memoria
        self.memory.store_distributed(
            pattern=input_data,
            association=neural_output.detach().numpy()
        )
        
        # Paso 4: Recuperación y análisis
        memory_results = self.memory.retrieve_distributed(input_data)
        
        # Actualizar estado del sistema
        self.state_manager.update_state({
            'quantum_state': quantum_state,
            'neural_state': neural_output.detach(),
            'memory_state': self.memory.get_stats()
        })
        
        # Retornar resultados con el formato correcto
        return {
            'quantum_results': {
                'state': quantum_state,
                'coherence': self.quantum_sim._calculate_coherence()
            },
            'neural_output': neural_output.detach().numpy(),
            'memory_results': memory_results,
            'system_state': self.state_manager.get_system_status()
        }
        
    def retrieve_pattern(self, query_pattern: np.ndarray) -> Dict[str, Any]:
        """
        Recupera un patrón similar de la memoria.
        
        Args:
            query_pattern: Patrón de consulta
            
        Returns:
            Dict con resultados de la recuperación
        """
        # Validar entrada
        if not isinstance(query_pattern, np.ndarray):
            query_pattern = np.array(query_pattern)
            
        # Preparar datos
        query_tensor = torch.from_numpy(query_pattern).float()
        
        # Procesar a través del sistema
        quantum_state = self.quantum_sim.evolve_state(query_pattern)
        neural_output = self.neural_network(query_tensor)
        
        # Recuperar de memoria
        memory_results = self.memory.retrieve_distributed(query_pattern)
        
        # Actualizar estado
        self.state_manager.update_state({
            'quantum_state': quantum_state,
            'neural_state': neural_output.detach(),
            'memory_state': self.memory.get_stats()
        })
        
        # Procesar patrones recuperados
        processed_patterns = np.array([r['pattern'] for r in memory_results])
        
        return {
            'quantum_results': {
                'state': quantum_state,
                'coherence': self.quantum_sim._calculate_coherence()
            },
            'neural_output': neural_output.detach().numpy(),
            'processed_patterns': processed_patterns,
            'memory_results': memory_results,
            'system_state': self.state_manager.get_system_status()
        }
        
    def adapt_system(self, feedback: Dict[str, np.ndarray]) -> None:
        """
        Adapta el sistema basado en retroalimentación.
        
        Args:
            feedback: Dict con retroalimentación para cada componente
        """
        # Validar feedback
        if not isinstance(feedback, dict):
            raise ValueError("feedback debe ser un diccionario")
            
        # Adaptar red neural
        if 'neural_feedback' in feedback:
            neural_feedback = torch.from_numpy(feedback['neural_feedback']).float()
            self.neural_network.adapt_all(neural_feedback)
            
        # Registrar adaptación
        self.state_manager.record_adaptation({
            'timestamp': datetime.now().isoformat(),
            'feedback_received': list(feedback.keys()),
            'neural_state': self.neural_network.get_network_state()
        })
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema.
        
        Returns:
            Dict con métricas del sistema
        """
        # Obtener métricas de cada componente
        quantum_metrics = {
            'quantum_circuit_depth': len(self.quantum_sim.gate_history),
            'coherence': self.quantum_sim._calculate_coherence(),
            'num_qubits': self.quantum_sim.n_qubits
        }
        
        neural_metrics = {
            'neural_network_size': sum(p.numel() for p in self.neural_network.parameters()),
            'layer_sizes': self.neural_network.layer_sizes,
            'adaptation_rate': self.neural_network.adaptation_rate
        }
        
        memory_metrics = self.memory.get_stats()
        memory_usage = sum(
            shard.plate.nbytes for shard in self.memory.shards
        ) / (1024 * 1024)  # Convertir a MB
        
        return {
            'quantum_circuit_depth': quantum_metrics['quantum_circuit_depth'],
            'neural_network_size': neural_metrics['neural_network_size'],
            'memory_usage': memory_usage,
            'detailed_metrics': {
                'quantum': quantum_metrics,
                'neural': neural_metrics,
                'memory': memory_metrics,
                'system_state': self.state_manager.get_system_status()
            }
        }
        
    def _serialize_complex(self, obj):
        """
        Serializa objetos para JSON, manejando números complejos y arrays numpy.
        
        Args:
            obj: Objeto a serializar
            
        Returns:
            Objeto serializable
        """
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_complex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_complex(item) for item in obj]
        return obj
        
    def save_checkpoint(self, filepath: str) -> None:
        """
        Guarda un checkpoint del sistema.
        
        Args:
            filepath: Ruta donde guardar el checkpoint
        """
        checkpoint = {
            'config': self.config,
            'initialization_time': self.initialization_time,
            'quantum_state': self._serialize_complex(self.quantum_sim.get_state()),
            'neural_state': self._serialize_complex(self.neural_network.get_network_state()),
            'memory_state': self._serialize_complex(self.memory.get_stats()),
            'system_metrics': self._serialize_complex(self.get_system_metrics()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def _deserialize_complex(self, obj):
        """
        Deserializa objetos de JSON, reconstruyendo números complejos.
        
        Args:
            obj: Objeto a deserializar
            
        Returns:
            Objeto deserializado
        """
        if isinstance(obj, dict):
            if 'real' in obj and 'imag' in obj:
                return complex(obj['real'], obj['imag'])
            return {k: self._deserialize_complex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_complex(item) for item in obj]
        return obj
            
    def load_checkpoint(self, filepath: str) -> None:
        """
        Carga un checkpoint del sistema.
        
        Args:
            filepath: Ruta del checkpoint a cargar
        """
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
            
        # Deserializar datos complejos
        checkpoint = self._deserialize_complex(checkpoint)
            
        # Reinicializar componentes con la configuración guardada
        self.__init__(checkpoint['config'])
        
        # Restaurar estados
        self.quantum_sim.load_state(checkpoint['quantum_state'])
        self.neural_network.load_state(checkpoint['neural_state'])
        
        # Registrar carga
        self.state_manager.record_adaptation({
            'timestamp': datetime.now().isoformat(),
            'action': 'checkpoint_loaded',
            'original_timestamp': checkpoint['timestamp']
        })
        
    def get_system_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema.
        
        Returns:
            Dict con el estado del sistema
        """
        return {
            'config': self.config,
            'initialization_time': self.initialization_time,
            'quantum_state': self.quantum_sim.get_state(),
            'neural_state': self.neural_network.get_network_state(),
            'memory_state': self.memory.get_stats(),
            'system_metrics': self.get_system_metrics(),
            'last_update': datetime.now().isoformat()
        }