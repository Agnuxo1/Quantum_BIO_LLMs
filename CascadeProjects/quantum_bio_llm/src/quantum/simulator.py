import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch

class QuantumSimulator:
    def __init__(self, n_qubits: int = 8, coherence_time: float = 100.0):
        """
        Simulador cuántico avanzado.
        
        Args:
            n_qubits: Número de qubits
            coherence_time: Tiempo de coherencia en unidades arbitrarias
        """
        self.n_qubits = n_qubits
        self.coherence_time = coherence_time
        self.state_vector = self._initialize_state()
        self.gate_history = []
        self.measurement_history = []
        self.evolution_history = []
        self.last_update = None
        
    def _initialize_state(self) -> np.ndarray:
        """
        Inicializa el estado cuántico en |0...0⟩.
        
        Returns:
            Vector de estado inicial
        """
        state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        state[0] = 1.0
        return state
        
    def apply_gate(self, gate_name: str, target_qubits: List[int], params: Optional[Dict[str, float]] = None) -> None:
        """
        Aplica una compuerta cuántica.
        
        Args:
            gate_name: Nombre de la compuerta
            target_qubits: Lista de qubits objetivo
            params: Parámetros opcionales de la compuerta
        """
        gate_matrix = self._get_gate_matrix(gate_name, params)
        self._apply_matrix(gate_matrix, target_qubits)
        
        self.gate_history.append({
            'timestamp': datetime.now().isoformat(),
            'gate': gate_name,
            'qubits': target_qubits,
            'params': params
        })
        self.last_update = datetime.now().isoformat()
        
    def evolve_state(self, input_data: np.ndarray) -> np.ndarray:
        """
        Evoluciona el estado cuántico según los datos de entrada.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Estado evolucionado
        """
        # Normalizar datos
        normalized_data = input_data / np.linalg.norm(input_data)
        
        # Codificar en amplitudes
        amplitudes = self._encode_amplitudes(normalized_data)
        
        # Aplicar evolución unitaria
        evolved_state = self._apply_evolution(amplitudes)
        
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_norm': np.linalg.norm(input_data),
            'state_norm': np.linalg.norm(evolved_state),
            'coherence': self._calculate_coherence()
        })
        
        return evolved_state
        
    def measure_state(self, basis: str = 'computational') -> Dict[str, Any]:
        """
        Realiza una medición del estado.
        
        Args:
            basis: Base de medición
            
        Returns:
            Resultados de la medición
        """
        probabilities = np.abs(self.state_vector)**2
        outcome = np.random.choice(2**self.n_qubits, p=probabilities)
        
        # Colapsar estado
        self.state_vector = np.zeros_like(self.state_vector)
        self.state_vector[outcome] = 1.0
        
        measurement = {
            'outcome': outcome,
            'probabilities': probabilities.tolist(),
            'basis': basis,
            'timestamp': datetime.now().isoformat(),
            'coherence': self._calculate_coherence()
        }
        
        self.measurement_history.append(measurement)
        return measurement
        
    def _get_gate_matrix(self, gate_name: str, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Obtiene la matriz de una compuerta.
        
        Args:
            gate_name: Nombre de la compuerta
            params: Parámetros de la compuerta
            
        Returns:
            Matriz de la compuerta
        """
        if gate_name == 'H':  # Hadamard
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif gate_name == 'X':  # Pauli-X
            return np.array([[0, 1], [1, 0]])
        elif gate_name == 'Z':  # Pauli-Z
            return np.array([[1, 0], [0, -1]])
        elif gate_name == 'RY':  # Rotación-Y
            theta = params.get('theta', 0.0)
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ])
        else:
            raise ValueError(f"Compuerta {gate_name} no implementada")
            
    def _apply_matrix(self, matrix: np.ndarray, target_qubits: List[int]) -> None:
        """
        Aplica una matriz a los qubits objetivo.
        
        Args:
            matrix: Matriz a aplicar
            target_qubits: Qubits objetivo
        """
        # Implementación simplificada para 1 y 2 qubits
        if len(target_qubits) == 1:
            self._apply_single_qubit(matrix, target_qubits[0])
        elif len(target_qubits) == 2:
            self._apply_two_qubit(matrix, target_qubits)
            
    def _apply_single_qubit(self, matrix: np.ndarray, qubit: int) -> None:
        """
        Aplica una matriz a un solo qubit.
        
        Args:
            matrix: Matriz 2x2
            qubit: Índice del qubit
        """
        n = self.n_qubits
        for i in range(2**n):
            if i & (1 << qubit):
                j = i & ~(1 << qubit)
                temp = matrix[1, 0] * self.state_vector[j] + matrix[1, 1] * self.state_vector[i]
                self.state_vector[j] = matrix[0, 0] * self.state_vector[j] + matrix[0, 1] * self.state_vector[i]
                self.state_vector[i] = temp
                
    def _apply_two_qubit(self, matrix: np.ndarray, qubits: List[int]) -> None:
        """
        Aplica una matriz a dos qubits.
        
        Args:
            matrix: Matriz 4x4
            qubits: Lista de dos qubits
        """
        # Implementación básica para matrices de dos qubits
        q0, q1 = qubits
        n = self.n_qubits
        for i in range(2**n):
            if (i & (1 << q0)) and (i & (1 << q1)):
                idx = 3
            elif (i & (1 << q0)):
                idx = 2
            elif (i & (1 << q1)):
                idx = 1
            else:
                idx = 0
            self.state_vector[i] *= matrix[idx, idx]
            
    def _encode_amplitudes(self, data: np.ndarray) -> np.ndarray:
        """
        Codifica datos en amplitudes cuánticas.
        
        Args:
            data: Datos a codificar
            
        Returns:
            Amplitudes codificadas
        """
        # Asegurar que los datos caben en el espacio de Hilbert
        padded_data = np.zeros(2**self.n_qubits, dtype=np.complex128)
        padded_data[:len(data)] = data
        return padded_data / np.linalg.norm(padded_data)
        
    def _apply_evolution(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Aplica evolución unitaria.
        
        Args:
            amplitudes: Amplitudes iniciales
            
        Returns:
            Estado evolucionado
        """
        # Simular decoherencia
        decay = np.exp(-1/self.coherence_time)
        evolved = amplitudes * decay
        
        # Normalizar
        return evolved / np.linalg.norm(evolved)
        
    def _calculate_coherence(self) -> float:
        """
        Calcula la coherencia actual.
        
        Returns:
            Valor de coherencia
        """
        return np.abs(np.vdot(self.state_vector, self.state_vector))
        
    def get_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del simulador.
        
        Returns:
            Dict con el estado
        """
        return {
            'n_qubits': self.n_qubits,
            'coherence_time': self.coherence_time,
            'state_vector': self.state_vector.tolist(),
            'current_coherence': self._calculate_coherence(),
            'history': {
                'gates': self.gate_history[-10:],  # últimos 10 registros
                'measurements': self.measurement_history[-10:],
                'evolution': self.evolution_history[-10:]
            },
            'last_update': self.last_update
        }