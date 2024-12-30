import numpy as np
import torch
import sys
import os
import logging

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.system import QuantumBioSystem

def main():
    # Configuración del sistema
    config = {
        "n_qubits": 8,
        "coherence_time": 100,
        "layer_sizes": [64, 32, 16],  # Dimensiones consistentes
        "adaptation_rate": 0.01,
        "num_memory_shards": 4,
        "memory_dimensions": (64, 64),  # Dimensiones de memoria ajustadas
        "memory_capacity": 10000
    }
    
    # Inicializar sistema
    system = QuantumBioSystem(config)
    
    # Generar datos de ejemplo
    input_data = np.random.randn(64)  # Vector de entrada de dimensión 64
    
    # Mostrar información de entrada
    logging.info(f"Input data shape: {input_data.shape}")
    logging.info(f"Input data contents: {input_data}")
    
    # Procesar datos
    print("Procesando datos...")
    results = system.process_data(input_data)
    
    # Mostrar resultados
    print("\nResultados del procesamiento:")
    print(f"Estado cuántico coherencia: {results['quantum_results']['coherence']:.4f}")
    print(f"Dimensiones salida neural: {results['neural_output'].shape}")
    print(f"Patrones en memoria: {len(results['memory_results'])}")
    print(f"Calidad de reconstrucción: {results['memory_results'][0]['reconstruction_quality']:.4f}")
    
    # Recuperar patrón
    print("\nRecuperando patrón similar...")
    query_pattern = input_data + np.random.randn(64) * 0.1  # Ruido pequeño
    retrieval_results = system.retrieve_pattern(query_pattern)
    
    print("\nResultados de recuperación:")
    print(f"Número de patrones recuperados: {len(retrieval_results['memory_results'])}")
    print(f"Mejor similitud: {max(r['similarity'] for r in retrieval_results['memory_results']):.4f}")
    
    # Mostrar métricas del sistema
    print("\nMétricas del sistema:")
    metrics = system.get_system_metrics()
    print(f"Profundidad circuito cuántico: {metrics['quantum_circuit_depth']}")
    print(f"Tamaño red neural: {metrics['neural_network_size']} parámetros")
    print(f"Uso de memoria: {metrics['memory_usage']:.2f} MB")
    
    # Guardar checkpoint
    print("\nGuardando checkpoint...")
    system.save_checkpoint("system_checkpoint.json")
    print("Checkpoint guardado exitosamente.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()