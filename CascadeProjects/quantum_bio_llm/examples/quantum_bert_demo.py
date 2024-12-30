import sys
import os
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.core.system import QuantumBioSystem
from src.llm.bert_integration import QuantumBertIntegration
from src.interface.chat import ChatInterface
from src.visualization.optical_network import OpticalNetworkVisualizer
from src.visualization.advanced_visualizer import AdvancedNetworkVisualizer

def main():
    # Configuración del sistema
    config = {
        "n_qubits": 8,
        "coherence_time": 100,
        "layer_sizes": [768, 384, 192],  # Compatible con BERT-base
        "adaptation_rate": 0.01,
        "memory_dimensions": (64, 64),
        "num_memory_shards": 4,
        "bert_model": "bert-base-uncased",
        "max_length": 512
    }
    
    print("Inicializando sistema...")
    
    # Inicializar componentes
    quantum_bio_system = QuantumBioSystem(config)
    bert_integration = QuantumBertIntegration(config)
    
    # Crear visualizador
    visualizer = OpticalNetworkVisualizer(quantum_bio_system.neural_network)
    advanced_viz = AdvancedNetworkVisualizer(quantum_bio_system.neural_network)
    
    class EnhancedModel:
        def __init__(self, quantum_system, bert_system):
            self.quantum_system = quantum_system
            self.bert_system = bert_system
            
        def generate_response(self, text):
            # Procesar texto con BERT
            bert_results = self.bert_system.encode_text(text)
            embeddings, attention = bert_results
            
            # Procesar con sistema cuántico
            quantum_results = self.quantum_system.process_data(
                embeddings.detach().numpy().flatten()
            )
            
            # Integrar resultados
            enhanced_results = self.bert_system.process_with_quantum(
                text,
                quantum_results['quantum_results']['state']
            )
            
            # Generar respuesta basada en los tokens modificados
            response = " ".join(enhanced_results['tokens'])
            
            return response
    
    # Crear modelo mejorado
    enhanced_model = EnhancedModel(quantum_bio_system, bert_integration)
    
    print("Sistema inicializado. Iniciando interfaz de chat...")
    
    # Crear y ejecutar interfaz
    chat_interface = ChatInterface(enhanced_model)
    
    # Visualizar red antes de empezar
    visualizer.visualize()
    visualizer.save("network_visualization.png")
    print("Visualización de red guardada en 'network_visualization.png'")
    
    # Crear visualizaciones avanzadas
    advanced_viz.plot_3d_network(animate=True)
    advanced_viz.plot_activity_heatmap(torch.tensor([0.5, 0.3, 0.2]))
    advanced_viz.plot_weight_distribution()
    advanced_viz.create_dashboard(torch.tensor([0.5, 0.3, 0.2]))
    
    # Iniciar chat
    chat_interface.run()

if __name__ == "__main__":
    main()
