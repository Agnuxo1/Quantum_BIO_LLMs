import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any

class OpticalNetworkVisualizer:
    def __init__(self, network: Any):
        """
        Visualizador de la red neuronal óptica.
        
        Args:
            network: Red neuronal a visualizar
        """
        self.network = network
        self.graph = nx.Graph()
        self.fig = None
        self.ax = None
        
    def build_graph(self):
        """Construye el grafo de la red."""
        # Limpiar grafo existente
        self.graph.clear()
        
        # Añadir nodos y conexiones
        for i, layer in enumerate(self.network.layers):
            layer_size = layer.output_dim  # Usar output_dim en lugar de len()
            for j in range(layer_size):
                # Posición vertical centrada para cada capa
                pos_y = (j - layer_size/2) * 0.5
                self.graph.add_node(f"L{i}N{j}", 
                                  pos=(i, pos_y),
                                  layer=i,
                                  neuron=j)
                
                # Añadir conexiones con la capa anterior
                if i > 0:
                    prev_layer = self.network.layers[i-1]
                    prev_layer_size = prev_layer.output_dim
                    for k in range(prev_layer_size):
                        weight = layer.weights[k, j].item()  # Acceder directamente a los pesos
                        self.graph.add_edge(f"L{i-1}N{k}", 
                                          f"L{i}N{j}",
                                          weight=weight)
    
    def get_edge_colors(self) -> List[str]:
        """Obtiene colores para las conexiones basados en los pesos."""
        weights = [self.graph.edges[edge]['weight'] for edge in self.graph.edges()]
        normalized_weights = np.array(weights)
        normalized_weights = (normalized_weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return plt.cm.viridis(normalized_weights)
    
    def visualize(self, show_weights: bool = True):
        """
        Visualiza la red neuronal.
        
        Args:
            show_weights: Si se muestran los pesos de las conexiones
        """
        self.build_graph()
        
        # Crear nueva figura
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Obtener posiciones de los nodos
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Dibujar nodos
        nx.draw_networkx_nodes(self.graph, pos,
                             node_color='lightblue',
                             node_size=500)
        
        # Dibujar conexiones
        edge_colors = self.get_edge_colors()
        nx.draw_networkx_edges(self.graph, pos,
                             edge_color=edge_colors,
                             width=2)
        
        # Añadir etiquetas
        nx.draw_networkx_labels(self.graph, pos,
                              font_size=8,
                              font_weight='bold')
        
        if show_weights:
            edge_labels = nx.get_edge_attributes(self.graph, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(self.graph, pos,
                                       edge_labels=edge_labels,
                                       font_size=6)
        
        # Configurar visualización
        plt.title("Red Neuronal Óptica Cuántica")
        plt.axis('off')
        plt.tight_layout()
        
    def show(self):
        """Muestra la visualización."""
        if self.fig is not None:
            plt.show()
            
    def save(self, filepath: str):
        """
        Guarda la visualización.
        
        Args:
            filepath: Ruta donde guardar la imagen
        """
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
