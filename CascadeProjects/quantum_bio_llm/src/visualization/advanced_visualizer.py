import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import torch
from datetime import datetime
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.express as px

class AdvancedNetworkVisualizer:
    """Visualizador avanzado con gráficos 3D y animaciones"""
    
    def __init__(self, network: Any, figsize: tuple = (12, 8)):
        self.network = network
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.graph = nx.Graph()
        self.node_positions_3d = {}
        self.animation = None
        
    def create_3d_layout(self) -> None:
        """Crea un layout 3D para la red neuronal"""
        for i, layer in enumerate(self.network.layers):
            layer_size = layer.output_dim
            for j in range(layer_size):
                # Posición en espiral para mejor visualización
                theta = 2 * np.pi * j / layer_size
                r = 1.0
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = i * 2  # Separación entre capas
                
                node_id = f"L{i}N{j}"
                self.node_positions_3d[node_id] = (x, y, z)
                self.graph.add_node(node_id, 
                                  pos=(x, y, z),
                                  layer=i,
                                  neuron=j)
                
                # Conexiones con capa anterior
                if i > 0:
                    prev_layer = self.network.layers[i-1]
                    for k in range(prev_layer.output_dim):
                        weight = layer.weights[k, j].item()
                        prev_node = f"L{i-1}N{k}"
                        self.graph.add_edge(prev_node, node_id, weight=weight)
                        
    def plot_3d_network(self, show_weights: bool = True, animate: bool = True) -> None:
        """
        Visualiza la red en 3D con plotly para interactividad.
        
        Args:
            show_weights: Si se muestran los pesos
            animate: Si se anima la visualización
        """
        self.create_3d_layout()
        
        # Crear figura 3D interactiva
        edge_x = []
        edge_y = []
        edge_z = []
        edge_weights = []
        
        for edge in self.graph.edges():
            x0, y0, z0 = self.node_positions_3d[edge[0]]
            x1, y1, z1 = self.node_positions_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            edge_weights.append(self.graph.edges[edge]['weight'])
            
        # Normalizar pesos para colores
        weights_norm = np.array(edge_weights)
        weights_norm = (weights_norm - weights_norm.min()) / (weights_norm.max() - weights_norm.min())
        
        # Crear figura
        fig = go.Figure()
        
        # Añadir conexiones
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(color=px.colors.sequential.Viridis[::int(256/len(weights_norm))],
                     width=2),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Añadir nodos
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        
        for node in self.graph.nodes():
            x, y, z = self.node_positions_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            layer = self.graph.nodes[node]['layer']
            neuron = self.graph.nodes[node]['neuron']
            node_text.append(f'Layer {layer}, Neuron {neuron}')
            
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=8,
                color=node_z,
                colorscale='Viridis',
                line=dict(color='white', width=0.5)
            ),
            text=node_text,
            hoverinfo='text'
        ))
        
        # Configurar layout
        fig.update_layout(
            title='Red Neural Cuántica-Bioinspirada (3D)',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Capa'),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False
        )
        
        if animate:
            frames = []
            for i in range(36):  # Rotación de 360 grados
                frame = go.Frame(
                    layout=dict(
                        scene_camera=dict(
                            eye=dict(
                                x=1.5 * np.cos(i * np.pi / 18),
                                y=1.5 * np.sin(i * np.pi / 18),
                                z=1.5
                            )
                        )
                    )
                )
                frames.append(frame)
            
            fig.frames = frames
            
            # Añadir controles de animación
            fig.update_layout(
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    buttons=[dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    )]
                )]
            )
            
        return fig
        
    def plot_activity_heatmap(self, activity_data: torch.Tensor) -> go.Figure:
        """
        Crea un mapa de calor interactivo de la actividad neuronal.
        
        Args:
            activity_data: Tensor de actividad
            
        Returns:
            Figura de plotly
        """
        activity = activity_data.detach().numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=activity,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Capa: %{x}<br>Neurona: %{y}<br>Activación: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Mapa de Activación Neural',
            xaxis_title='Capa',
            yaxis_title='Neurona',
            height=600
        )
        
        return fig
        
    def plot_weight_distribution(self) -> go.Figure:
        """
        Visualiza la distribución de pesos en la red.
        
        Returns:
            Figura de plotly
        """
        weights = []
        layer_labels = []
        
        for i, layer in enumerate(self.network.layers):
            w = layer.weights.detach().numpy().flatten()
            weights.extend(w)
            layer_labels.extend([f'Capa {i}'] * len(w))
            
        fig = go.Figure()
        
        for i in range(len(self.network.layers)):
            layer_weights = [w for w, l in zip(weights, layer_labels) if l == f'Capa {i}']
            fig.add_trace(go.Violin(
                y=layer_weights,
                name=f'Capa {i}',
                box_visible=True,
                meanline_visible=True
            ))
            
        fig.update_layout(
            title='Distribución de Pesos por Capa',
            yaxis_title='Valor del Peso',
            violinmode='group'
        )
        
        return fig
        
    def create_dashboard(self, activity_data: Optional[torch.Tensor] = None) -> None:
        """
        Crea un dashboard interactivo completo.
        
        Args:
            activity_data: Datos de actividad opcional
        """
        # Crear subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'heatmap'}],
                  [{'type': 'violin', 'colspan': 2}, None]],
            subplot_titles=('Estructura de Red 3D', 'Mapa de Activación',
                          'Distribución de Pesos')
        )
        
        # Añadir visualizaciones
        network_fig = self.plot_3d_network(animate=False)
        for trace in network_fig.data:
            fig.add_trace(trace, row=1, col=1)
            
        if activity_data is not None:
            activity_fig = self.plot_activity_heatmap(activity_data)
            fig.add_trace(activity_fig.data[0], row=1, col=2)
            
        weight_fig = self.plot_weight_distribution()
        for trace in weight_fig.data:
            fig.add_trace(trace, row=2, col=1)
            
        # Actualizar layout
        fig.update_layout(
            title='Dashboard de Red Neural Cuántica-Bioinspirada',
            height=1200,
            showlegend=True
        )
        
        return fig
        
    def save_visualization(self, filepath: str, fig: Optional[go.Figure] = None) -> None:
        """
        Guarda la visualización en HTML interactivo.
        
        Args:
            filepath: Ruta donde guardar
            fig: Figura opcional para guardar
        """
        if fig is None:
            fig = self.plot_3d_network()
        
        fig.write_html(filepath)
