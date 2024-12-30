import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy.fft import fft2, ifft2
import torch
from torch import tensor
from functools import lru_cache

class PhaseConjugator:
    """Componente para conjugación de fase en memoria holográfica"""
    
    def __init__(self):
        self.conjugation_history = []
        
    def conjugate(self, wave: np.ndarray) -> np.ndarray:
        """
        Realiza conjugación de fase de una onda.
        
        Args:
            wave: Onda a conjugar
            
        Returns:
            Onda conjugada
        """
        conjugated = np.conj(wave)
        self.conjugation_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_shape': wave.shape,
            'conjugate_norm': np.linalg.norm(conjugated)
        })
        return conjugated
        
class HolographicPlate:
    """Placa holográfica para almacenamiento de patrones"""
    
    def __init__(self, dimensions: Tuple[int, int]):
        """
        Inicializa una placa holográfica.
        
        Args:
            dimensions: Dimensiones de la placa (altura, ancho)
        """
        self.dimensions = dimensions
        self.plate = np.zeros(dimensions, dtype=np.complex64)
        self.recording_history = []
        
    def record(self, wave: np.ndarray) -> None:
        """
        Registra una onda en la placa.
        
        Args:
            wave: Onda a registrar
        """
        # Asegurar dimensiones correctas
        if wave.shape != self.dimensions:
            # Redimensionar la onda al tamaño de la placa
            padded = np.zeros(self.dimensions, dtype=np.complex64)
            min_rows = min(wave.shape[0], self.dimensions[0])
            min_cols = min(wave.shape[1], self.dimensions[1])
            padded[:min_rows, :min_cols] = wave[:min_rows, :min_cols]
            wave = padded
        
        self.plate += wave
        self.recording_history.append({
            'timestamp': datetime.now().isoformat(),
            'wave_norm': np.linalg.norm(wave),
            'plate_norm': np.linalg.norm(self.plate)
        })
        
    def reconstruct(self, reference_wave: np.ndarray) -> np.ndarray:
        """
        Reconstruye un patrón usando una onda de referencia.
        
        Args:
            reference_wave: Onda de referencia
            
        Returns:
            Patrón reconstruido
        """
        # Asegurar dimensiones correctas
        if reference_wave.shape != self.dimensions:
            padded = np.zeros(self.dimensions, dtype=np.complex64)
            min_rows = min(reference_wave.shape[0], self.dimensions[0])
            min_cols = min(reference_wave.shape[1], self.dimensions[1])
            padded[:min_rows, :min_cols] = reference_wave[:min_rows, :min_cols]
            reference_wave = padded
            
        reconstruction = self.plate * reference_wave
        return reconstruction
        
    def clear(self) -> None:
        """Limpia la placa"""
        self.plate.fill(0)
        self.recording_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'clear'
        })
        
class HolographicMemory:
    """Sistema de memoria holográfica distribuida"""
    
    def __init__(self, dimensions: Tuple[int, int], num_shards: int = 4):
        """
        Inicializa el sistema de memoria holográfica.
        
        Args:
            dimensions: Dimensiones de las placas (altura, ancho)
            num_shards: Número de fragmentos de memoria
        """
        self.dimensions = dimensions
        self.num_shards = num_shards
        self.shards = [HolographicPlate(dimensions) for _ in range(num_shards)]
        self.phase_conjugator = PhaseConjugator()
        self.pattern_registry = {}
        self.interference_patterns = []
        self.retrieval_history = []
        
        # Cache para optimización
        self._pattern_cache = {}
        self._wave_cache = {}
        self._similarity_cache = {}
        self.cache_size = 1000  # Tamaño máximo de caché
        
        # Paralelización
        self.use_parallel = True
        self.batch_size = 32
        
    def _parallel_process(self, items: List[Any], process_func: callable) -> List[Any]:
        """
        Procesa items en paralelo usando torch.
        
        Args:
            items: Lista de items a procesar
            process_func: Función de procesamiento
            
        Returns:
            Lista de resultados
        """
        if not self.use_parallel or len(items) <= self.batch_size:
            return [process_func(item) for item in items]
            
        # Procesar en batches
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            if torch.cuda.is_available():
                # Usar GPU si está disponible
                batch_tensor = tensor(batch).cuda()
                results.extend(process_func(batch_tensor).cpu().numpy())
            else:
                # Procesamiento en CPU
                batch_tensor = tensor(batch)
                results.extend(process_func(batch_tensor).numpy())
        return results
        
    def _cache_key(self, data: np.ndarray) -> str:
        """Genera una clave de caché para los datos"""
        return hash(data.tobytes())
        
    def _manage_cache(self) -> None:
        """Gestiona el tamaño del caché"""
        if len(self._pattern_cache) > self.cache_size:
            # Eliminar elementos más antiguos
            remove_keys = list(self._pattern_cache.keys())[:-self.cache_size]
            for k in remove_keys:
                self._pattern_cache.pop(k, None)
                self._wave_cache.pop(k, None)
                self._similarity_cache.pop(k, None)
                
    def store_distributed(self, pattern: np.ndarray, association: np.ndarray) -> None:
        """
        Almacena un patrón distribuido en los fragmentos.
        
        Args:
            pattern: Patrón a almacenar
            association: Patrón asociado
        """
        # Usar caché si está disponible
        pattern_key = self._cache_key(pattern)
        assoc_key = self._cache_key(association)
        
        if pattern_key in self._wave_cache:
            pattern_wave = self._wave_cache[pattern_key]
        else:
            pattern_wave = self._encode_pattern(pattern)
            self._wave_cache[pattern_key] = pattern_wave
            
        if assoc_key in self._wave_cache:
            assoc_wave = self._wave_cache[assoc_key]
        else:
            assoc_wave = self._encode_pattern(association)
            self._wave_cache[assoc_key] = assoc_wave
            
        # Distribuir en fragmentos usando procesamiento paralelo
        def process_shard(shard_idx):
            shard = self.shards[shard_idx]
            shard.record(pattern_wave)
            return self._compute_interference(pattern_wave, assoc_wave)
            
        interference_patterns = self._parallel_process(
            range(self.num_shards),
            process_shard
        )
        self.interference_patterns.extend(interference_patterns)
        
        # Registrar patrón
        self.pattern_registry[pattern_key] = {
            'pattern': pattern,
            'association': association,
            'timestamp': datetime.now().isoformat(),
            'interference_strength': np.mean([np.abs(p) for p in interference_patterns]),
            'original_shape': pattern.shape
        }
        
        self._manage_cache()
        
    def retrieve_distributed(self, query_pattern: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recupera patrones similares de forma distribuida.
        
        Args:
            query_pattern: Patrón de consulta
            
        Returns:
            Lista de patrones recuperados con metadatos
        """
        query_key = self._cache_key(query_pattern)
        
        # Usar caché para el patrón de consulta
        if query_key in self._wave_cache:
            query_wave = self._wave_cache[query_key]
        else:
            query_wave = self._encode_pattern(query_pattern)
            self._wave_cache[query_key] = query_wave
            
        conjugate_wave = self.phase_conjugator.conjugate(query_wave)
        
        # Recuperación paralela de fragmentos
        def process_reconstruction(shard_idx):
            shard = self.shards[shard_idx]
            reconstruction = shard.reconstruct(conjugate_wave)
            pattern = self._decode_pattern(reconstruction)
            
            # Calcular similitud usando caché
            sim_key = (query_key, self._cache_key(pattern))
            if sim_key in self._similarity_cache:
                similarity = self._similarity_cache[sim_key]
            else:
                similarity = self._compute_similarity(pattern, query_pattern)
                self._similarity_cache[sim_key] = similarity
                
            return {
                'pattern': pattern,
                'similarity': similarity,
                'reconstruction': reconstruction,
                'shard_idx': shard_idx
            }
            
        reconstructions = self._parallel_process(
            range(self.num_shards),
            process_reconstruction
        )
        
        # Procesar resultados
        for rec in reconstructions:
            rec['reconstruction_quality'] = np.mean(np.abs(rec['reconstruction']))
            rec['phase_coherence'] = np.angle(rec['reconstruction']).std()
            
        # Registrar recuperación
        self.retrieval_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_results': len(reconstructions),
            'max_similarity': max(r['similarity'] for r in reconstructions),
            'mean_quality': np.mean([r['reconstruction_quality'] for r in reconstructions]),
            'query_shape': query_pattern.shape
        })
        
        self._manage_cache()
        return sorted(reconstructions, key=lambda x: x['similarity'], reverse=True)
        
    # @lru_cache(maxsize=1000)  # Removed caching due to unhashable type error
    def _encode_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """
        Codifica un patrón en una onda compleja.
        
        Args:
            pattern: Patrón a codificar
            
        Returns:
            Onda codificada
        """
        # Redimensionar el patrón a 2D si es necesario
        if pattern.ndim == 1:
            # Calcular dimensiones objetivo basadas en las dimensiones de la placa
            target_size = int(np.sqrt(np.prod(self.dimensions)))
            pattern_2d = np.zeros((target_size, target_size))
            # Asegurar que el patrón cabe en las dimensiones objetivo
            pattern_flat = pattern.flatten()[:target_size * target_size]
            pattern_2d.flat[:len(pattern_flat)] = pattern_flat
        else:
            # Si el patrón ya es 2D, redimensionar a las dimensiones de la placa
            pattern_2d = np.zeros(self.dimensions)
            min_rows = min(pattern.shape[0], self.dimensions[0])
            min_cols = min(pattern.shape[1], self.dimensions[1])
            pattern_2d[:min_rows, :min_cols] = pattern[:min_rows, :min_cols]
            
        max_value = np.max(np.abs(pattern))
        if max_value == 0:
            raise ValueError("El patrón no puede ser todo ceros para la codificación.")
        normalized = pattern_2d / max_value
            
        # Aplicar FFT y asegurar dimensiones correctas
        wave = fft2(normalized)
        if wave.shape != self.dimensions:
            # Redimensionar si es necesario
            padded = np.zeros(self.dimensions, dtype=np.complex64)
            min_rows = min(wave.shape[0], self.dimensions[0])
            min_cols = min(wave.shape[1], self.dimensions[1])
            padded[:min_rows, :min_cols] = wave[:min_rows, :min_cols]
            wave = padded
            
        return wave.astype(np.complex64)
        
    # @lru_cache(maxsize=1000)
    def _decode_pattern(self, wave: np.ndarray) -> np.ndarray:
        """
        Decodifica una onda compleja en un patrón.
        
        Args:
            wave: Onda a decodificar
            
        Returns:
            Patrón decodificado
        """
        # Asegurar dimensiones correctas antes de IFFT
        if wave.shape != self.dimensions:
            padded = np.zeros(self.dimensions, dtype=np.complex64)
            min_rows = min(wave.shape[0], self.dimensions[0])
            min_cols = min(wave.shape[1], self.dimensions[1])
            padded[:min_rows, :min_cols] = wave[:min_rows, :min_cols]
            wave = padded
            
        pattern_2d = np.abs(ifft2(wave)).real
        
        # Si el patrón original era 1D, convertir de vuelta
        if self.dimensions[0] == self.dimensions[1]:
            # Calcular el tamaño original basado en las dimensiones de la placa
            original_size = int(np.sqrt(np.prod(self.dimensions)))
            return pattern_2d.flatten()[:original_size]
        return pattern_2d
        
    def _compute_interference(self, wave1: np.ndarray, wave2: np.ndarray) -> np.ndarray:
        """
        Calcula el patrón de interferencia entre dos ondas.
        
        Args:
            wave1: Primera onda
            wave2: Segunda onda
            
        Returns:
            Patrón de interferencia
        """
        return np.abs(wave1 + wave2) ** 2
        
    def _compute_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calcula la similitud entre dos patrones.
        
        Args:
            pattern1: Primer patrón
            pattern2: Segundo patrón
            
        Returns:
            Valor de similitud
        """
        # Asegurar mismas dimensiones
        if pattern1.shape != pattern2.shape:
            min_size = min(pattern1.size, pattern2.size)
            pattern1 = pattern1.flatten()[:min_size]
            pattern2 = pattern2.flatten()[:min_size]
            
        # Calcular correlación normalizada
        return np.corrcoef(pattern1.flatten(), pattern2.flatten())[0, 1]
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema de memoria.
        
        Returns:
            Dict con estadísticas
        """
        return {
            'num_patterns': len(self.pattern_registry),
            'num_shards': self.num_shards,
            'dimensions': self.dimensions,
            'total_interference_patterns': len(self.interference_patterns),
            'mean_interference_strength': np.mean([np.mean(np.abs(p)) for p in self.interference_patterns]) if self.interference_patterns else 0,
            'retrieval_history': self.retrieval_history[-5:],  # últimos 5 eventos
            'last_update': datetime.now().isoformat()
        }
        
    def clear_all(self) -> None:
        """Limpia toda la memoria"""
        for shard in self.shards:
            shard.clear()
        self.pattern_registry.clear()
        self.interference_patterns.clear()
        self.retrieval_history.clear()