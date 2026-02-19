import numpy as np
from typing import Tuple, Optional


class SelfAttention:
    """
    Implementação de Self-Attention - Mecanismo fundamental de Transformers.
    
    A auto-atenção permite que cada posição numa sequência se relacione com
    todas as outras posições, aprendendo pesos de importância dinamicamente.
    
    Attributes:
        None (implementação sem parâmetros treináveis nesta versão básica)
    """
    
    def __init__(self) -> None:
        """Inicializa o módulo de Self-Attention."""
        pass
    
    def _validate_input_dimensions(
        self, 
        query_matrix: np.ndarray, 
        key_matrix: np.ndarray, 
        value_matrix: np.ndarray
    ) -> None:
        """
        Valida as dimensões de entrada para atenção.
        
        Args:
            query_matrix: Matriz de queries com shape (seq_len, feature_dim)
            key_matrix: Matriz de keys com shape (seq_len, feature_dim)
            value_matrix: Matriz de values com shape (seq_len, output_dim)
            
        Raises:
            ValueError: Se as dimensões forem incompatíveis
            TypeError: Se os inputs não forem numpy arrays
        """
        # Validação de tipos
        if not all(isinstance(matrix, np.ndarray) for matrix in [query_matrix, key_matrix, value_matrix]):
            raise TypeError("Q, K e V devem ser numpy arrays")
        
        # Validação de dimensionalidade (2D)
        if query_matrix.ndim != 2 or key_matrix.ndim != 2 or value_matrix.ndim != 2:
            raise ValueError("Q, K e V devem ser matrizes 2D (seq_length, feature_dim)")
        
        # Validação de compatibilidade de dimensões
        seq_length_q, feature_dim_q = query_matrix.shape
        seq_length_k, feature_dim_k = key_matrix.shape
        seq_length_v, _ = value_matrix.shape
        
        if feature_dim_q != feature_dim_k:
            raise ValueError(
                f"Dimensões incompatíveis: Q tem {feature_dim_q} features, "
                f"mas K tem {feature_dim_k} features. Devem ser iguais."
            )
        
        if seq_length_k != seq_length_v:
            raise ValueError(
                f"Tamanhos de sequência incompatíveis: K tem {seq_length_k} posições, "
                f"mas V tem {seq_length_v} posições. Devem ser iguais."
            )
        
        if seq_length_q == 0 or seq_length_k == 0:
            raise ValueError("Sequências não podem estar vazias (seq_length > 0)")
    
    def _softmax(self, scores_matrix: np.ndarray) -> np.ndarray:
        """
        Calcula softmax com estabilidade numérica.
        
        Implementação segura que subtrai o máximo antes de exponenciar,
        prevenindo overflow/underflow numérico em dados de ponto flutuante.
        
        Args:
            scores_matrix: Matriz de scores com shape (seq_len, seq_len)
            
        Returns:
            np.ndarray: Matriz de probabilidades com shape (seq_len, seq_len),
                       onde cada linha soma a 1.0
                       
        Formula:
            softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        """
        # Subtrai máximo para estabilidade numérica
        shifted_scores = scores_matrix - np.max(scores_matrix, axis=1, keepdims=True)
        
        # Calcula exponenciais
        exp_scores = np.exp(shifted_scores)
        
        # Normaliza por soma (softmax)
        probability_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probability_weights
    
    def forward(
        self, 
        query_matrix: np.ndarray, 
        key_matrix: np.ndarray, 
        value_matrix: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula Self-Attention: Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
        
        Args:
            query_matrix: Matriz Query com shape (seq_length, feature_dim)
            key_matrix: Matriz Key com shape (seq_length, feature_dim)
            value_matrix: Matriz Value com shape (seq_length, output_dim)
            mask: Máscara opcional com shape (seq_length, seq_length), 
                  onde False indica posições a mascarar (padrão: None)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - output: Saída com shape (seq_length, output_dim)
                - attention_weights: Pesos de atenção com shape (seq_length, seq_length)
                
        Raises:
            ValueError: Se as dimensões de entrada forem inválidas
            TypeError: Se os inputs não forem numpy arrays
            
        Algorithm:
            1. Calcula scores: Q @ K^T
            2. Escala: scores / sqrt(d_k)
            3. Aplica máscara (opcional)
            4. Softmax para obter pesos de atenção
            5. Multiplica pela matriz Value
        """
        # Validação de entrada
        self._validate_input_dimensions(query_matrix, key_matrix, value_matrix)
        
        # Dimensão das chaves para escaling
        key_dimension = key_matrix.shape[1]
        
        # Passo 1: Calcula compatibility scores (Q @ K^T)
        compatibility_scores = np.dot(query_matrix, key_matrix.T)
        
        # Passo 2: Escala pelos sqrt da dimensão das chaves
        # Isso evita que os valores fiquem muito grandes ou pequenos
        scaled_scores = compatibility_scores / np.sqrt(key_dimension)
        
        # Passo 3: Aplica máscara se fornecida (útil para atenção causal)
        if mask is not None:
            if mask.shape != scaled_scores.shape:
                raise ValueError(
                    f"Máscara tem shape {mask.shape}, mas expected {scaled_scores.shape}"
                )
            # Define posições mascaradas para -inf (resultarão em peso 0 após softmax)
            scaled_scores = np.where(mask, scaled_scores, -np.inf)
        
        # Passo 4: Aplica softmax para converter scores em pesos de atenção
        attention_weights = self._softmax(scaled_scores)
        
        # Passo 5: Multiplica pelos valores para computar saída ponderada
        attention_output = np.dot(attention_weights, value_matrix)
        
        return attention_output, attention_weights
