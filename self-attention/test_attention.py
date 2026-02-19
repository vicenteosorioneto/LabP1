import numpy as np
import time
from typing import Tuple
from attention import SelfAttention


class TestSelfAttention:
    """Suite de testes para validação da implementação de Self-Attention."""
    
    def __init__(self):
        """Inicializa o executor de testes."""
        self.tests_passed = 0
        self.tests_failed = 0
    
    def assert_equal(self, actual: np.ndarray, expected: np.ndarray, 
                    test_name: str, tolerance: float = 1e-10) -> bool:
        """
        Compara dois arrays com tolerância numérica.
        
        Args:
            actual: Valor obtido
            expected: Valor esperado
            test_name: Nome do teste
            tolerance: Tolerância numérica para comparação
            
        Returns:
            bool: True se teste passou, False caso contrário
        """
        try:
            if isinstance(actual, (int, float)):
                passed = abs(actual - expected) < tolerance
            else:
                passed = np.allclose(actual, expected, atol=tolerance)
            
            if passed:
                print(f"✓ {test_name}")
                self.tests_passed += 1
            else:
                print(f"✗ {test_name}")
                print(f"  Esperado: {expected}")
                print(f"  Obtido: {actual}")
                self.tests_failed += 1
            
            return passed
        except Exception as e:
            print(f"✗ {test_name} - Erro: {e}")
            self.tests_failed += 1
            return False
    
    def assert_raises(self, func, exception_type: type, test_name: str) -> bool:
        """
        Verifica se uma função levanta a exceção esperada.
        
        Args:
            func: Função a executar
            exception_type: Tipo de exceção esperada
            test_name: Nome do teste
            
        Returns:
            bool: True se exceção foi levantada, False caso contrário
        """
        try:
            func()
            print(f"✗ {test_name} - Nenhuma exceção foi levantada")
            self.tests_failed += 1
            return False
        except exception_type:
            print(f"✓ {test_name}")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"✗ {test_name} - Exceção incorreta: {type(e).__name__}")
            self.tests_failed += 1
            return False
    
    def run_all_tests(self) -> None:
        """Executa todos os testes da suite."""
        print("=" * 70)
        print("TESTES DE SELF-ATTENTION")
        print("=" * 70)
        
        # Testes básicos
        print("\n[TESTES BÁSICOS]")
        self.test_basic_forward_pass()
        self.test_output_shape()
        self.test_attention_weights_sum_to_one()
        self.test_attention_weights_positive()
        
        # Testes com diferentes dimensões
        print("\n[TESTES COM DIFERENTES DIMENSÕES]")
        self.test_different_dimensions()
        self.test_small_sequence()
        self.test_large_sequence()
        
        # Testes com máscaras
        print("\n[TESTES COM MÁSCARAS]")
        self.test_causal_mask()
        self.test_mask_validation()
        
        # Testes de validação de entrada
        print("\n[TESTES DE VALIDAÇÃO]")
        self.test_dimension_validation()
        self.test_type_validation()
        self.test_empty_sequence()
        
        # Benchmark
        print("\n[BENCHMARK DE PERFORMANCE]")
        self.benchmark_performance()
        
        # Resumo
        self.print_summary()
    
    def test_basic_forward_pass(self) -> None:
        """Testa um forward pass básico com valores simples."""
        att = SelfAttention()
        
        # Matrizes simples e conhecidas
        query_matrix = np.array([[1.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]], dtype=np.float32)
        key_matrix = np.array([[1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]], dtype=np.float32)
        value_matrix = np.array([[1.0, 2.0],
                                [3.0, 4.0]], dtype=np.float32)
        
        attention_output, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
        
        self.assert_equal(attention_output.shape, (2, 2), 
                         "Forward pass retorna shape correto")
        self.assert_equal(attention_weights.shape, (2, 2),
                         "Pesos de atenção têm shape correto")
    
    def test_output_shape(self) -> None:
        """Testa se as dimensões de saída estão corretas para múltiplos tamanhos."""
        att = SelfAttention()
        
        seq_lengths = [1, 5, 10, 50, 100]
        feature_dims = [8, 16, 32, 64]
        output_dims = [4, 8, 16]
        
        for seq_len in seq_lengths:
            for feat_dim in feature_dims:
                for out_dim in output_dims:
                    query_matrix = np.random.randn(seq_len, feat_dim)
                    key_matrix = np.random.randn(seq_len, feat_dim)
                    value_matrix = np.random.randn(seq_len, out_dim)
                    
                    attention_output, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
                    
                    self.assert_equal(attention_output.shape, (seq_len, out_dim),
                                     f"Output shape correto ({seq_len}, {feat_dim}) -> ({seq_len}, {out_dim})")
                    self.assert_equal(attention_weights.shape, (seq_len, seq_len),
                                     f"Weights shape correto ({seq_len}, {seq_len})")
    
    def test_attention_weights_sum_to_one(self) -> None:
        """Testa que cada linha dos pesos soma a 1.0 (propriedade de softmax)."""
        att = SelfAttention()
        
        for seq_length in [5, 10, 20]:
            feature_dim = 16
            query_matrix = np.random.randn(seq_length, feature_dim)
            key_matrix = np.random.randn(seq_length, feature_dim)
            value_matrix = np.random.randn(seq_length, feature_dim)
            
            _, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
            
            row_sums = np.sum(attention_weights, axis=1)
            expected_sums = np.ones(seq_length)
            
            self.assert_equal(row_sums, expected_sums,
                             f"Pesos somam a 1 (seq_len={seq_length})", tolerance=1e-6)
    
    def test_attention_weights_positive(self) -> None:
        """Testa que todos os pesos de atenção são não-negativos."""
        att = SelfAttention()
        
        query_matrix = np.random.randn(10, 16)
        key_matrix = np.random.randn(10, 16)
        value_matrix = np.random.randn(10, 16)
        
        _, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
        
        all_positive = np.all(attention_weights >= 0)
        
        if all_positive:
            print("✓ Todos os pesos de atenção são não-negativos")
            self.tests_passed += 1
        else:
            negatives = np.sum(attention_weights < 0)
            print(f"✗ {negatives} pesos negativos encontrados")
            self.tests_failed += 1
    
    def test_different_dimensions(self) -> None:
        """Testa com diferentes combinações de dimensões."""
        att = SelfAttention()
        
        test_cases = [
            (5, 8, 4),      # seq_len=5, feat_dim=8, out_dim=4
            (10, 64, 32),   # seq_len=10, feat_dim=64, out_dim=32
            (1, 128, 64),   # seq_len=1, feat_dim=128, out_dim=64
            (100, 16, 16),  # seq_len=100, feat_dim=16, out_dim=16
        ]
        
        for seq_len, feat_dim, out_dim in test_cases:
            query_matrix = np.random.randn(seq_len, feat_dim)
            key_matrix = np.random.randn(seq_len, feat_dim)
            value_matrix = np.random.randn(seq_len, out_dim)
            
            attention_output, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
            
            self.assert_equal(attention_output.shape, (seq_len, out_dim),
                             f"Output dimensions ({seq_len}, {feat_dim}, {out_dim})")
    
    def test_small_sequence(self) -> None:
        """Testa com sequência mínima (tamanho 1)."""
        att = SelfAttention()
        
        query_matrix = np.array([[5.0, 3.0, 2.0]])
        key_matrix = np.array([[5.0, 3.0, 2.0]])
        value_matrix = np.array([[1.0, 2.0, 3.0]])
        
        attention_output, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
        
        # Com seq_len=1, o peso deve ser exatamente 1.0
        self.assert_equal(attention_weights[0, 0], 1.0,
                         "Peso de uma posição única é 1.0", tolerance=1e-10)
        
        # Output deve ser exatamente o value
        self.assert_equal(attention_output, value_matrix,
                         "Output com seq_len=1 é exatamente o value", tolerance=1e-10)
    
    def test_large_sequence(self) -> None:
        """Testa com sequência grande."""
        att = SelfAttention()
        
        seq_length = 1000
        feature_dim = 128
        
        query_matrix = np.random.randn(seq_length, feature_dim)
        key_matrix = np.random.randn(seq_length, feature_dim)
        value_matrix = np.random.randn(seq_length, feature_dim)
        
        attention_output, attention_weights = att.forward(query_matrix, key_matrix, value_matrix)
        
        self.assert_equal(attention_output.shape, (seq_length, feature_dim),
                         f"Large sequence ({seq_length} tokens) processada corretamente")
    
    def test_causal_mask(self) -> None:
        """Testa máscara causal (atenção apenas para posições anteriores)."""
        att = SelfAttention()
        
        seq_length = 5
        feature_dim = 8
        
        query_matrix = np.random.randn(seq_length, feature_dim)
        key_matrix = np.random.randn(seq_length, feature_dim)
        value_matrix = np.random.randn(seq_length, feature_dim)
        
        # Máscara causal: triangular inferior
        causal_mask = np.tril(np.ones((seq_length, seq_length), dtype=bool))
        
        attention_output, attention_weights = att.forward(
            query_matrix, key_matrix, value_matrix, mask=causal_mask
        )
        
        # Verifica que posições futuras têm peso zero
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                self.assert_equal(attention_weights[i, j], 0.0,
                                 f"Posição {i} não atende à posição futura {j}", tolerance=1e-10)
    
    def test_mask_validation(self) -> None:
        """Testa validação de máscara com dimensões incorretas."""
        att = SelfAttention()
        
        query_matrix = np.random.randn(5, 8)
        key_matrix = np.random.randn(5, 8)
        value_matrix = np.random.randn(5, 8)
        wrong_mask = np.ones((3, 3), dtype=bool)  # Dimensão incorreta
        
        def call_with_wrong_mask():
            att.forward(query_matrix, key_matrix, value_matrix, mask=wrong_mask)
        
        self.assert_raises(call_with_wrong_mask, ValueError,
                          "Máscara com dimensão incorreta levanta ValueError")
    
    def test_dimension_validation(self) -> None:
        """Testa validação de dimensões incompatíveis."""
        att = SelfAttention()
        
        # Q e K com dimensões diferentes
        query_matrix = np.random.randn(5, 8)
        key_matrix = np.random.randn(5, 16)  # Dimensão diferente!
        value_matrix = np.random.randn(5, 8)
        
        def call_with_wrong_dims():
            att.forward(query_matrix, key_matrix, value_matrix)
        
        self.assert_raises(call_with_wrong_dims, ValueError,
                          "Dimensões incompatíveis (Q vs K) levantam ValueError")
    
    def test_type_validation(self) -> None:
        """Testa validação de tipos (deve ser numpy array)."""
        att = SelfAttention()
        
        # Lista em vez de numpy array
        query_matrix = [[1.0, 2.0], [3.0, 4.0]]
        key_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        value_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        def call_with_wrong_type():
            att.forward(query_matrix, key_matrix, value_matrix)
        
        self.assert_raises(call_with_wrong_type, TypeError,
                          "Input que não é numpy array levanta TypeError")
    
    def test_empty_sequence(self) -> None:
        """Testa validação de sequências vazias."""
        att = SelfAttention()
        
        query_matrix = np.random.randn(0, 8)  # Sequência vazia!
        key_matrix = np.random.randn(0, 8)
        value_matrix = np.random.randn(0, 8)
        
        def call_with_empty_seq():
            att.forward(query_matrix, key_matrix, value_matrix)
        
        self.assert_raises(call_with_empty_seq, ValueError,
                          "Sequência vazia levanta ValueError")
    
    def benchmark_performance(self) -> None:
        """Executa benchmark de performance para diferentes tamanhos."""
        att = SelfAttention()
        
        test_configs = [
            ("Pequeno", 10, 64),
            ("Médio", 100, 256),
            ("Grande", 500, 512),
            ("Muito Grande", 1000, 1024),
        ]
        
        print("\nTamanho      | Seq Len | Features | Tempo (ms) | Ops/sec")
        print("-" * 60)
        
        for name, seq_length, feature_dim in test_configs:
            query_matrix = np.random.randn(seq_length, feature_dim)
            key_matrix = np.random.randn(seq_length, feature_dim)
            value_matrix = np.random.randn(seq_length, feature_dim)
            
            # Warmup
            att.forward(query_matrix, key_matrix, value_matrix)
            
            # Timed runs
            num_runs = 10
            start_time = time.time()
            for _ in range(num_runs):
                att.forward(query_matrix, key_matrix, value_matrix)
            elapsed_time = (time.time() - start_time) * 1000 / num_runs
            
            # Calcula operações (aproximado: 2*seq^2*feat + seq^2*feat = 3*seq^2*feat)
            ops = 3 * seq_length ** 2 * feature_dim * num_runs / (time.time() - start_time) / 1e9
            
            print(f"{name:8} | {seq_length:7} | {feature_dim:8} | {elapsed_time:9.3f} | {ops:8.2f}G")
    
    def print_summary(self) -> None:
        """Imprime resumo dos testes."""
        total_tests = self.tests_passed + self.tests_failed
        percentage = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 70)
        print(f"RESULTADO: {self.tests_passed}/{total_tests} testes passaram ({percentage:.1f}%)")
        print("=" * 70)
        
        if self.tests_failed == 0:
            print("✓ Todos os testes passaram com sucesso!")
        else:
            print(f"✗ {self.tests_failed} teste(s) falharam")


def main() -> None:
    """Função principal para executar todos os testes."""
    test_suite = TestSelfAttention()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
