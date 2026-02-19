"""
Exemplo prático de uso de Self-Attention para processamento de sequências.

Demonstra:
1. Atenção básica com dados simples
2. Atenção com máscara causal (como em GPT)
3. Análise dos pesos de atenção
"""

import numpy as np
from attention import SelfAttention


def example_1_basic_attention():
    """Exemplo 1: Self-Attention básica com matriz simples."""
    print("=" * 70)
    print("EXEMPLO 1: Self-Attention Básica")
    print("=" * 70)
    
    # Cria um módulo de atenção
    attention = SelfAttention()
    
    # Dados de entrada simples:
    # - 2 posições na sequência
    # - 3 dimensões de features
    query_matrix = np.array([
        [1.0, 0.0, 1.0],  # Posição 0: "objeto"
        [0.0, 1.0, 0.0]   # Posição 1: "verbo"
    ], dtype=np.float32)
    
    key_matrix = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    
    value_matrix = np.array([
        [1.0, 2.0],       # Valor para posição 0
        [3.0, 4.0]        # Valor para posição 1
    ], dtype=np.float32)
    
    # Calcula atenção
    output, attention_weights = attention.forward(query_matrix, key_matrix, value_matrix)
    
    print("\nPesos de Atenção:")
    print("Mostra como cada posição atende às outras posições")
    print(attention_weights)
    print(f"\nSoma de cada linha: {attention_weights.sum(axis=1)}")
    print("✓ Cada linha soma a 1.0 (propriedade de softmax)")
    
    print("\nSaída (valores agregados):")
    print(output)
    print(f"Shape: {output.shape}")


def example_2_causal_mask():
    """Exemplo 2: Self-Attention com máscara causal (como em GPT)."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Self-Attention com Máscara Causal")
    print("=" * 70)
    print("\nMáscara causal: cada posição só ve posições anteriores + ela mesma")
    print("Útil para: GPT, geração auto-regressiva, modelos causais")
    
    attention = SelfAttention()
    
    # Sequência de 4 tokens
    seq_length = 4
    feature_dim = 8
    
    query_matrix = np.random.randn(seq_length, feature_dim)
    key_matrix = np.random.randn(seq_length, feature_dim)
    value_matrix = np.random.randn(seq_length, feature_dim)
    
    # Cria máscara causal (triangular inferior)
    causal_mask = np.tril(np.ones((seq_length, seq_length), dtype=bool))
    
    print(f"\nMáscara Causal ({seq_length}x{seq_length}):")
    print("(True = posição pode atender, False = máscara as posições)")
    for i, row in enumerate(causal_mask):
        print(f"Posição {i} atende a: {np.where(row)[0].tolist()}")
    
    # Sem máscara (para comparação)
    output_no_mask, weights_no_mask = attention.forward(query_matrix, key_matrix, value_matrix)
    
    # Com máscara causal
    output_with_mask, weights_with_mask = attention.forward(
        query_matrix, key_matrix, value_matrix, mask=causal_mask
    )
    
    print("\nEfeito da Máscara:")
    print(f"Posição 0 (sem máscara): atende a {np.count_nonzero(weights_no_mask[0])} posições")
    print(f"Posição 0 (com máscara): atende a {np.count_nonzero(weights_with_mask[0])} posição")
    print()
    print(f"Posição 2 (sem máscara): atende a {np.count_nonzero(weights_no_mask[2])} posições")
    print(f"Posição 2 (com máscara): atende a {np.count_nonzero(weights_with_mask[2])} posições")
    
    print("\nPesos da Posição 0 COM máscara causal:")
    print(f"{weights_with_mask[0]}")
    print("✓ Apenas posição 0 tem peso, futuro é bloqueado")


def example_3_different_dimensions():
    """Exemplo 3: Diferentes dimensionalidades de entrada/saída."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Diferentes Dimensões")
    print("=" * 70)
    
    attention = SelfAttention()
    
    configs = [
        ("Pequeno", 5, 16, 8),
        ("Médio", 50, 128, 64),
        ("Grande", 200, 512, 256),
    ]
    
    print("\nExperimentando com diferentes tamanhos:\n")
    print(f"{'Tamanho':<10} | {'Seq Len':<8} | {'Features':<10} | {'Output':<8} | {'Output Shape':<15}")
    print("-" * 65)
    
    for name, seq_len, feat_dim, out_dim in configs:
        query = np.random.randn(seq_len, feat_dim)
        key = np.random.randn(seq_len, feat_dim)
        value = np.random.randn(seq_len, out_dim)
        
        output, weights = attention.forward(query, key, value)
        
        print(f"{name:<10} | {seq_len:<8} | {feat_dim:<10} | {out_dim:<8} | {str(output.shape):<15}")
        
        # Verifica propriedades
        assert output.shape == (seq_len, out_dim), "Shape incorreto!"
        assert weights.shape == (seq_len, seq_len), "Weights com shape errado!"
        assert np.allclose(weights.sum(axis=1), 1.0), "Softmax não soma a 1!"
    
    print("\n✓ Todos os testes passaram!")


def example_4_attention_analysis():
    """Exemplo 4: Análise detalhada dos pesos de atenção."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Análise de Pesos de Atenção")
    print("=" * 70)
    
    attention = SelfAttention()
    
    # Dados pequenos para visualizar
    query = np.array([
        [1.0, 0.0],  # Token 0: "o"
        [0.0, 1.0],  # Token 1: "gato"
        [1.0, 1.0],  # Token 2: "dorme"
    ], dtype=np.float32)
    
    key = query.copy()
    value = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
        [2.5, 2.5],
    ], dtype=np.float32)
    
    output, weights = attention.forward(query, key, value)
    
    print("\nPesos de Atenção (normalizado com softmax):")
    print("Como cada token atende aos outros tokens:\n")
    
    token_names = ["[o]", "[gato]", "[dorme]"]
    
    for i, token in enumerate(token_names):
        print(f"Token {i} {token:<10}: ", end="")
        for j, weight in enumerate(weights[i]):
            bar = "█" * int(weight * 40)  # Visualização em barra
            print(f"{token_names[j]}: {weight:.3f} {bar:<40}", end=" | ")
        print()
    
    print("\nInterpretação:")
    print("- Quanto maior o peso, mais atenção para aquele token")
    print("- Todos os pesos de uma linha somam a 1.0")
    print("- A atenção é aprendida durante o treinamento")


def example_5_error_handling():
    """Exemplo 5: Tratamento de erros e validação."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Tratamento de Erros e Validação")
    print("=" * 70)
    
    attention = SelfAttention()
    
    print("\n1. Dimensões incompatíveis (Query vs Key):")
    try:
        query = np.random.randn(5, 8)    # 8 features
        key = np.random.randn(5, 16)     # 16 features ❌ Incompatível!
        value = np.random.randn(5, 8)
        attention.forward(query, key, value)
    except ValueError as e:
        print(f"   ✓ Erro capturado: {e}")
    
    print("\n2. Input que não é numpy array:")
    try:
        query = [[1.0, 2.0], [3.0, 4.0]]  # Lista, não array ❌
        key = np.array([[1.0, 2.0], [3.0, 4.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        attention.forward(query, key, value)
    except TypeError as e:
        print(f"   ✓ Erro capturado: {e}")
    
    print("\n3. Sequência vazia:")
    try:
        query = np.random.randn(0, 8)  # Seq_length = 0 ❌
        key = np.random.randn(0, 8)
        value = np.random.randn(0, 8)
        attention.forward(query, key, value)
    except ValueError as e:
        print(f"   ✓ Erro capturado: {e}")
    
    print("\n4. Máscara com dimensão incorreta:")
    try:
        query = np.random.randn(5, 8)
        key = np.random.randn(5, 8)
        value = np.random.randn(5, 8)
        wrong_mask = np.ones((3, 3), dtype=bool)  # Shape errado ❌
        attention.forward(query, key, value, mask=wrong_mask)
    except ValueError as e:
        print(f"   ✓ Erro capturado: {e}")
    
    print("\n✓ Todos os erros foram capturados corretamente!")


def main():
    """Executa todos os exemplos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "EXEMPLOS DE SELF-ATTENTION" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    
    example_1_basic_attention()
    example_2_causal_mask()
    example_3_different_dimensions()
    example_4_attention_analysis()
    example_5_error_handling()
    
    print("\n" + "=" * 70)
    print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
    print("=" * 70)
    print("\nPróximos passos:")
    print("- Estude o código em attention.py")
    print("- Execute os testes: python test_attention.py")
    print("- Integrate em seus projetos")
    print()


if __name__ == "__main__":
    main()
