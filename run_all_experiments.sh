#!/bin/bash

# Script para executar todos os experimentos de fusão
# Cada modelo será executado em sequência

set -e  # Para no primeiro erro

echo "=========================================="
echo "INICIANDO TODOS OS EXPERIMENTOS"
echo "=========================================="
echo ""

# Lista de scripts para executar
scripts=(
    "src/scripts/main_encoder.py"
    "src/scripts/main_autoencoder.py"
    "src/scripts/main_vae.py"
    "src/scripts/main_encoder_fusion_inside.py"
    "src/scripts/main_autoencoder_fusion_inside.py"
    "src/scripts/main_vae_fusion_inside.py"
)

# Contador de progresso
total=${#scripts[@]}
current=0

# Executa cada script
for script in "${scripts[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total] Executando: $script"
    echo "=========================================="
    echo ""

    # Executa o script e captura o código de saída
    if python "$script"; then
        echo ""
        echo "✓ $script concluído com sucesso"
    else
        echo ""
        echo "✗ ERRO ao executar $script"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "TODOS OS EXPERIMENTOS CONCLUÍDOS!"
echo "=========================================="
echo ""
echo "Resultados salvos em:"
echo "  - results/encoder/"
echo "  - results/autoencoder/"
echo "  - results/vae/"
echo "  - results/encoder_fusion_inside/"
echo "  - results/autoencoder_fusion_inside/"
echo "  - results/vae_fusion_inside/"
