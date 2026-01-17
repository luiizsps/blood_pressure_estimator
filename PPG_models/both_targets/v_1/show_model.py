from tensorflow import keras

# Carregar o modelo sem compilar (ignora métricas e loss customizadas)
model = keras.models.load_model('best_bp_model.h5', compile=False)

# Exibir o summary do modelo
print("=" * 70)
print("SUMMARY DO MODELO")
print("=" * 70)
model.summary()

# Informações adicionais
print("\n" + "=" * 70)
print("INFORMAÇÕES ADICIONAIS")
print("=" * 70)
print(f"Total de camadas: {len(model.layers)}")
print(f"Parâmetros treináveis: {model.count_params():,}")

# Exibir configuração do modelo
print("\n" + "=" * 70)
print("CONFIGURAÇÃO DO MODELO")
print("=" * 70)
config = model.get_config()
print(f"Nome do modelo: {config.get('name', 'N/A')}")

# Exibir input e output shapes
print("\n" + "=" * 70)
print("INPUT/OUTPUT SHAPES")
print("=" * 70)
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")