import os
import numpy as np
import pandas as pd
import torch
from opo import NeuralNetwork
from evaluador_datos import cargar_banco_preguntas, obtener_embedding_gemini, obtener_embedding_groq, obtener_embedding_cohere
import matplotlib.pyplot as plt


def graficar_loss(modelo, id_pregunta):
    """Grafica la curva de aprendizaje del modelo."""
    plt.figure(figsize=(10, 5))
    plt.plot(modelo.loss_list, color='#7c6af7', linewidth=1.5, label='Train Loss')
    if modelo.val_loss_list and any(v != modelo.loss_list[0] for v in modelo.val_loss_list):
        plt.plot(modelo.val_loss_list, color='#f44336', linewidth=1.5, label='Val Loss', linestyle='--')
    plt.title(f'Curva de Aprendizaje - Pregunta {id_pregunta}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'loss_pregunta_{id_pregunta}.png')
    plt.close()
    print(f"📊 Gráfica guardada: loss_pregunta_{id_pregunta}.png")


def obtener_embedding(texto, stats=None):
    """
    Estrategia de embeddings en cascada:
    A) Gemini (principal)
    B) Groq (respaldo)
    C) Cohere (último recurso)
    """
    # Plan A: Gemini
    try:
        print("      🧠 [Plan A] Gemini...")
        embedding = obtener_embedding_gemini(texto)
        if stats is not None:
            stats["gemini"] += 1
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"      ⚠️ Gemini falló: {str(e)[:100]}")
        
        # Plan B: Groq
        try:
            print("      🟡 [Plan B] Groq...")
            embedding = obtener_embedding_groq(texto)
            if stats is not None:
                stats["groq"] += 1
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"      ⚠️ Groq falló: {str(e)[:100]}")
            
            # Plan C: Cohere
            try:
                print("      🟠 [Plan C] Cohere...")
                embedding = obtener_embedding_cohere(texto)
                if stats is not None:
                    stats["cohere"] += 1
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                print(f"      ❌ Todos los servicios fallaron: {str(e)[:100]}")
                if stats is not None:
                    stats["fallos"] += 1
                print("      🔴 Usando vector de ceros como último recurso")
                return np.zeros(768, dtype=np.float32)


def cargar_dataset(id_pregunta):
    """Carga el CSV generado y convierte respuestas a embeddings."""
    archivo = f"data/dataset_pregunta_{id_pregunta}.csv"

    if not os.path.exists(archivo):
        raise FileNotFoundError(f"❌ No existe {archivo}. Corre primero generar_dataset.py")

    df = pd.read_csv(archivo)
    print(f"📂 Cargando {len(df)} ejemplos de {archivo}...")

    X_list, Y_list = [], []
    
    # Contadores para estadísticas
    stats = {"gemini": 0, "groq": 0, "cohere": 0, "fallos": 0}

    for i, row in df.iterrows():
        texto = str(row["respuesta"])
        nota = float(row["nota"])

        print(f"   [{i+1}/{len(df)}] Generando embedding...")
        embedding = obtener_embedding(texto, stats)

        X_list.append(embedding)
        Y_list.append([nota / 10.0])

    # Mostrar estadísticas finales
    print(f"\n📊 Estadísticas de embeddings:")
    print(f"   🧠 Gemini: {stats['gemini']}")
    print(f"   🟡 Groq:   {stats['groq']}")
    print(f"   🟠 Cohere: {stats['cohere']}")
    print(f"   🔴 Fallos: {stats['fallos']}")
    print(f"   ✅ Total exitosos: {len(X_list) - stats['fallos']}/{len(X_list)}")
    
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def calcular_metricas(modelo, X, Y, umbral=0.5):
    """Calcula y muestra métricas de evaluación del modelo."""
    predicciones = modelo.predict(X)

    mse = np.mean((Y - predicciones) ** 2)

    pred_bin = (predicciones >= umbral).astype(int)
    real_bin = (Y >= umbral).astype(int)

    tp = np.sum((pred_bin == 1) & (real_bin == 1))
    tn = np.sum((pred_bin == 0) & (real_bin == 0))
    fp = np.sum((pred_bin == 1) & (real_bin == 0))
    fn = np.sum((pred_bin == 0) & (real_bin == 1))

    accuracy = (tp + tn) / len(Y) * 100 if len(Y) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "="*50)
    print("         MÉTRICAS DEL MODELO ENTRENADO       ")
    print("="*50)
    print(f"  📉 MSE (Error cuadrático medio) : {mse:.6f}")
    print(f"  🎯 Accuracy                     : {accuracy:.2f}%")
    print(f"  🎯 Precision                    : {precision:.4f}")
    print(f"  🎯 Recall                       : {recall:.4f}")
    print(f"  🎯 F1 Score                     : {f1:.4f}")
    print("="*50)

    return mse, accuracy, precision, recall, f1


def entrenar_evaluador(id_pregunta, epochs=2000, lr=0.001):
    """Entrena un evaluador neuronal para una pregunta específica."""
    print(f"\n{'='*60}")
    print(f"  🚀 Entrenando IA Semántica para pregunta {id_pregunta}")
    print(f"{'='*60}")

    # Cargar datos
    X_train, Y_train = cargar_dataset(id_pregunta)
    print(f"✅ Dataset cargado: {len(X_train)} ejemplos con embeddings de {X_train.shape[1]} dimensiones")

    # Crear modelo (instancia de NeuralNetwork, NO torch.nn)
    modelo = NeuralNetwork()
    
    # Agregar capas - inputs_size solo necesario en la primera
    modelo.add_layer(num_neurons=256, inputs_size=X_train.shape[1], activation='relu', dropout_rate=0.2)
    modelo.add_layer(num_neurons=128, activation='relu', dropout_rate=0.2)
    modelo.add_layer(num_neurons=64, activation='relu', dropout_rate=0.1)
    modelo.add_layer(num_neurons=1, activation='sigmoid')

    # Tamaño de lote adaptativo
    tamanio_lote = min(8, len(X_train))

    print(f"\n🎯 Iniciando entrenamiento...")
    print(f"   Épocas: {epochs} | Learning rate: {lr} | Batch size: {tamanio_lote}")
    
    # Entrenar modelo
    modelo.train_model(
        X_train, Y_train,
        learning_rate=lr,
        epochs=epochs,
        patience=300,
        lr_factor=0.5,
        lr_patience=100,
        batch_size=tamanio_lote,
        val_split=0.2,
        clip_norm=1.0
    )

    # Calcular métricas
    calcular_metricas(modelo, X_train, Y_train)

    # Guardar modelo
    nombre_archivo = f"cerebro_pregunta_{id_pregunta}.json"
    modelo.save(nombre_archivo)
    
    # Graficar pérdida
    graficar_loss(modelo, id_pregunta)

    return modelo


if __name__ == "__main__":
    print("="*60)
    print("  🧠 SISTEMA DE CALIBRACIÓN POR RED NEURONAL SEMÁNTICA")
    print("="*60)

    banco = cargar_banco_preguntas()
    total_preguntas = len(banco.keys())
    
    print(f"\n📚 Preguntas encontradas: {total_preguntas}")
    
    for idx, id_p in enumerate(banco.keys(), 1):
        print(f"\n{'🔄'*30}")
        print(f"  Procesando pregunta {idx}/{total_preguntas}: ID {id_p}")
        print(f"{'🔄'*30}")
        
        try:
            entrenar_evaluador(id_p, epochs=2000, lr=0.001)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            continue
        except Exception as e:
            print(f"❌ Error entrenando pregunta {id_p}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("  ✅ ¡PROCESO TERMINADO CON ÉXITO!")
    print("="*60)