"""
entrenar_vocabulario_pytorch.py (CON REANUDACIÓN)
Word2Vec con PyTorch - Guarda y reanuda entrenamiento
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# ⚡ CONFIGURACIÓN
CORPUS_DIR = "data/corpus"
MODEL_PATH = "data/vocabulario_pytorch.pt"
VOCAB_PATH = "data/vocabulario_pytorch.json"
EMBEDDING_DIM = 150
WINDOW_SIZE = 4
MIN_COUNT = 5
BATCH_SIZE = 1024
EPOCHS = 8
LEARNING_RATE = 0.001


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def cargar_textos():
    archivos = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')]
    if not archivos:
        print("❌ No hay archivos .txt en data/corpus/")
        return []
    
    todas_palabras = []
    print(f"\n📚 Cargando {len(archivos)} archivos...")
    for archivo in archivos:
        ruta = os.path.join(CORPUS_DIR, archivo)
        print(f"   📄 {archivo}")
        with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
            texto = f.read()
        texto_limpio = limpiar_texto(texto)
        todas_palabras.extend(texto_limpio.split())
    
    print(f"   ✅ {len(todas_palabras)} palabras cargadas")
    return todas_palabras


def crear_vocabulario(palabras):
    contador = Counter(palabras)
    palabras_filtradas = [p for p in palabras if contador[p] >= MIN_COUNT]
    vocabulario = sorted(set(palabras_filtradas))
    palabra_a_idx = {p: i for i, p in enumerate(vocabulario)}
    idx_a_palabra = {i: p for p, i in palabra_a_idx.items()}
    
    print(f"   📖 Vocabulario: {len(vocabulario)} palabras")
    return vocabulario, palabra_a_idx, idx_a_palabra


def generar_pares(palabras, palabra_a_idx):
    """Genera pares - versión rápida (1 de cada 3)"""
    pares = []
    total = len(palabras)
    contador = 0
    
    print(f"   🔗 Generando pares (modo rápido)...")
    for i, palabra in enumerate(palabras):
        if i % 200000 == 0:
            print(f"      {i}/{total} ({(i/total)*100:.0f}%)")
        
        if palabra not in palabra_a_idx:
            continue
        
        centro = palabra_a_idx[palabra]
        inicio = max(0, i - WINDOW_SIZE)
        fin = min(total, i + WINDOW_SIZE + 1)
        
        for j in range(inicio, fin):
            if i != j and palabras[j] in palabra_a_idx:
                contador += 1
                if contador % 3 == 0:
                    contexto = palabra_a_idx[palabras[j]]
                    pares.append((centro, contexto))
    
    print(f"   ✅ {len(pares)} pares generados")
    return pares


class Word2VecDataset(Dataset):
    def __init__(self, pares):
        self.centros = torch.tensor([p[0] for p in pares], dtype=torch.long)
        self.contextos = torch.tensor([p[1] for p in pares], dtype=torch.long)
    
    def __len__(self):
        return len(self.centros)
    
    def __getitem__(self, idx):
        return self.centros[idx], self.contextos[idx]


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)
    
    def forward(self, centro, contexto):
        v_centro = self.in_embed(centro)
        v_contexto = self.out_embed(contexto)
        score = (v_centro * v_contexto).sum(dim=1)
        return score


def entrenar_modelo(pares, vocab_size, modelo_previo=None):
    """
    Entrena el modelo. Si se pasa modelo_previo, REANUDA desde ahí.
    """
    dataset = Word2VecDataset(pares)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    if modelo_previo is not None:
        modelo = modelo_previo
        print(f"\n🔄 REANUDANDO entrenamiento desde modelo guardado...")
    else:
        modelo = Word2VecModel(vocab_size, EMBEDDING_DIM)
        print(f"\n🆕 INICIANDO entrenamiento desde cero...")
    
    optimizer = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"   Épocas: {EPOCHS} | Batch: {BATCH_SIZE} | Dim: {EMBEDDING_DIM}")
    print(f"   Pares: {len(pares):,} | Pasos/época: ~{len(pares)//BATCH_SIZE}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
        
        for centro, contexto in dataloader:
            score_pos = modelo(centro, contexto)
            loss_pos = criterion(score_pos, torch.ones_like(score_pos))
            
            contexto_neg = torch.randint(0, vocab_size, contexto.shape)
            score_neg = modelo(centro, contexto_neg)
            loss_neg = criterion(score_neg, torch.zeros_like(score_neg))
            
            loss = loss_pos + loss_neg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        print(f"   Época {epoch+1}/{EPOCHS} | Loss: {total_loss/n_batches:.4f}")
    
    return modelo


def guardar_modelo(modelo, vocabulario, palabra_a_idx, idx_a_palabra, total_epochs=0):
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'vocab_size': len(vocabulario),
        'embedding_dim': EMBEDDING_DIM,
        'epochs_entrenados': total_epochs,
    }, MODEL_PATH)
    print(f"\n💾 Modelo guardado en {MODEL_PATH} ({total_epochs} épocas totales)")
    
    embeddings = modelo.in_embed.weight.detach().numpy()
    data = {
        "vector_size": EMBEDDING_DIM,
        "vocabulario": vocabulary if 'vocabulary' in locals() else vocabulario,
        "palabra_a_idx": palabra_a_idx,
        "embeddings": embeddings.tolist()
    }
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"💾 Embeddings guardados en {VOCAB_PATH}")


def cargar_modelo_guardado(vocab_size):
    if os.path.exists(MODEL_PATH):
        print(f"\n📂 Modelo existente encontrado")
        checkpoint = torch.load(MODEL_PATH)
        old_vocab_size = checkpoint['model_state_dict']['in_embed.weight'].shape[0]
        
        modelo = Word2VecModel(vocab_size, EMBEDDING_DIM)
        
        if old_vocab_size == vocab_size:
            modelo.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Reanudando (mismo vocabulario)")
        else:
            print(f"   ⚠️ Vocabulario cambió: {old_vocab_size} → {vocab_size}")
            print(f"   📋 Copiando pesos existentes + inicializando nuevas palabras...")
            
            old_weights_in = checkpoint['model_state_dict']['in_embed.weight']
            old_weights_out = checkpoint['model_state_dict']['out_embed.weight']
            
            modelo.in_embed.weight.data[:old_vocab_size] = old_weights_in
            modelo.out_embed.weight.data[:old_vocab_size] = old_weights_out
        
        epochs_previos = checkpoint.get('epochs_entrenados', 0)
        print(f"   Épocas acumuladas: {epochs_previos}")
        return modelo, epochs_previos
    return None, 0


def probar_modelo(modelo, palabra_a_idx, idx_a_palabra, vocabulario):
    """Evalúa las relaciones usando operaciones vectorizadas en todo el vocabulario"""
    embeddings = modelo.in_embed.weight.detach().numpy()
    normas = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / normas  # Matriz normalizada [Vocab_size, Dim]
    
    def get_vec(palabra):
        if palabra in palabra_a_idx:
            return embeddings[palabra_a_idx[palabra]]
        return None
    
    def similitud(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    
    print("\n🔍 Probando relaciones:")
    pares = [
        ("guerra", "paz"),
        ("soldado", "general"),
        ("hombre", "mujer"),
        ("bueno", "malo"),
        ("guerra", "flor"),
        ("fusil", "arma"),
        ("batalla", "combate"),
    ]
    
    for p1, p2 in pares:
        v1 = get_vec(p1)
        v2 = get_vec(p2)
        if v1 is not None and v2 is not None:
            sim = similitud(v1, v2)
            barra = "█" * int((sim + 1) * 12)
            print(f"   '{p1}' ↔ '{p2}': {sim:+.3f} {barra}")
        else:
            print(f"   '{p1}' o '{p2}' no encontradas")
    
    print("\n📊 Palabras similares (Evaluando todo el vocabulario):")
    for palabra in ["guerra", "soldado", "batalla"]:
        if palabra in palabra_a_idx:
            idx_objetivo = palabra_a_idx[palabra]
            v_obj = embeddings_norm[idx_objetivo]
            
            # Operación matricial instantánea contra las 27,525 palabras
            cosenos = np.dot(embeddings_norm, v_obj)
            indices_similares = np.argsort(cosenos)[::-1]
            
            print(f"\n   '{palabra}':")
            cont = 0
            for idx in indices_similares:
                p_sim = idx_a_palabra[idx]
                if p_sim != palabra:
                    print(f"      {p_sim}: {cosenos[idx]:.3f}")
                    cont += 1
                if cont == 5:
                    break
        else:
            print(f"\n   '{palabra}' no está en el vocabulario.")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("   🧠 WORD2VEC CON PYTORCH (REANUDABLE)")
    print("=" * 60)
    
    palabras = cargar_textos()
    if not palabras:
        exit()
    
    vocabulario, palabra_a_idx, idx_a_palabra = crear_vocabulario(palabras)
    pares = generar_pares(palabras, palabra_a_idx)
    
    modelo_previo, epochs_previos = cargar_modelo_guardado(len(vocabulario))
    
    modelo = entrenar_modelo(pares, len(vocabulario), modelo_previo)
    
    total_epochs = epochs_previos + EPOCHS
    guardar_modelo(modelo, vocabulario, palabra_a_idx, idx_a_palabra, total_epochs)
    probar_modelo(modelo, palabra_a_idx, idx_a_palabra, vocabulario)
    
    print(f"\n✅ ¡Entrenamiento completado! (Total acumulado: {total_epochs} épocas)")