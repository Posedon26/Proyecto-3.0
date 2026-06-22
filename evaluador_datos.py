import os
import re
import json
import sqlite3
import unicodedata
import hashlib
import numpy as np
import requests
import torch
import time
from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
API_KEY_GEMINI = os.environ.get("GEMINI_API_KEY")
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
API_KEY_COHERE = os.environ.get("API_KEY_COHERE")

client_gemini = genai.Client(
    api_key=API_KEY_GEMINI,
    http_options={'api_version': 'v1'}
)
client_groq = Groq(api_key=API_KEY_GROQ)

DB_FILE = "sistema_evaluacion.db"

# ============================================================
# 1. CONTROL Y PERSISTENCIA CON BASE DE DATOS (SQLite)
# ============================================================

def conectar_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            estudiante TEXT NOT NULL,
            id_pregunta TEXT NOT NULL,
            pregunta TEXT,
            respuesta_estudiante TEXT,
            vector_neurona TEXT,
            puntuacion_ia TEXT,
            feedback_ia TEXT,
            alerta_plagio TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, cursor


def cargar_banco_preguntas():
    PREGUNTAS_FILE = "banco_preguntas.json"
    BANCO_PREGUNTAS_BASE = {
        "1": {
            "pregunta": "¿En qué año se dio la Revolución Francesa y qué sistema cayó?",
            "criterios": []
        },
        "2": {
            "pregunta": "¿Cómo se repite un bloque de código en Python de forma controlada?",
            "criterios": []
        }
    }
    if os.path.exists(PREGUNTAS_FILE):
        with open(PREGUNTAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(PREGUNTAS_FILE, "w", encoding="utf-8") as f:
            json.dump(BANCO_PREGUNTAS_BASE, f, ensure_ascii=False, indent=4)
        return BANCO_PREGUNTAS_BASE


def agregar_pregunta_dinamica(id_pregunta, pregunta_texto, lista_criterios=None):
    PREGUNTAS_FILE = "banco_preguntas.json"
    banco = cargar_banco_preguntas()
    banco[str(id_pregunta)] = {
        "pregunta": pregunta_texto,
        "criterios": []
    }
    with open(PREGUNTAS_FILE, "w", encoding="utf-8") as f:
        json.dump(banco, f, ensure_ascii=False, indent=4)
    print(f"\n[Docente] Pregunta {id_pregunta} añadida de forma semántica exitosamente.")


def guardar_registro_estudiante(
    nombre_estudiante, id_pregunta, pregunta_texto, respuesta_alumno,
    vector_generado, puntuacion_ia="N/A", feedback_ia="N/A", plagio_info="Ninguno"
):
    try:
        conn, cursor = conectar_db()
        
        if isinstance(vector_generado, torch.Tensor):
            vector_np = vector_generado.squeeze(0).cpu().numpy()
        elif isinstance(vector_generado, np.ndarray):
            vector_np = vector_generado.ravel()
        else:
            vector_np = np.array(vector_generado).ravel()

        vector_str = ",".join([f"{x:.4f}" for x in vector_np[:5]]) + "... [Truncado Semántico 768]"

        cursor.execute("""
            INSERT INTO historial_evaluaciones 
            (estudiante, id_pregunta, pregunta, respuesta_estudiante, vector_neurona, puntuacion_ia, feedback_ia, alerta_plagio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (nombre_estudiante, id_pregunta, pregunta_texto, respuesta_alumno, vector_str, puntuacion_ia, feedback_ia, plagio_info))

        conn.commit()
        conn.close()
        print(f"[Sistema - SQLite] Registro de '{nombre_estudiante}' guardado con éxito semántico.")
    except Exception as e:
        print(f"❌ Error al interactuar con SQLite: {e}")


# ============================================================
# 2. ANÁLISIS DE PLAGIO OPTIMIZADO DESDE SQLITE
# ============================================================

def normalizar_texto(texto):
    if not texto:
        return ""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(re.compile(r'[.,;:?¿!¡"()\'\-]'), ' ', texto)
    return ' '.join(texto.split())


def verificar_plagio(respuesta_nueva, id_pregunta, umbral=0.70):
    from difflib import SequenceMatcher
    try:
        conn, cursor = conectar_db()
        cursor.execute("""
            SELECT estudiante, respuesta_estudiante 
            FROM historial_evaluaciones 
            WHERE id_pregunta = ?
        """, (str(id_pregunta),))
        registros = cursor.fetchall()
        conn.close()
    except Exception:
        return False, 0.0, ""

    texto_nueva_limpio = normalizar_texto(respuesta_nueva)

    for registro in registros:
        estudiante_anterior = registro[0]
        respuesta_anterior = normalizar_texto(registro[1])
        similitud = SequenceMatcher(None, texto_nueva_limpio, respuesta_anterior).ratio()
        if similitud >= umbral:
            return True, similitud * 100, estudiante_anterior

    return False, 0.0, ""


# ============================================================
# 3. FUNCIONES DE EMBEDDING (Plan A: Gemini, Plan B: Groq, Plan C: Cohere)
# ============================================================

def reducir_dimensiones(embedding, dim_objetivo=768):
    dim_original = len(embedding)
    if dim_original == dim_objetivo:
        return embedding
    
    chunk_size = dim_original / dim_objetivo
    embedding_reducido = np.array([
        np.mean(embedding[int(i*chunk_size):int((i+1)*chunk_size)]) 
        for i in range(dim_objetivo)
    ], dtype=np.float32)
    
    return embedding_reducido


def obtener_embedding_gemini(texto):
    try:
        resultado = client_gemini.models.embed_content(
            model="gemini-embedding-2",
            contents=texto
        )
        embedding = np.array(resultado.embeddings[0].values, dtype=np.float32)
        dims = len(embedding)
        print(f"      📏 Gemini: {dims} dimensiones → reduciendo a 768")
        embedding = reducir_dimensiones(embedding, 768)
        return embedding
    except Exception as e:
        print(f"      ⚠️ Error Gemini: {str(e)[:150]}")
        raise e


def obtener_embedding_groq(texto):
    try:
        respuesta = client_groq.embeddings.create(
            model="llama-3.2-3b-preview",
            input=texto
        )
        embedding_raw = np.array(respuesta.data[0].embedding, dtype=np.float32)
        dims = len(embedding_raw)
        print(f"      📏 Groq: {dims} dimensiones → reduciendo a 768")
        embedding = reducir_dimensiones(embedding_raw, 768)
        return embedding
    except Exception as e:
        print(f"      ⚠️ Error Groq: {str(e)[:150]}")
        raise e


def obtener_embedding_cohere(texto):
    if not API_KEY_COHERE:
        raise Exception("Falta API_KEY_COHERE en .env")

    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {API_KEY_COHERE}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [texto],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_document"
    }
    
    try:
        respuesta = requests.post(url, json=payload, headers=headers, timeout=10)
        if respuesta.status_code == 200:
            embedding = np.array(respuesta.json()["embeddings"][0], dtype=np.float32)
            dims = len(embedding)
            print(f"      📏 Cohere: {dims} dimensiones → reduciendo a 768")
            embedding = reducir_dimensiones(embedding, 768)
            return embedding
        else:
            raise Exception(f"Código {respuesta.status_code}: {respuesta.text[:200]}")
    except requests.exceptions.Timeout:
        raise Exception("Timeout en Cohere API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error de red Cohere: {str(e)[:100]}")


def transformar_respuesta_a_vector(respuesta_alumno, lista_criterios_dummy=None, umbral_dummy=None):
    """
    Extrae el embedding normalizado a 768 dimensiones.
    Plan A: Gemini → Plan B: Groq → Plan C: Cohere → Plan D: Vocabulario Local
    Retorna torch.Tensor de shape (1, 768)
    """
    # PLAN A: Gemini
    try:
        print("      🧠 [Plan A] Gemini...")
        embedding = obtener_embedding_gemini(respuesta_alumno)
        print(f"      ✅ Embedding listo: {embedding.shape}")
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        error_msg = str(e)[:120]
        print(f"      ⚠️ Gemini falló: {error_msg}")
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("      ⏳ Rate limit, esperando 3s...")
            time.sleep(3)

    # PLAN B: Groq
    try:
        print("      🟡 [Plan B] Groq...")
        embedding = obtener_embedding_groq(respuesta_alumno)
        print(f"      ✅ Embedding listo: {embedding.shape}")
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"      ⚠️ Groq falló: {str(e)[:120]}")

    # PLAN C: Cohere
    try:
        print("      🟠 [Plan C] Cohere...")
        embedding = obtener_embedding_cohere(respuesta_alumno)
        print(f"      ✅ Embedding listo: {embedding.shape}")
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"      ❌ Cohere también falló: {str(e)[:120]}")
        
    # PLAN D: Vocabulario Local (OFFLINE)
    try:
        print("      📚 [Plan D] Vocabulario local...")
        embedding_local = texto_a_vector_local(respuesta_alumno)
        if embedding_local is not None and len(embedding_local) > 0:
            # Reducir de 200 a 768 (padding con ceros o reducción)
            if len(embedding_local) < 768:
                embedding_padded = np.pad(embedding_local, (0, 768 - len(embedding_local)), 'constant')
            else:
                embedding_padded = reducir_dimensiones(embedding_local, 768)
            print(f"      ✅ Vector local generado: {embedding_padded.shape}")
            return torch.tensor(embedding_padded, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"      ⚠️ Vocabulario local falló: {str(e)[:80]}")

    # EMERGENCIA
    print("      🔴 VECTOR DE CEROS (768) - Todos los servicios fallaron")
    return torch.zeros((1, 768), dtype=torch.float32)


# ============================================================
# 4. FUNCIONES DE USUARIOS (LOGIN)
# ============================================================

def crear_tabla_usuarios():
    conn, cursor = conectar_db()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rol TEXT NOT NULL DEFAULT 'docente',
            nombre_completo TEXT,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO usuarios (usuario, password_hash, rol, nombre_completo) VALUES (?, ?, ?, ?)",
            ("admin", password_hash, "docente", "Administrador Principal")
        )
        print("✅ Usuario admin creado (usuario: admin, contraseña: admin123)")
    conn.commit()
    conn.close()


def verificar_usuario(usuario, password):
    conn, cursor = conectar_db()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "SELECT id, usuario, rol, nombre_completo FROM usuarios WHERE usuario = ? AND password_hash = ?",
        (usuario, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user[0], "usuario": user[1], "rol": user[2], "nombre": user[3]}
    return None


def registrar_usuario(usuario, password, rol="docente", nombre_completo=""):
    conn, cursor = conectar_db()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute(
            "INSERT INTO usuarios (usuario, password_hash, rol, nombre_completo) VALUES (?, ?, ?, ?)",
            (usuario, password_hash, rol, nombre_completo)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        print(f"❌ El usuario '{usuario}' ya existe.")
        conn.close()
        return False
    except Exception as e:
        print(f"❌ Error al registrar usuario: {e}")
        conn.close()
        return False


# ============================================================
# 5. VOCABULARIO LITERARIO (Embeddings de palabras - OFFLINE)
# ============================================================

_modelo_vocabulario = None

def cargar_modelo_vocabulario():
    global _modelo_vocabulario
    ruta_modelo = "data/vocabulario.json"
    if not os.path.exists(ruta_modelo):
        print("⚠️ Modelo de vocabulario no encontrado. Ejecuta entrenar_vocabulario.py primero.")
        return False
    try:
        with open(ruta_modelo, 'r', encoding='utf-8') as f:
            _modelo_vocabulario = json.load(f)
        print(f"📚 Vocabulario cargado: {len(_modelo_vocabulario['vocabulario'])} palabras")
        return True
    except Exception as e:
        print(f"❌ Error al cargar vocabulario: {e}")
        return False


def palabra_a_vector(palabra):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return None
    palabra = palabra.lower().strip()
    if palabra in _modelo_vocabulario['palabra_a_idx']:
        idx = _modelo_vocabulario['palabra_a_idx'][palabra]
        return np.array(_modelo_vocabulario['embeddings'][idx], dtype=np.float32)
    return np.zeros(_modelo_vocabulario['vector_size'], dtype=np.float32)


def texto_a_vector_local(texto):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return None
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    palabras = texto.split()
    vectores = []
    for palabra in palabras:
        if palabra in _modelo_vocabulario['palabra_a_idx']:
            idx = _modelo_vocabulario['palabra_a_idx'][palabra]
            vectores.append(np.array(_modelo_vocabulario['embeddings'][idx], dtype=np.float32))
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(_modelo_vocabulario['vector_size'], dtype=np.float32)


def palabras_similares(palabra, topn=5):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return []
    palabra = palabra.lower().strip()
    if palabra not in _modelo_vocabulario['palabra_a_idx']:
        print(f"❌ '{palabra}' no está en el vocabulario")
        return []
    idx = _modelo_vocabulario['palabra_a_idx'][palabra]
    vec = np.array(_modelo_vocabulario['embeddings'][idx])
    similitudes = []
    for i, p in enumerate(_modelo_vocabulario['vocabulario']):
        if p != palabra:
            vec_p = np.array(_modelo_vocabulario['embeddings'][i])
            sim = np.dot(vec, vec_p) / (np.linalg.norm(vec) * np.linalg.norm(vec_p) + 1e-8)
            similitudes.append((p, sim))
    similitudes.sort(key=lambda x: x[1], reverse=True)
    return similitudes[:topn]


def comparar_palabras(palabra1, palabra2):
    vec1 = palabra_a_vector(palabra1)
    vec2 = palabra_a_vector(palabra2)
    if vec1 is None or vec2 is None:
        return None
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    return float(sim)


# Cargar banco de preguntas al iniciar
BANCO_PREGUNTAS = cargar_banco_preguntas()