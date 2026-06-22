import os
import json
import time
import numpy as np
import sqlite3
import torch
import torch.optim as optim
from dotenv import load_dotenv
from google import genai
from opo import NeuralNetwork
from evaluador_datos import (
    transformar_respuesta_a_vector,
    guardar_registro_estudiante,
    verificar_plagio,
    agregar_pregunta_dinamica,
    cargar_banco_preguntas,
    conectar_db,
    DB_FILE
)

load_dotenv()
API_KEY_GEMINI = os.environ.get("GEMINI_API_KEY")
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
API_KEY_COHERE = os.environ.get("API_KEY_COHERE")

client_gemini_global = genai.Client(api_key=API_KEY_GEMINI)


def generar_retroalimentacion_ia(pregunta, respuesta_alumno, nota_predicha):
    """
    Genera feedback usando Gemini (Plan A) o Groq (Plan B).
    Ahora recibe la nota predicha en lugar de criterios.
    """
    prompt = f"""
    Actúa como un evaluador académico experto. Evalúa la siguiente respuesta:
    
    - PREGUNTA: "{pregunta}"
    - RESPUESTA DEL ESTUDIANTE: "{respuesta_alumno}"
    - NOTA ASIGNADA POR IA: {nota_predicha:.2f}/10
    
    Debes responder ÚNICAMENTE con un objeto JSON válido:
    {{
        "analisis_error": "Explicación breve de fortalezas y debilidades de la respuesta",
        "retroalimentacion_alumno": "Mensaje constructivo de máximo 3 líneas para el estudiante"
    }}
    """

    for intento in range(2):
        try:
            response = client_gemini_global.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            print("✅ Feedback generado con Gemini")
            return json.loads(response.text.strip())
        except Exception as e:
            if "503" in str(e) and intento < 1:
                print(f"⏳ Gemini saturado, reintentando... ({intento+1}/2)")
                time.sleep(3)
            else:
                print("⚠️ Gemini falló en feedback, cambiando a Groq...")
                break

    try:
        from groq import Groq
        client_groq = Groq(api_key=API_KEY_GROQ)
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        print("✅ Feedback generado con Groq (respaldo)")
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "analisis_error": "Error en el procesamiento del LLM",
            "retroalimentacion_alumno": f"No se pudo generar feedback: {str(e)[:100]}"
        }


def menu_docente():
    print("\n--- PANEL DEL DOCENTE ---")
    try:
        id_p = input("Asigne un número a la nueva pregunta: ").strip()
        preg_texto = input("Escriba la pregunta del examen:\n> ").strip()

        agregar_pregunta_dinamica(id_p, preg_texto)

        print("\n🧠 Entrenando red neuronal para la nueva pregunta...")
        from evaluador_entrenar import entrenar_evaluador
        entrenar_evaluador(id_p, epochs=2000, lr=0.001) 
        print("✅ Red neuronal lista para evaluar esta pregunta.")

    except Exception as e:
        print(f"❌ Error: {e}")


def ver_historial():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha FROM historial_evaluaciones")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No hay evaluaciones guardadas aún.")
            return

        print("\n" + "="*75)
        print("                    HISTORIAL DE EVALUACIONES                    ")
        print("="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota':<12} {'Plagio':<10} {'Fecha'}")
        print("-"*75)

        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<12} {r[4]:<10} {r[5]}")

        print("="*75)
        print(f"Total de evaluaciones: {len(registros)}")

    except Exception as e:
        print(f"❌ Error al leer historial: {e}")


def buscar_historial():
    print("\n--- BUSCAR EN HISTORIAL ---")
    print("[1] Buscar por nombre de estudiante")
    print("[2] Buscar por número de pregunta")
    op = input("Seleccione: ").strip()

    try:
        conn, cursor = conectar_db()

        if op == "1":
            nombre = input("Ingrese el nombre del estudiante: ").strip()
            cursor.execute("""
                SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha 
                FROM historial_evaluaciones 
                WHERE LOWER(estudiante) LIKE LOWER(?)
            """, (f"%{nombre}%",))

        elif op == "2":
            id_p = input("Ingrese el número de pregunta: ").strip()
            cursor.execute("""
                SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha 
                FROM historial_evaluaciones 
                WHERE id_pregunta = ?
            """, (id_p,))
        else:
            print("❌ Opción inválida.")
            conn.close()
            return

        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No se encontraron resultados.")
            return

        print("\n" + "="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota':<12} {'Plagio':<10} {'Fecha'}")
        print("-"*75)
        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<12} {r[4]:<10} {r[5]}")
        print("="*75)
        print(f"Resultados encontrados: {len(registros)}")

    except Exception as e:
        print(f"❌ Error al buscar: {e}")


def estadisticas_por_pregunta():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT DISTINCT id_pregunta FROM historial_evaluaciones")
        preguntas = cursor.fetchall()

        if not preguntas:
            print("❌ No hay evaluaciones guardadas aún.")
            conn.close()
            return

        print("\n" + "="*60)
        print("           ESTADÍSTICAS POR PREGUNTA           ")
        print("="*60)

        for (id_p,) in preguntas:
            cursor.execute("""
                SELECT puntuacion_ia FROM historial_evaluaciones WHERE id_pregunta = ?
            """, (id_p,))
            notas_raw = cursor.fetchall()

            notes = []
            for (nota_str,) in notas_raw:
                try:
                    nota = float(nota_str.split("/")[0].strip())
                    notes.append(nota)
                except:
                    pass

            if not notes:
                continue

            promedio = sum(notes) / len(notes)
            aprobados = sum(1 for n in notes if n >= 6)
            reprobados = len(notes) - aprobados

            print(f"\n📋 Pregunta {id_p}:")
            print(f"   Total evaluados : {len(notes)}")
            print(f"   Promedio        : {promedio:.2f} / 10.00")
            print(f"   ✅ Aprobados    : {aprobados}")
            print(f"   ❌ Reprobados   : {reprobados}")

        conn.close()
        print("="*60)

    except Exception as e:
        print(f"❌ Error al calcular estadísticas: {e}")


def eliminar_historial():
    confirm = input("\n⚠️ ¿Estás seguro de ELIMINAR TODAS las evaluaciones? (SI para confirmar): ").strip()
    if confirm.upper() != "SI":
        print("❌ Operación cancelada.")
        return

    try:
        conn, cursor = conectar_db()
        cursor.execute("DELETE FROM historial_evaluaciones")
        conn.commit()
        conn.close()
        print("✅ Historial eliminado correctamente.")
    except Exception as e:
        print(f"❌ Error al eliminar historial: {e}")


def exportar_historial_excel():
    try:
        import pandas as pd
        conn, cursor = conectar_db()
        df = pd.read_sql_query("SELECT * FROM historial_evaluaciones", conn)
        conn.close()

        if df.empty:
            print("❌ La base de datos está vacía.")
            return

        nombre_archivo = "historial_evaluaciones.xlsx"
        df.to_excel(nombre_archivo, index=False)
        print(f"✅ Historial exportado a {nombre_archivo}")
    except Exception as e:
        print(f"❌ Error al exportar: {e}")


def corregir_evaluacion():
    print("\n--- CORRECCIÓN DE EVALUACIÓN POR DOCENTE ---")

    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, fecha FROM historial_evaluaciones")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No hay evaluaciones para corregir.")
            return

        print("\n" + "="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota actual':<15} {'Fecha'}")
        print("-"*75)
        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<15} {r[4]}")
        print("="*75)

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    try:
        id_registro = int(input("\nIngrese el # de la evaluación a corregir: ").strip())
        nota_correcta = float(input("Ingrese la nota correcta (0-10): ").strip())

        if not (0 <= nota_correcta <= 10):
            print("❌ La nota debe estar entre 0 y 10.")
            return

    except ValueError:
        print("❌ Valor inválido.")
        return

    try:
        conn, cursor = conectar_db()
        cursor.execute("""
            SELECT id_pregunta, vector_neurona FROM historial_evaluaciones WHERE id = ?
        """, (id_registro,))
        resultado = cursor.fetchone()
        conn.close()

        if not resultado:
            print("❌ No se encontró esa evaluación.")
            return

        id_pregunta = resultado[0]
        # CORREGIDO: El vector ahora es semántico (768 dims truncado)
        print("⚠️ La corrección manual usa el vector almacenado (primeros 5 valores)")

        # Actualizar solo la nota en la BD
        conn, cursor = conectar_db()
        cursor.execute("""
            UPDATE historial_evaluaciones SET puntuacion_ia = ? WHERE id = ?
        """, (f"{nota_correcta:.2f} / 10.00", id_registro))
        conn.commit()
        conn.close()

        print(f"✅ Nota actualizada a {nota_correcta:.2f} / 10.00")

    except Exception as e:
        print(f"❌ Error al corregir: {e}")


def evaluar_con_api():
    """
    Evalúa una respuesta usando la red neuronal semántica (768 dims).
    """
    preguntas_actuales = cargar_banco_preguntas()

    print("\n" + "="*60)
    print("   SISTEMA INTEGRADO: RED NEURONAL + LLM   ")
    print("="*60)

    nombre_alumno = input("Ingrese el nombre completo del estudiante: ").strip()
    if not nombre_alumno:
        nombre_alumno = "Estudiante Anónimo"

    print("\nPreguntas disponibles para evaluar:")
    for id_p, info in preguntas_actuales.items():
        print(f" [{id_p}] - {info['pregunta']}")

    id_elegido = input("\nSeleccione el número de la pregunta: ").strip()
    if id_elegido not in preguntas_actuales:
        print("❌ Pregunta no válida.")
        return

    info_pregunta = preguntas_actuales[id_elegido]

    # ============================================================
    # CORRECCIÓN 1: Nombre de archivo correcto
    # ============================================================
    nn = NeuralNetwork()
    nombre_cerebro = f"cerebro_pregunta_{id_elegido}.json"  # ✅ CORREGIDO
    
    try:
        nn.load(nombre_cerebro)
        print(f"✅ Modelo {nombre_cerebro} cargado correctamente.")
    except FileNotFoundError:
        print(f"❌ No existe {nombre_cerebro}. Entrena primero la pregunta {id_elegido}.")
        return

    print("="*60)
    respuesta_alumno = input("Escriba la respuesta del examen: ").strip()
    while not respuesta_alumno:
        print("❌ La respuesta no puede estar vacía.")
        respuesta_alumno = input("Escriba la respuesta del examen: ").strip()

    # Verificar plagio
    es_plagio, porcentaje, sospechoso_de = verificar_plagio(respuesta_alumno, id_elegido)
    plagio_string = "Ninguno"

    if es_plagio:
        print(f"\n⚠️ ALERTA: {porcentaje:.2f}% de similitud con '{sospechoso_de}'")
        plagio_string = f"Sospecha alta ({porcentaje:.2f}% con {sospechoso_de})"

    # ============================================================
    # CORRECCIÓN 2: Vector de 768 dimensiones (sin criterios viejos)
    # ============================================================
    print("🧠 Generando embedding semántico...")
    vector_entradas = transformar_respuesta_a_vector(respuesta_alumno)  # ✅ SIN criterios
    
    # Predecir nota
    prediccion_np = nn.predict(vector_entradas)
    nota_final = float(prediccion_np.ravel()[0]) * 10
    
    # Limitar nota entre 0 y 10
    nota_final = max(0, min(10, nota_final))

    print(f"\n📊 Vector semántico generado: {vector_entradas.shape}")
    print(f"🎯 Nota predicha por la red: {nota_final:.2f} / 10.00")

    # Generar feedback con LLM
    print("🤖 Generando feedback con IA...")
    feedback_estructurado = generar_retroalimentacion_ia(
        info_pregunta["pregunta"],
        respuesta_alumno,
        nota_final
    )

    # Mostrar resultados
    print("\n" + "="*60)
    print("               RESULTADO DE LA EVALUACIÓN          ")
    print("="*60)
    print(f"▶️ ESTUDIANTE: {nombre_alumno}")
    print(f"▶️ PREGUNTA: {info_pregunta['pregunta']}")
    print(f"▶️ NOTA: {nota_final:.2f} / 10.00")
    print(f"▶️ PLAGIO: {plagio_string}")
    print(f"▶️ ANÁLISIS: {feedback_estructurado.get('analisis_error', 'N/A')}")
    print(f"▶️ FEEDBACK:\n{feedback_estructurado.get('retroalimentacion_alumno', 'N/A')}")
    print("="*60 + "\n")

    # Guardar registro
    guardar_registro_estudiante(
        nombre_estudiante=nombre_alumno,
        id_pregunta=id_elegido,
        pregunta_texto=info_pregunta['pregunta'],
        respuesta_alumno=respuesta_alumno,
        vector_generado=vector_entradas,
        puntuacion_ia=f"{nota_final:.2f} / 10.00",
        feedback_ia=feedback_estructurado.get('retroalimentacion_alumno', ''),
        plagio_info=plagio_string
    )


def panel_principal():
    while True:
        print("\n" + "="*50)
        print("   MENÚ PRINCIPAL - SISTEMA DE EVALUACIÓN IA   ")
        print("="*50)
        print("[1] Panel del Docente (Agregar preguntas)")
        print("[2] Evaluar Respuesta (Red Neuronal + LLM)")
        print("[3] Ver Historial de Evaluaciones")
        print("[4] Buscar en Historial")
        print("[5] Estadísticas por Pregunta")
        print("[6] Exportar Historial a Excel")
        print("[7] Eliminar Historial")
        print("[8] Corregir Evaluación")
        print("[9] Salir")
        print("="*50)
        
        op = input("Seleccione una opción: ").strip()

        if op == "1":
            menu_docente()
        elif op == "2":
            evaluar_con_api()
        elif op == "3":
            ver_historial()
        elif op == "4":
            buscar_historial()
        elif op == "5":
            estadisticas_por_pregunta()
        elif op == "6":
            exportar_historial_excel()
        elif op == "7":
            eliminar_historial()
        elif op == "8":
            corregir_evaluacion()
        elif op == "9":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    panel_principal()