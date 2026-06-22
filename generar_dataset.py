import os
import json
import csv
from groq import Groq
from dotenv import load_dotenv
from evaluador_datos import cargar_banco_preguntas

load_dotenv()
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generar_respuestas_con_groq(pregunta, criterios, n_respuestas=50):
    """Groq genera respuestas variadas con sus notas para una pregunta."""

    criterios_texto = "\n".join([f"- Grupo {i+1}: {', '.join(g)}" for i, g in enumerate(criterios)])

    prompt = f"""
    Eres un generador de dataset académico. Genera exactamente {n_respuestas} respuestas de estudiantes para esta pregunta de examen, con diferentes niveles de calidad.

    PREGUNTA: "{pregunta}"

    CRITERIOS DE EVALUACIÓN (cada grupo vale 2 puntos sobre 10):
    {criterios_texto}

    INSTRUCCIONES:
    - Genera respuestas variadas: excelentes (9-10), buenas (7-8), regulares (5-6), malas (3-4) y muy malas (0-2)
    - La nota debe reflejar cuántos criterios cumple la respuesta
    - Las respuestas deben ser naturales, como las escribiría un estudiante real
    - Varía el estilo: algunas formales, otras informales, algunas con errores ortográficos

    Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
    {{
        "dataset": [
            {{"respuesta": "texto de la respuesta", "nota": 8.5}},
            ...
        ]
    }}
    """

    completion = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.8
    )

    resultado = json.loads(completion.choices[0].message.content)
    return resultado["dataset"]


def guardar_dataset_csv(id_pregunta, dataset):
    """Guarda el dataset generado en un archivo CSV."""
    os.makedirs("data", exist_ok=True)
    archivo = f"data/dataset_pregunta_{id_pregunta}.csv"

    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["respuesta", "nota"])
        writer.writeheader()
        writer.writerows(dataset)

    print(f"✅ Dataset guardado en {archivo} con {len(dataset)} ejemplos")
    return archivo


def generar_datasets_todos():
    """Genera datasets para todas las preguntas del banco."""
    banco = cargar_banco_preguntas()

    for id_p, info in banco.items():
        print(f"\n🤖 Generando dataset para pregunta {id_p}...")
        print(f"   Pregunta: {info['pregunta']}")

        try:
            dataset = generar_respuestas_con_groq(
                pregunta=info["pregunta"],
                criterios=info["criterios"],
                n_respuestas=50
            )
            guardar_dataset_csv(id_p, dataset)

        except Exception as e:
            print(f"❌ Error generando dataset para pregunta {id_p}: {e}")

    print("\n✅ Todos los datasets generados.")


if __name__ == "__main__":
    generar_datasets_todos()