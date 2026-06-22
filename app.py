import os
import json
import pdfkit
import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv
from functools import wraps
from opo import NeuralNetwork
from evaluador_datos import (
    transformar_respuesta_a_vector,
    guardar_registro_estudiante,
    verificar_plagio,
    agregar_pregunta_dinamica,
    cargar_banco_preguntas,
    conectar_db,
    crear_tabla_usuarios,
    verificar_usuario,
    registrar_usuario
)
from main_api import generar_retroalimentacion_ia

load_dotenv()
app = Flask(__name__)
app.secret_key = "clave_secreta_123"

# Crear tablas al iniciar
crear_tabla_usuarios()

# Crear usuarios de prueba
def crear_usuarios_prueba():
    registrar_usuario("admin", "admin123", "docente", "Profesor Principal")
    registrar_usuario("profe", "profe123", "docente", "María García")
    registrar_usuario("alumno", "alumno123", "estudiante", "Juan Pérez")
    registrar_usuario("lucas", "lucas123", "estudiante", "Lucas Modificado")

crear_usuarios_prueba()


# ============================================================
# DECORADORES
# ============================================================

def login_requerido(f):
    """Cualquier usuario logueado (docente o estudiante)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario' not in session:
            flash("🔐 Debes iniciar sesión primero.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def docente_requerido(f):
    """Solo docentes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario' not in session:
            flash("🔐 Debes iniciar sesión primero.", "error")
            return redirect(url_for('login'))
        if session.get('rol') != 'docente':
            flash("❌ Solo los docentes pueden acceder aquí.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# RUTA: INICIO
# ============================================================
@app.route("/")
def index():
    user_info = None
    if 'usuario' in session:
        user_info = {
            'usuario': session.get('usuario'),
            'rol': session.get('rol'),
            'nombre': session.get('nombre')
        }
    return render_template("index.html", user=user_info)


# ============================================================
# RUTA: LOGIN
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    # Si ya está logueado, redirigir al index
    if 'usuario' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        usuario = request.form.get("usuario", "").strip()
        password = request.form.get("password", "").strip()
        
        if not usuario or not password:
            flash("❌ Usuario y contraseña son requeridos.", "error")
            return render_template("login.html")
        
        user = verificar_usuario(usuario, password)
        
        if user:
            session['usuario'] = user['usuario']
            session['rol'] = user['rol']
            session['nombre'] = user['nombre']
            session['user_id'] = user['id']
            flash(f"✅ Bienvenido, {user['nombre'] or user['usuario']}!", "success")
            return redirect(url_for('index'))
        else:
            flash("❌ Usuario o contraseña incorrectos.", "error")
    
    return render_template("login.html")


# ============================================================
# RUTA: LOGOUT
# ============================================================
@app.route("/logout")
def logout():
    session.clear()
    flash("👋 Sesión cerrada correctamente.", "success")
    return redirect(url_for('login'))


# ============================================================
# RUTA: EVALUAR (Estudiantes y Docentes)
# ============================================================
@app.route("/evaluar", methods=["GET", "POST"])
def evaluar():     
    preguntas = cargar_banco_preguntas()

    if request.method == "POST":
        nombre_alumno = request.form.get("nombre", "Estudiante Anónimo").strip()
        id_elegido = request.form.get("pregunta").strip()
        respuesta_alumno = request.form.get("respuesta", "").strip()

        if not respuesta_alumno:
            flash("❌ La respuesta no puede estar vacía.")
            return render_template("evaluar.html", preguntas=preguntas)

        if id_elegido not in preguntas:
            flash("❌ Pregunta no válida.")
            return render_template("evaluar.html", preguntas=preguntas)

        info_pregunta = preguntas[id_elegido]

        # Cargar modelo
        nn = NeuralNetwork()
        nombre_cerebro = f"cerebro_pregunta_{id_elegido}.json"
        
        try:
            nn.load(nombre_cerebro)
            print(f"✅ Modelo {nombre_cerebro} cargado")
        except FileNotFoundError:
            flash(f"❌ Primero entrena el modelo para la pregunta {id_elegido}.")
            return render_template("evaluar.html", preguntas=preguntas)

        # Verificar plagio
        es_plagio, porcentaje, sospechoso_de = verificar_plagio(respuesta_alumno, id_elegido)
        plagio_string = "Ninguno"
        if es_plagio:
            plagio_string = f"Sospecha alta ({porcentaje:.2f}% similitud con {sospechoso_de})"

        # Generar embedding y predecir
        print("🧠 Generando embedding semántico...")
        vector_entradas = transformar_respuesta_a_vector(respuesta_alumno)
        
        prediccion_np = nn.predict(vector_entradas)
        nota_final = float(prediccion_np.ravel()[0]) * 10
        nota_final = max(0.0, min(10.0, nota_final))
        
        print(f"📊 Nota predicha: {nota_final:.2f} / 10.00")

        # Generar feedback
        print("🤖 Generando feedback con LLM...")
        feedback = generar_retroalimentacion_ia(
            info_pregunta["pregunta"],
            respuesta_alumno,
            nota_final
        )

        # Criterios de resultado
        criterios_resultado = [
            {"grupo": "Evaluación Semántica", "cumplido": nota_final >= 5.0},
            {"grupo": "Análisis de Contenido", "cumplido": nota_final >= 4.0},
            {"grupo": "Precisión Conceptual", "cumplido": nota_final >= 6.0}
        ]

        # Guardar registro
        guardar_registro_estudiante(
            nombre_estudiante=nombre_alumno,
            id_pregunta=id_elegido,
            pregunta_texto=info_pregunta["pregunta"],
            respuesta_alumno=respuesta_alumno,
            vector_generado=vector_entradas,
            puntuacion_ia=f"{nota_final:.2f} / 10.00",
            feedback_ia=feedback.get("retroalimentacion_alumno", ""),
            plagio_info=plagio_string
        )

        return render_template(
            "resultado.html",
            nombre=nombre_alumno,
            nota=f"{nota_final:.2f}",
            criterios=criterios_resultado,
            plagio=plagio_string,
            analisis=feedback.get("analisis_error", "Análisis no disponible"),
            retroalimentacion=feedback.get("retroalimentacion_alumno", "Sin retroalimentación")
        )

    return render_template("evaluar.html", preguntas=preguntas)


# ============================================================
# RUTA: DOCENTE (Solo docentes)
# ============================================================
@app.route("/docente", methods=["GET", "POST"])
@docente_requerido
def docente():
    if request.method == "POST":
        id_p = request.form.get("id_pregunta").strip()
        preg_texto = request.form.get("pregunta_texto").strip()
        criterios = []
        for i in range(1, 6):
            grupo = request.form.get(f"grupo{i}", "").lower().replace(" ", "").split(",")
            criterios.append(grupo)

        agregar_pregunta_dinamica(id_p, preg_texto, criterios)

        from evaluador_entrenar import entrenar_evaluador
        entrenar_evaluador(id_p, epochs=2000, lr=0.001)

        flash(f"✅ Pregunta {id_p} agregada y red neuronal entrenada correctamente.")
        return redirect(url_for("docente"))

    return render_template("docente.html")


# ============================================================
# RUTA: DASHBOARD (Solo docentes)
# ============================================================
@app.route("/dashboard")
@docente_requerido
def dashboard():
    try:
        conn, cursor = conectar_db()
        
        cursor.execute("SELECT COUNT(*) FROM historial_evaluaciones")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones")
        notas_raw = cursor.fetchall()
        
        notas = []
        for (nota_str,) in notas_raw:
            try:
                valor = float(nota_str.split("/")[0].strip())
                notas.append(valor)
            except:
                pass
        
        promedio = round(sum(notas) / len(notas), 2) if notas else 0
        aprobados = sum(1 for n in notas if n >= 6)
        reprobados = len(notas) - aprobados
        porcentaje_aprobados = round((aprobados / len(notas)) * 100, 1) if notas else 0
        
        cursor.execute("SELECT DISTINCT id_pregunta FROM historial_evaluaciones")
        preguntas_ids = [row[0] for row in cursor.fetchall()]
        
        stats_por_pregunta = []
        for pid in preguntas_ids:
            cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones WHERE id_pregunta = ?", (pid,))
            notas_preg = cursor.fetchall()
            notas_p = []
            for (ns,) in notas_preg:
                try:
                    notas_p.append(float(ns.split("/")[0].strip()))
                except:
                    pass
            if notas_p:
                stats_por_pregunta.append({
                    "id": pid, "total": len(notas_p),
                    "promedio": round(sum(notas_p) / len(notas_p), 2),
                    "maxima": round(max(notas_p), 2),
                    "minima": round(min(notas_p), 2)
                })
        
        excelente = sum(1 for n in notas if n >= 9)
        bueno = sum(1 for n in notas if 7 <= n < 9)
        regular = sum(1 for n in notas if 5 <= n < 7)
        malo = sum(1 for n in notas if n < 5)
        
        distribucion = {"excelente": excelente, "bueno": bueno, "regular": regular, "malo": malo}
        
        cursor.execute("SELECT estudiante, id_pregunta, puntuacion_ia, fecha FROM historial_evaluaciones ORDER BY id DESC LIMIT 10")
        ultimas = cursor.fetchall()
        
        cursor.execute("SELECT puntuacion_ia, fecha FROM historial_evaluaciones ORDER BY id ASC")
        todas = cursor.fetchall()
        
        historial_notas = []
        for i, (nota_str, fecha) in enumerate(todas):
            try:
                valor = float(nota_str.split("/")[0].strip())
                historial_notas.append({"indice": i + 1, "nota": valor, "fecha": fecha})
            except:
                pass
        
        conn.close()
        
        return render_template(
            "dashboard.html", total=total, promedio=promedio,
            aprobados=aprobados, reprobados=reprobados,
            porcentaje_aprobados=porcentaje_aprobados,
            stats_por_pregunta=stats_por_pregunta,
            distribucion=distribucion, ultimas=ultimas,
            historial_notas=historial_notas
        )
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")
        return render_template(
            "dashboard.html", total=0, promedio=0, aprobados=0, reprobados=0,
            porcentaje_aprobados=0, stats_por_pregunta=[],
            distribucion={"excelente": 0, "bueno": 0, "regular": 0, "malo": 0},
            ultimas=[], historial_notas=[]
        )


# ============================================================
# RUTA: HISTORIAL (Solo docentes)
# ============================================================
@app.route("/historial")
@docente_requerido
def historial():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha FROM historial_evaluaciones")
        registros = cursor.fetchall()
        conn.close()
    except Exception as e:
        registros = []
    return render_template("historial.html", registros=registros)


# ============================================================
# RUTA: DESCARGAR PDF (Solo docentes)
# ============================================================
@app.route("/descargar-pdf")
@docente_requerido
def descargar_pdf():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT COUNT(*) FROM historial_evaluaciones")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones")
        notas_raw = cursor.fetchall()
        notas = []
        for (nota_str,) in notas_raw:
            try:
                notas.append(float(nota_str.split("/")[0].strip()))
            except:
                pass
        promedio = round(sum(notas) / len(notas), 2) if notas else 0
        aprobados = sum(1 for n in notas if n >= 6)
        reprobados = len(notas) - aprobados
        porcentaje_aprobados = round((aprobados / len(notas)) * 100, 1) if notas else 0
        excelente = sum(1 for n in notas if n >= 9)
        bueno = sum(1 for n in notas if 7 <= n < 9)
        regular = sum(1 for n in notas if 5 <= n < 7)
        malo = sum(1 for n in notas if n < 5)
        cursor.execute("SELECT estudiante, id_pregunta, puntuacion_ia, fecha FROM historial_evaluaciones ORDER BY id DESC LIMIT 20")
        ultimas = cursor.fetchall()
        cursor.execute("SELECT DISTINCT id_pregunta FROM historial_evaluaciones")
        preguntas_ids = [row[0] for row in cursor.fetchall()]
        stats_por_pregunta = []
        for pid in preguntas_ids:
            cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones WHERE id_pregunta = ?", (pid,))
            notas_preg = cursor.fetchall()
            notas_p = []
            for (ns,) in notas_preg:
                try:
                    notas_p.append(float(ns.split("/")[0].strip()))
                except:
                    pass
            if notas_p:
                stats_por_pregunta.append({
                    "id": pid, "total": len(notas_p),
                    "promedio": round(sum(notas_p) / len(notas_p), 2),
                    "maxima": round(max(notas_p), 2), "minima": round(min(notas_p), 2)
                })
        conn.close()
        
        fecha_actual = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
        año_actual = datetime.datetime.now().year
        
        filas_mision = ""
        for s in stats_por_pregunta:
            filas_mision += f"<tr><td>Pregunta {s['id']}</td><td>{s['total']}</td><td>{s['promedio']}</td><td style='color:#4ade80;'>{s['maxima']}</td><td style='color:#f87171;'>{s['minima']}</td></tr>"
        
        filas_ultimas = ""
        for u in ultimas:
            nota_str = u[2]
            try:
                valor = float(nota_str.split("/")[0].strip())
                clase = "color:#4ade80;" if valor >= 7 else ("color:#fbbf24;" if valor >= 5 else "color:#f87171;")
            except:
                clase = ""
            filas_ultimas += f"<tr><td>{u[0]}</td><td>Pregunta {u[1]}</td><td style='{clase}'>{nota_str}</td><td>{u[3]}</td></tr>"
        
        html_reporte = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><style>
            body{{font-family:Arial,sans-serif;background:#0a0a1a;color:#e0e0f0;padding:30px;}}
            .header{{text-align:center;border-bottom:2px solid #7c8aff;padding-bottom:15px;margin-bottom:25px;}}
            .header h1{{color:#7c8aff;font-size:26px;margin:0;}}
            .header p{{color:#9090b0;font-size:13px;}}
            .stats-grid{{display:flex;gap:12px;margin-bottom:25px;}}
            .stat-card{{flex:1;background:#1a1a2e;border:1px solid #2a2a40;border-radius:8px;padding:18px;text-align:center;}}
            .stat-value{{font-size:28px;font-weight:bold;color:#d0d8ff;}}
            .stat-label{{color:#9090b0;font-size:11px;margin-top:4px;}}
            .section{{margin-bottom:22px;}}
            .section h2{{color:#7c8aff;font-size:16px;border-bottom:1px solid #2a2a40;padding-bottom:6px;}}
            table{{width:100%;border-collapse:collapse;margin-top:10px;}}
            th{{background:#1a1a2e;padding:8px 10px;text-align:left;color:#7c8aff;font-size:11px;border-bottom:2px solid #2a2a40;}}
            td{{padding:7px 10px;border-bottom:1px solid #1a1a2e;font-size:12px;}}
            .footer{{text-align:center;margin-top:35px;padding-top:15px;border-top:1px solid #2a2a40;color:#9090b0;font-size:10px;}}
        </style></head><body>
            <div class="header"><h1>🚀 REPORTE DE MISIÓN</h1><p>Estación Espacial Evaluadora — {fecha_actual}</p></div>
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Misiones Totales</div></div>
                <div class="stat-card"><div class="stat-value">{promedio}</div><div class="stat-label">Promedio General</div></div>
                <div class="stat-card"><div class="stat-value">{porcentaje_aprobados}%</div><div class="stat-label">Tasa de Aprobación</div></div>
                <div class="stat-card"><div class="stat-value">{aprobados}/{reprobados}</div><div class="stat-label">Aprobados/Reprobados</div></div>
            </div>
            <div class="section"><h2>📊 Distribución de Notas</h2><table>
                <tr><th>Categoría</th><th>Cantidad</th></tr>
                <tr><td>🌟 Excelente (9-10)</td><td>{excelente}</td></tr>
                <tr><td>🛰️ Bueno (7-8)</td><td>{bueno}</td></tr>
                <tr><td>🔧 Regular (5-6)</td><td>{regular}</td></tr>
                <tr><td>⚠️ Malo (&lt;5)</td><td>{malo}</td></tr>
            </table></div>
            <div class="section"><h2>📋 Rendimiento por Misión</h2><table>
                <tr><th>Misión</th><th>Total</th><th>Promedio</th><th>Máxima</th><th>Mínima</th></tr>
                {filas_mision}</table></div>
            <div class="section"><h2>🕐 Últimas Misiones</h2><table>
                <tr><th>Tripulante</th><th>Misión</th><th>Puntuación</th><th>Fecha</th></tr>
                {filas_ultimas}</table></div>
            <div class="footer"><p>📡 Estación Espacial Evaluadora v2.0 | Reporte generado automáticamente</p><p>© {año_actual}</p></div>
        </body></html>"""
        
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        pdf = pdfkit.from_string(html_reporte, False, configuration=config)
        
        from flask import make_response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=reporte_mision.pdf'
        return response
        
    except Exception as e:
        print(f"❌ Error al generar PDF: {e}")
        flash("❌ Error al generar el PDF.")
        return redirect(url_for("dashboard"))


# ============================================================
# INICIO
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)