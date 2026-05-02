import json
from ingest import build_collection
from retrieve import retrieve_context
from github_models_client import ask_github_model

SYSTEM_PROMPT = """
Eres un asistente experto en mesa de ayuda TI universitaria.

Tu trabajo es:
- entender el problema escrito por el usuario,
- analizar contexto recuperado desde tickets similares,
- clasificar el incidente,
- proponer una respuesta clara para el usuario,
- recomendar acciones al personal de soporte,
- decidir si requiere escalamiento.

Reglas:
- Usa solo el contexto entregado.
- No inventes políticas ni procedimientos.
- Si falta información, indícalo.
- Responde SOLO en JSON válido.
""".strip()

FEW_SHOT_EXAMPLES = """
Ejemplo 1
Entrada:
No puedo iniciar sesión en mi correo institucional, dice contraseña incorrecta.

Salida:
{
  "categoria": "Acceso a cuentas",
  "subcategoria": "Recuperación de contraseña",
  "prioridad": "Alta",
  "sistema_afectado": "Correo institucional",
  "diagnostico_probable": "Posible error de credenciales o bloqueo temporal por intentos fallidos",
  "respuesta_usuario": "Te recomendamos restablecer tu contraseña desde el portal institucional y volver a intentar el acceso.",
  "acciones_soporte": [
    "Validar identidad del usuario",
    "Verificar estado de la cuenta y aplicar procedimiento de recuperación"
  ],
  "requiere_escalamiento": "No",
  "justificacion": "Es un problema frecuente de acceso que puede resolverse en primer nivel."
}

Ejemplo 2
Entrada:
La VPN institucional no conecta desde mi casa y necesito acceso remoto urgente.

Salida:
{
  "categoria": "Conectividad",
  "subcategoria": "Problema VPN",
  "prioridad": "Alta",
  "sistema_afectado": "VPN institucional",
  "diagnostico_probable": "Fallo de conexión remota o configuración incorrecta del cliente VPN",
  "respuesta_usuario": "Verifica tu conexión a internet, credenciales y configuración del cliente VPN. Si el problema persiste, se revisará el acceso remoto.",
  "acciones_soporte": [
    "Validar credenciales y permisos de acceso remoto",
    "Revisar configuración del cliente VPN y estado del servicio"
  ],
  "requiere_escalamiento": "Si",
  "justificacion": "Los problemas VPN pueden requerir validación técnica adicional y afectar acceso remoto crítico."
}

Ejemplo 3
Entrada:
Las clases grabadas no cargan en la plataforma educativa.

Salida:
{
  "categoria": "Plataforma educativa",
  "subcategoria": "Contenido no disponible",
  "prioridad": "Media",
  "sistema_afectado": "LMS",
  "diagnostico_probable": "Fallo de carga de contenido o recurso no disponible en la plataforma",
  "respuesta_usuario": "Te recomendamos actualizar la página, probar desde otro navegador y verificar si el contenido sigue disponible más tarde.",
  "acciones_soporte": [
    "Verificar disponibilidad del recurso en la plataforma",
    "Revisar si existe incidencia general del LMS"
  ],
  "requiere_escalamiento": "No",
  "justificacion": "El problema puede estar asociado a contenido o caché y normalmente se revisa en primer nivel."
}
""".strip()


def build_prompt(ticket_text: str, contexts, mode="zero-shot"):
    bloques = []

    for i, (doc, meta, dist) in enumerate(contexts, start=1):
        bloques.append(
            f"Contexto {i}:\n"
            f"{doc}\n"
            f"Metadata: categoria={meta.get('categoria')}, "
            f"subcategoria={meta.get('subcategoria')}, "
            f"prioridad={meta.get('prioridad')}, "
            f"sistema={meta.get('sistema')}, "
            f"escalamiento={meta.get('escalamiento')}, "
            f"distancia={dist:.4f}\n"
        )

    context_text = "\n\n".join(bloques)

    base_instruction = f"""
Caso: Mesa de ayuda TI universitaria.

Problema ingresado por el usuario:
{ticket_text}

Contexto recuperado:
{context_text}

Devuelve SOLO este JSON:
{{
  "categoria": "",
  "subcategoria": "",
  "prioridad": "",
  "sistema_afectado": "",
  "diagnostico_probable": "",
  "respuesta_usuario": "",
  "acciones_soporte": ["", ""],
  "requiere_escalamiento": "Si o No",
  "justificacion": ""
}}
""".strip()

    if mode == "few-shot":
        return FEW_SHOT_EXAMPLES + "\n\n" + base_instruction

    return base_instruction


def answer_ticket(ticket_text: str, mode="zero-shot"):
    build_collection(reset=False)
    contexts = retrieve_context(ticket_text)
    prompt = build_prompt(ticket_text, contexts, mode=mode)
    raw_response = ask_github_model(SYSTEM_PROMPT, prompt)

    try:
        parsed = json.loads(raw_response)
    except Exception:
        parsed = {
            "categoria": "",
            "subcategoria": "",
            "prioridad": "",
            "sistema_afectado": "",
            "diagnostico_probable": "",
            "respuesta_usuario": raw_response,
            "acciones_soporte": [],
            "requiere_escalamiento": "",
            "justificacion": "La respuesta no vino en JSON válido."
        }

    return parsed, contexts