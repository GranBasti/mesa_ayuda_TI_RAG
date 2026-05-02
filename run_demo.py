import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ingest import build_collection
from rag_pipeline import answer_ticket


def print_response(response):
    print("\n" + "=" * 60)
    print("RESPUESTA DEL ASISTENTE")
    print("=" * 60)
    print(f"Categoría: {response.get('categoria', '')}")
    print(f"Subcategoría: {response.get('subcategoria', '')}")
    print(f"Prioridad: {response.get('prioridad', '')}")
    print(f"Sistema afectado: {response.get('sistema_afectado', '')}")
    print(f"Diagnóstico probable: {response.get('diagnostico_probable', '')}")
    print(f"Respuesta al usuario: {response.get('respuesta_usuario', '')}")
    print(f"Acciones soporte: {response.get('acciones_soporte', [])}")
    print(f"Requiere escalamiento: {response.get('requiere_escalamiento', '')}")
    print(f"Justificación: {response.get('justificacion', '')}")
    print("=" * 60 + "\n")


def main():
    print("Inicializando base de conocimiento...")
    build_collection(reset=False)
    print("Sistema listo.")
    print("Modo disponible: zero-shot / few-shot")
    print("Escribe 'salir' para terminar.\n")

    while True:
        mode = input("Modo (zero-shot/few-shot): ").strip().lower()
        if mode == "salir":
            break
        if mode not in ["zero-shot", "few-shot"]:
            print("Modo no válido.")
            continue

        user_input = input("Usuario: ").strip()
        if user_input.lower() == "salir":
            break
        if not user_input:
            print("Debes escribir un problema.")
            continue

        try:
            response, contexts = answer_ticket(user_input, mode=mode)
            print_response(response)

            print("Contextos recuperados:")
            for i, (_, meta, dist) in enumerate(contexts, start=1):
                print(f"{i}. categoria={meta.get('categoria')} | subcategoria={meta.get('subcategoria')} | sistema={meta.get('sistema')} | distancia={dist:.4f}")
            print()

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()