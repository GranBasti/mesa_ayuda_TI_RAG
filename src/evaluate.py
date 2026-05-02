import pandas as pd
from ingest import load_dataset, build_collection
from rag_pipeline import answer_ticket


def norm(x):
    return str(x).strip().lower()


def evaluate(n=5):
    build_collection(reset=False)
    df = load_dataset().head(n)
    rows = []

    for _, row in df.iterrows():
        pred, _ = answer_ticket(row["mensajeusuario"])

        rows.append({
            "ticketid": row["ticketid"],
            "categoria_real": row["categoria"],
            "categoria_pred": pred.get("categoria", ""),
            "prioridad_real": row["prioridad"],
            "prioridad_pred": pred.get("prioridad", ""),
            "sistema_real": row["sistemaafectado"],
            "sistema_pred": pred.get("sistema_afectado", ""),
            "match_categoria": norm(row["categoria"]) == norm(pred.get("categoria", "")),
            "match_prioridad": norm(row["prioridad"]) == norm(pred.get("prioridad", "")),
            "match_sistema": norm(row["sistemaafectado"]) == norm(pred.get("sistema_afectado", ""))
        })

    out = pd.DataFrame(rows)
    print(out)
    return out


if __name__ == "__main__":
    evaluate(5)