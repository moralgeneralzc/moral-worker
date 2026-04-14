"""
procesar_pdfs.py
================
Procesa PDFs teologicos, los divide en chunks y sube los embeddings a Supabase.
Usa OpenRouter (NVIDIA llama-nemotron-embed-vl-1b-v2:free) — costo $0.

Instalacion de dependencias:
  pip install pypdf supabase python-dotenv requests

Estructura de carpetas:
  /pdfs
    /santo_tomas/   -> PDFs de Santo Tomas
    /royo_marin/    -> PDFs de Royo Marin
    /magisterio/    -> PDFs del Magisterio

Archivo .env requerido:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_KEY=eyJ...   (service_role key)
  OPENROUTER_KEY=sk-or-...
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import create_client, Client

load_dotenv()

# =============================================================
# CONFIGURACION
# =============================================================

SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# Mapeo carpeta -> nombre de fuente en la BD
FUENTES = {
    "santo_tomas": "Santo Tomas",
    "royo_marin":  "Royo Marin",
    "magisterio":  "Magisterio",
}

# Mapeo nombre de archivo (sin extension) -> (obra, referencia base)
# Agregar aca los nombres exactos de tus archivos
OBRAS = {
    "suma_theologica_i":          ("Suma Teologica", "S.Th. Prima Pars"),
    "suma_theologica_i_ii":       ("Suma Teologica", "S.Th. Prima Secundae"),
    "suma_theologica_ii_ii":      ("Suma Teologica", "S.Th. Secunda Secundae"),
    "suma_theologica_iii":        ("Suma Teologica", "S.Th. Tertia Pars"),
    "suma_contra_gentiles":       ("Suma Contra los Gentiles", "SCG"),
    "cuestiones_disputadas":      ("Cuestiones Disputadas", "QD"),
    "teologia_moral_i":           ("Teologia Moral para Seglares", "TM vol.I"),
    "teologia_moral_ii":          ("Teologia Moral para Seglares", "TM vol.II"),
    "veritatis_splendor":         ("Veritatis Splendor", "VS"),
    "humanae_vitae":              ("Humanae Vitae", "HV"),
    "amoris_laetitia":            ("Amoris Laetitia", "AL"),
    "evangelium_vitae":           ("Evangelium Vitae", "EV"),
    "catecismo":                  ("Catecismo de la Iglesia Catolica", "CIC"),
    "gaudium_et_spes":            ("Gaudium et Spes", "GS"),
    "familiaris_consortio":       ("Familiaris Consortio", "FC"),
    "deus_caritas_est":           ("Deus Caritas Est", "DCE"),
}

CARPETA_PDFS   = Path("./pdfs")
CHUNK_PALABRAS = 450
CHUNK_OVERLAP  = 50
PAUSA_EMBED    = 0.5

# =============================================================
# INICIALIZACION
# =============================================================

def verificar_config():
    errores = []
    if not SUPABASE_URL:
        errores.append("Falta SUPABASE_URL en el .env")
    if not SUPABASE_KEY:
        errores.append("Falta SUPABASE_KEY en el .env")
    if not OPENROUTER_KEY:
        errores.append("Falta OPENROUTER_KEY en el .env")
    if errores:
        for e in errores:
            print(f"  ❌ {e}")
        exit(1)

verificar_config()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================================================
# FUNCIONES
# =============================================================

def extraer_texto_pdf(ruta_pdf: Path) -> str:
    """Extrae todo el texto de un PDF pagina por pagina."""
    reader = PdfReader(str(ruta_pdf))
    paginas = []
    for pagina in reader.pages:
        texto = pagina.extract_text()
        if texto:
            paginas.append(texto.strip())
    return "\n\n".join(paginas)


def dividir_en_chunks(texto: str, palabras_por_chunk: int, overlap: int) -> list:
    """Divide el texto en chunks de N palabras con superposicion entre chunks."""
    palabras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + palabras_por_chunk
        chunk = " ".join(palabras[inicio:fin])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        inicio += palabras_por_chunk - overlap
    return chunks


def generar_embedding(texto: str) -> list:
    """Genera embedding con NVIDIA via OpenRouter (gratis, 2048 dimensiones)."""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": texto
            },
            timeout=30
        )
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"\n    ⚠️  Error generando embedding: {e}")
        return None


def ya_procesado(obra: str) -> bool:
    """Verifica si esta obra ya fue procesada (permite retomar si se interrumpe)."""
    resultado = supabase.table("fragmentos") \
        .select("id") \
        .eq("obra", obra) \
        .limit(1) \
        .execute()
    return len(resultado.data) > 0


def subir_chunk(fuente: str, obra: str, referencia: str, contenido: str, embedding: list):
    """Sube un fragmento con su embedding a Supabase."""
    supabase.table("fragmentos").insert({
        "fuente":     fuente,
        "obra":       obra,
        "referencia": referencia,
        "contenido":  contenido,
        "embedding":  embedding,
    }).execute()


def normalizar_nombre(nombre: str) -> str:
    """Convierte el nombre del archivo a clave del diccionario OBRAS."""
    return nombre.lower() \
        .replace(" ", "_") \
        .replace("-", "_") \
        .replace("á", "a").replace("é", "e").replace("í", "i") \
        .replace("ó", "o").replace("ú", "u").replace("ü", "u") \
        .replace("ñ", "n").replace("ä", "a").replace("ö", "o")


def procesar_pdf(ruta_pdf: Path, nombre_fuente: str) -> int:
    """Pipeline completo: extraer texto -> chunking -> embeddings -> subir a Supabase."""

    nombre_archivo = normalizar_nombre(ruta_pdf.stem)

    # Buscar en OBRAS por coincidencia exacta primero, luego parcial
    obra, referencia_base = None, None
    if nombre_archivo in OBRAS:
        obra, referencia_base = OBRAS[nombre_archivo]
    else:
        for clave, (o, r) in OBRAS.items():
            if clave in nombre_archivo or nombre_archivo in clave:
                obra, referencia_base = o, r
                break

    if not obra:
        # Fallback: usar el nombre del archivo limpio
        obra = ruta_pdf.stem.replace("_", " ").replace("-", " ")
        referencia_base = obra[:40]
        print(f"    ℹ️  No se encontro en OBRAS — usando: '{obra}'")
        print(f"    ℹ️  Para mejor precision, agregar al diccionario OBRAS en el script.")

    if ya_procesado(obra):
        print(f"    ⏭️  Ya procesado — saltando.")
        return 0

    print(f"    📄 Extrayendo texto...")
    texto = extraer_texto_pdf(ruta_pdf)

    if not texto.strip():
        print(f"    ⚠️  PDF sin texto extraible (probablemente escaneado/imagen).")
        return 0

    palabras_total = len(texto.split())
    print(f"    📝 {palabras_total:,} palabras extraidas")

    print(f"    ✂️  Dividiendo en chunks de {CHUNK_PALABRAS} palabras...")
    chunks = dividir_en_chunks(texto, CHUNK_PALABRAS, CHUNK_OVERLAP)
    print(f"    → {len(chunks)} chunks generados")

    subidos = 0
    errores = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"    🔢 Chunk {i}/{len(chunks)} — subiendo...          ", end="\r")

        embedding = generar_embedding(chunk)
        if embedding is None:
            errores += 1
            continue

        referencia = f"{referencia_base} [fragmento {i}/{len(chunks)}]"
        subir_chunk(
            fuente=nombre_fuente,
            obra=obra,
            referencia=referencia,
            contenido=chunk,
            embedding=embedding,
        )
        subidos += 1
        time.sleep(PAUSA_EMBED)

    if errores > 0:
        print(f"    ⚠️  {subidos} subidos, {errores} errores.                    ")
    else:
        print(f"    ✅ {subidos}/{len(chunks)} chunks subidos correctamente.      ")

    return subidos


# =============================================================
# EJECUCION PRINCIPAL
# =============================================================

def main():
    print("\n" + "="*60)
    print("  PROCESADOR DE PDFs")
    print("  Moral General y Especial — Seminario Diocesano")
    print(f"  Modelo: {EMBEDDING_MODEL}")
    print("="*60 + "\n")

    if not CARPETA_PDFS.exists():
        print(f"❌ No existe la carpeta '{CARPETA_PDFS}'.")
        print(f"   Crea la carpeta y pon los PDFs adentro.")
        return

    total_chunks    = 0
    total_archivos  = 0
    archivos_error  = []

    for carpeta_fuente in sorted(CARPETA_PDFS.iterdir()):
        if not carpeta_fuente.is_dir():
            continue

        nombre_fuente = FUENTES.get(carpeta_fuente.name, carpeta_fuente.name.title())
        pdfs = sorted(carpeta_fuente.glob("*.pdf"))

        if not pdfs:
            print(f"📂 {carpeta_fuente.name}/ — sin PDFs, saltando\n")
            continue

        print(f"📚 FUENTE: {nombre_fuente} ({len(pdfs)} archivo/s)\n")

        for pdf in pdfs:
            print(f"  → {pdf.name}")
            try:
                chunks = procesar_pdf(pdf, nombre_fuente)
                total_chunks += chunks
                total_archivos += 1
            except Exception as e:
                print(f"    ❌ Error procesando: {e}")
                archivos_error.append(pdf.name)
            print()

    print("="*60)
    print(f"  ✅ PROCESO FINALIZADO")
    print(f"  Archivos procesados : {total_archivos}")
    print(f"  Chunks subidos      : {total_chunks:,}")
    if archivos_error:
        print(f"  Archivos con error  : {len(archivos_error)}")
        for a in archivos_error:
            print(f"    - {a}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()