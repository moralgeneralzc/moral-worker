"""
scraper_corpus_thomisticum.py
==============================
Scrapea toda la Opera Omnia de Santo Tomas del Corpus Thomisticum
y sube los fragmentos con embeddings a Supabase.

El texto es en latin. El LLM del chatbot traduce al responder.

Instalacion:
  pip install requests beautifulsoup4 supabase python-dotenv

Archivo .env requerido:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_KEY=eyJ...   (service_role key)
  OPENROUTER_KEY=sk-or-...

Uso:
  python scraper_corpus_thomisticum.py

El script es reanudable: si se interrumpe, retoma desde donde quedo.
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from supabase import create_client, Client

load_dotenv()

# =============================================================
# CONFIGURACION
# =============================================================

SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

BASE_URL        = "https://www.corpusthomisticum.org"
INDEX_URL       = f"{BASE_URL}/iopera.html"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
FUENTE          = "Santo Tomas"

CHUNK_PALABRAS = 400   # Un poco menor para textos latinos (palabras mas largas)
CHUNK_OVERLAP  = 40
PAUSA_SCRAPING = 1.5   # Segundos entre requests al sitio (respetar el servidor)
PAUSA_EMBED    = 0.5   # Segundos entre llamadas a OpenRouter

# Archivo para guardar progreso (URLs ya procesadas)
PROGRESO_FILE  = Path("./progreso_corpus.txt")

# =============================================================
# INICIALIZACION
# =============================================================

def verificar_config():
    errores = []
    if not SUPABASE_URL:  errores.append("Falta SUPABASE_URL en el .env")
    if not SUPABASE_KEY:  errores.append("Falta SUPABASE_KEY en el .env")
    if not OPENROUTER_KEY: errores.append("Falta OPENROUTER_KEY en el .env")
    if errores:
        for e in errores: print(f"  ❌ {e}")
        exit(1)

verificar_config()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================================================
# FUNCIONES DE SCRAPING
# =============================================================

def get_html(url: str, reintentos: int = 3) -> BeautifulSoup | None:
    """Descarga una pagina y devuelve el objeto BeautifulSoup."""
    for intento in range(reintentos):
        try:
            response = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (research/academic bot)"
            })
            if response.status_code == 200:
                response.encoding = "latin-1"  # El sitio usa latin-1
                return BeautifulSoup(response.text, "html.parser")
            else:
                print(f"    ⚠️  HTTP {response.status_code} para {url}")
        except Exception as e:
            print(f"    ⚠️  Error (intento {intento+1}): {e}")
            time.sleep(2)
    return None


def extraer_links_index() -> list[dict]:
    """
    Parsea iopera.html y extrae todos los links a obras con su contexto.
    Devuelve lista de {url, obra} donde obra es el nombre de la seccion.
    """
    print(f"📖 Descargando indice: {INDEX_URL}")
    soup = get_html(INDEX_URL)
    if not soup:
        print("❌ No se pudo descargar el indice.")
        return []

    links = []
    obra_actual = "Opera Omnia"

    # Recorrer todos los elementos buscando titulos de obras y links
    for elemento in soup.find_all(["td", "p", "h1", "h2", "h3", "b", "a"]):
        texto = elemento.get_text(strip=True)

        # Detectar titulos de obras (texto en negrita sin link)
        if elemento.name in ["b", "h2", "h3"] and texto and not elemento.find("a"):
            if len(texto) > 3:
                obra_actual = texto

        # Detectar links a obras
        if elemento.name == "a":
            href = elemento.get("href", "")
            if href and href.endswith(".html") and "iopera" not in href:
                url_completa = href if href.startswith("http") else f"{BASE_URL}/{href}"
                links.append({
                    "url": url_completa,
                    "obra_seccion": obra_actual,
                    "titulo_link": texto
                })

    # Deduplicar por URL
    vistos = set()
    links_unicos = []
    for link in links:
        if link["url"] not in vistos:
            vistos.add(link["url"])
            links_unicos.append(link)

    print(f"   → {len(links_unicos)} paginas encontradas en el indice\n")
    return links_unicos


def extraer_contenido_pagina(url: str) -> dict | None:
    """
    Extrae el titulo de la obra y el texto de una pagina del Corpus Thomisticum.
    Devuelve {obra, referencia, texto} o None si falla.
    """
    soup = get_html(url)
    if not soup:
        return None

    # El titulo de la pagina contiene obra + seccion
    # Ej: "Thomas de Aquino, Scriptum super Sententiis, q. 1"
    titulo = soup.find("title")
    titulo_texto = titulo.get_text(strip=True) if titulo else ""

    # Limpiar el titulo
    titulo_texto = titulo_texto.replace("Thomas de Aquino, ", "").strip()
    titulo_texto = titulo_texto.replace("Thomas Aquinas, ", "").strip()

    # Separar obra de referencia
    # "Scriptum super Sententiis, q. 1" -> obra="Scriptum super Sententiis", ref="q. 1"
    partes = titulo_texto.split(", ", 1)
    if len(partes) >= 2:
        obra = partes[0].strip()
        referencia = partes[1].strip()
    else:
        obra = titulo_texto
        referencia = ""

    # Extraer el texto principal del contenido
    # El sitio tiene el texto en parrafos dentro del body
    # Eliminar navegacion, scripts, estilos
    for tag in soup.find_all(["script", "style", "img", "a"]):
        tag.decompose()

    # Tomar el texto del body completo
    body = soup.find("body")
    if not body:
        return None

    texto = body.get_text(separator=" ", strip=True)

    # Limpiar ruido comun del sitio
    ruido = [
        "CORPUS THOMISTICUM",
        "Textum Parmae 1856 editum",
        "et automato translatum a Roberto Busa SJ in taenias magneticas",
        "denuo recognovit Enrique Alarcón atque instruxit",
        "Thomas de Aquino",
        "Thomas Aquinas",
    ]
    for r in ruido:
        texto = texto.replace(r, "")

    # Eliminar espacios multiples
    import re
    texto = re.sub(r'\s+', ' ', texto).strip()

    if len(texto) < 100:
        return None

    return {
        "obra": obra,
        "referencia": referencia if referencia else obra,
        "texto": texto,
        "url": url
    }


# =============================================================
# FUNCIONES DE EMBEDDINGS Y SUPABASE
# =============================================================

def dividir_en_chunks(texto: str) -> list[str]:
    palabras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + CHUNK_PALABRAS
        chunk = " ".join(palabras[inicio:fin])
        if len(chunk.strip()) > 80:
            chunks.append(chunk)
        inicio += CHUNK_PALABRAS - CHUNK_OVERLAP
    return chunks


def generar_embedding(texto: str) -> list | None:
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": EMBEDDING_MODEL, "input": texto},
            timeout=30
        )
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"\n    ⚠️  Error embedding: {e}")
        return None


def pagina_ya_procesada(url: str) -> bool:
    """Usa archivo local de progreso para verificar si ya fue procesada."""
    if not PROGRESO_FILE.exists():
        return False
    return url in PROGRESO_FILE.read_text(encoding="utf-8")


def marcar_procesada(url: str):
    with open(PROGRESO_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def subir_fragmento(obra: str, referencia: str, contenido: str, url: str, embedding: list):
    supabase.table("fragmentos").insert({
        "fuente":     FUENTE,
        "obra":       obra,
        "referencia": referencia,
        "contenido":  contenido,
        "embedding":  embedding,
    }).execute()


def procesar_pagina(link: dict) -> int:
    """Procesa una pagina: extrae, divide, embeds y sube. Devuelve chunks subidos."""
    url = link["url"]

    if pagina_ya_procesada(url):
        return 0

    contenido = extraer_contenido_pagina(url)
    time.sleep(PAUSA_SCRAPING)

    if not contenido:
        marcar_procesada(url)
        return 0

    chunks = dividir_en_chunks(contenido["texto"])
    if not chunks:
        marcar_procesada(url)
        return 0

    subidos = 0
    for i, chunk in enumerate(chunks, 1):
        embedding = generar_embedding(chunk)
        if not embedding:
            continue

        # La referencia incluye el numero de chunk para multiples chunks por pagina
        ref = contenido["referencia"]
        if len(chunks) > 1:
            ref = f"{ref} [{i}/{len(chunks)}]"

        subir_fragmento(
            obra=contenido["obra"],
            referencia=ref,
            contenido=chunk,
            url=url,
            embedding=embedding
        )
        subidos += 1
        time.sleep(PAUSA_EMBED)

    marcar_procesada(url)
    return subidos


# =============================================================
# EJECUCION PRINCIPAL
# =============================================================

def main():
    print("\n" + "="*60)
    print("  SCRAPER — Corpus Thomisticum")
    print("  Opera Omnia de Santo Tomas de Aquino")
    print(f"  Modelo: {EMBEDDING_MODEL}")
    print("="*60 + "\n")

    # Obtener todos los links del indice
    links = extraer_links_index()
    if not links:
        return

    # Contar cuantos ya fueron procesados
    ya_procesados = sum(1 for l in links if pagina_ya_procesada(l["url"]))
    pendientes = len(links) - ya_procesados

    print(f"📊 Estado:")
    print(f"   Total de paginas : {len(links)}")
    print(f"   Ya procesadas    : {ya_procesados}")
    print(f"   Pendientes       : {pendientes}")

    if pendientes == 0:
        print("\n✅ Todas las paginas ya fueron procesadas.")
        return

    tiempo_estimado = pendientes * (PAUSA_SCRAPING + PAUSA_EMBED * 3) / 60
    print(f"   Tiempo estimado  : ~{tiempo_estimado:.0f} minutos\n")

    total_chunks = 0
    total_paginas = 0
    obra_actual = ""

    for i, link in enumerate(links, 1):
        if pagina_ya_procesada(link["url"]):
            continue

        # Mostrar cuando cambia de obra
        if link["obra_seccion"] != obra_actual:
            obra_actual = link["obra_seccion"]
            print(f"\n📚 {obra_actual}")

        titulo = link["titulo_link"] or link["url"].split("/")[-1]
        print(f"  [{i}/{len(links)}] {titulo} ...", end=" ", flush=True)

        try:
            chunks = procesar_pagina(link)
            total_chunks += chunks
            total_paginas += 1
            print(f"✅ {chunks} chunks")
        except Exception as e:
            print(f"❌ {e}")
            marcar_procesada(link["url"])

    print("\n" + "="*60)
    print(f"  ✅ SCRAPING FINALIZADO")
    print(f"  Paginas procesadas : {total_paginas}")
    print(f"  Chunks subidos     : {total_chunks:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
