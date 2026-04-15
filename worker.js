// worker.js — Chatbot RAG Moral General y Especial
// Query Translation: español → latín escolástico → embedding → respuesta en español
// Deploy: npx wrangler deploy

const EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free";
const CHAT_MODEL      = "anthropic/claude-haiku-4-5";
const CORS_ORIGIN     = "*";

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return corsResponse();
    if (request.method !== "POST") {
      return jsonResponse({ error: "Método no permitido" }, 405);
    }

    try {
      const { pregunta, session_id, filtro_fuente, historial } = await request.json();

      if (!pregunta?.trim()) {
        return jsonResponse({ error: "La pregunta no puede estar vacía" }, 400);
      }

      // 1. Traducir la pregunta al latín escolástico
      const preguntaLatina = await traducirAlLatin(pregunta, env.OPENROUTER_KEY);

      // 2. Generar embedding del latín
      const embedding = await generarEmbedding(preguntaLatina, env.OPENROUTER_KEY);
      if (!embedding) {
        return jsonResponse({ error: "Error generando embedding" }, 500);
      }

      // 3. Buscar fragmentos relevantes en Neon
      const fragmentos = await buscarFragmentos(embedding, env, filtro_fuente);

      // 4. Si no hay fragmentos relevantes, responder sin inventar
      if (fragmentos.length === 0) {
        return jsonResponse({
          respuesta: "No encontré información sobre ese tema en las fuentes de la materia (Santo Tomás, Royo Marín y el Magisterio de la Iglesia). Te sugiero consultar directamente los textos o reformular la pregunta.",
          fuentes: [],
          session_id,
          pregunta_latina: preguntaLatina
        });
      }

      // 5. Generar respuesta en español traduciendo e interpretando el latín
      const { respuesta, fuentes } = await generarRespuesta(
        pregunta,
        preguntaLatina,
        fragmentos,
        historial || [],
        env.OPENROUTER_KEY
      );

      // 6. Guardar en historial (sin await para no bloquear)
      guardarHistorial(session_id, pregunta, respuesta, fuentes, env).catch(() => {});

      return jsonResponse({ respuesta, fuentes, session_id, pregunta_latina: preguntaLatina });

    } catch (error) {
      console.error("Error en el worker:", error);
      return jsonResponse({ error: "Error interno del servidor" }, 500);
    }
  }
};

// =============================================================
// PASO 1: Traducir pregunta al latín escolástico
// =============================================================

async function traducirAlLatin(pregunta, apiKey) {
  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: CHAT_MODEL,
        max_tokens: 200,
        temperature: 0.1,
        messages: [{
          role: "user",
          content: `Eres un experto en latín escolástico medieval. Traduce la siguiente pregunta al latín escolástico que usaría Santo Tomás de Aquino. Responde ÚNICAMENTE con la traducción latina, sin explicaciones ni comentarios.

Pregunta: "${pregunta}"

Traducción latina:`
        }]
      })
    });

    const data = await response.json();
    const traduccion = data?.choices?.[0]?.message?.content?.trim();
    if (!traduccion || traduccion.length < 5) return pregunta;
    return traduccion;

  } catch (e) {
    return pregunta;
  }
}

// =============================================================
// PASO 2: Generar embedding
// =============================================================

async function generarEmbedding(texto, apiKey) {
  try {
    const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: texto
      })
    });
    const data = await response.json();
    return data?.data?.[0]?.embedding ?? null;
  } catch (e) {
    return null;
  }
}

// =============================================================
// PASO 3: Buscar fragmentos en Neon
// =============================================================

async function buscarFragmentos(embedding, env, filtroFuente = null) {
  const embStr = "[" + embedding.join(",") + "]";

  const query = filtroFuente
    ? `SELECT id, fuente, obra, referencia, contenido, 1 - (embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048)) AS similitud FROM fragmentos WHERE 1 - (embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048)) > 0.30 AND fuente = '${filtroFuente}' ORDER BY embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048) LIMIT 6`
    : `SELECT id, fuente, obra, referencia, contenido, 1 - (embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048)) AS similitud FROM fragmentos WHERE 1 - (embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048)) > 0.30 ORDER BY embedding::halfvec(2048) <=> '${embStr}'::halfvec(2048) LIMIT 6`;

  const response = await fetch(
    "https://ep-hidden-mode-anw2shmo.c-6.us-east-1.aws.neon.tech/sql",
    {
      method: "POST",
      headers: {
        "Neon-Connection-String": env.NEON_FULL_URL,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    }
  );

  if (!response.ok) {
    console.error("Error Neon:", await response.text());
    return [];
  }

  const data = await response.json();
  return data.rows || [];
}

// =============================================================
// PASO 4: Generar respuesta en español
// =============================================================

async function generarRespuesta(preguntaEspanol, preguntaLatina, fragmentos, historial, apiKey) {

  const contexto = fragmentos.map((f, i) =>
    `[${i + 1}] ${f.obra} (${f.referencia}):\n"${f.contenido}"`
  ).join("\n\n");

  const fuentes = fragmentos.map(f => ({
    fuente: f.fuente,
    obra: f.obra,
    referencia: f.referencia,
    similitud: Math.round(f.similitud * 100)
  }));

  const mensajesHistorial = (historial || []).flatMap(h => [
    { role: "user", content: h.pregunta },
    { role: "assistant", content: h.respuesta }
  ]);

  const prompt = `Eres un experto en teología y filosofía tomista. Tu tarea es responder preguntas sobre moral católica basándote EXCLUSIVAMENTE en los textos de Santo Tomás de Aquino proporcionados como contexto.

INSTRUCCIONES:
- Los textos del contexto están en latín medieval. Debes TRADUCIR e INTERPRETAR cada fragmento relevante.
- Responde ÚNICAMENTE basándote en los textos proporcionados. NO agregues información externa.
- Citá siempre la obra y referencia usando el número entre corchetes: [1], [2], etc.
- Respondé en español claro y accesible para seminaristas.
- Cuando cites a Tomás, indicá la referencia exacta (ej: S.Th. I-II, q.94, a.2).
- Estructurá la respuesta: primero la respuesta directa, luego el desarrollo con citas y traducciones.
- Si el texto latino es relevante, incluyelo entre comillas seguido de su traducción.
- Si los textos no responden completamente la pregunta, indicalo.

TEXTOS EN LATÍN (con referencias):
${contexto}

PREGUNTA: ${preguntaEspanol}
(Versión latina usada para la búsqueda: "${preguntaLatina}")

RESPUESTA EN ESPAÑOL:`;

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://moralgeneralzc.pages.dev",
      "X-Title": "Moral General ZC"
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        ...mensajesHistorial,
        { role: "user", content: prompt }
      ],
      max_tokens: 2000,
      temperature: 0.2
    })
  });

  const data = await response.json();
  const respuesta = data?.choices?.[0]?.message?.content ?? "No se pudo generar una respuesta.";

  return { respuesta, fuentes };
}

// =============================================================
// HISTORIAL (usa Neon también)
// =============================================================

async function guardarHistorial(sessionId, pregunta, respuesta, fuentes, env) {
  if (!sessionId) return;
  const query = `
    INSERT INTO conversaciones (session_id, pregunta, respuesta, fuentes_usadas)
    VALUES ($1, $2, $3, $4)
  `;
  await fetch(
    "https://ep-hidden-mode-anw2shmo.c-6.us-east-1.aws.neon.tech/sql",
    {
      method: "POST",
      headers: {
        "Neon-Connection-String": env.NEON_FULL_URL,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query,
        params: [sessionId, pregunta, respuesta, JSON.stringify(fuentes)]
      })
    }
  );
}

// =============================================================
// HELPERS
// =============================================================

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": CORS_ORIGIN,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}

function corsResponse() {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": CORS_ORIGIN,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}