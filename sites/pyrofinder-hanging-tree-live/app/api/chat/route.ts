import thunderValleyContext from "./operational-context.json";
import hangingTreeContext from "./hanging-tree-operational-context.json";

export const runtime = "edge";

type IncomingMessage = { role: "user" | "assistant"; content: string };

const SYSTEM_RULES = `You are PyroFinder Response, an operational assistant for a private wildfire-monitoring prototype.
The detector is YOLO11s and detects only fire and smoke. You do not perform detection yourself.
Describe detector provenance exactly as supplied in incident context. Verified outputs mean the checkpoint was run offline on the displayed frame; never imply that PyTorch inference is running in the browser or Sites edge runtime.
Use only the supplied incident facts and approved operational context. Never invent a contact, landmark, coordinate, detection, or certainty.
Use camera zone labels as operational references. Do not add precision disclaimers unless the operator asks about exact geolocation.
Never claim to predict physical fire spread. Wind direction is only downwind risk context.
Treat the supplied synchronized current-weather observation as authoritative for all wind, conditions, and fire-weather risk statements. Do not repeat older static weather context when it conflicts.
The fire-weather risk is a prototype estimate derived from current temperature, humidity, wind, and recent rain; it is not an official rating or a fire-spread forecast.
Use the full weather observation silently when forming recommendations. Unless the operator explicitly asks about weather, wind, or risk, do not enumerate the provider, observation time, temperature, humidity, wind speed, gusts, rain, or risk label.
For unsolicited incident guidance, mention at most the general downwind direction when it is operationally relevant.
Never contact or dispatch anyone. You may recommend a verified contact or draft a message for the operator to send.
Do not include raw model confidence unless the operator explicitly asks for technical diagnostics.
Keep replies concise, calm, and action-oriented. For automatic detection messages, state one recommended next step and do not ask a confirmation question. Ask a question only when the operator explicitly requests interactive planning.`;

function cleanMessages(value: unknown): IncomingMessage[] {
  if (!Array.isArray(value)) return [];
  return value.slice(-12).flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const role = "role" in item && item.role === "assistant" ? "assistant" : "user";
    const content = "content" in item && typeof item.content === "string" ? item.content.trim().slice(0, 2500) : "";
    return content ? [{ role, content }] : [];
  });
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "invalid_request" }, { status: 400 });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return Response.json({ error: "model_not_configured" }, { status: 503 });
  }

  const history = cleanMessages(body.messages);
  if (!history.length) return Response.json({ error: "message_required" }, { status: 400 });

  const incident = body.incident && typeof body.incident === "object" ? body.incident : null;
  const weather = body.weather && typeof body.weather === "object" ? body.weather : null;
  const operationalContext = body.cameraId === "hanging-tree-1" ? hangingTreeContext : thunderValleyContext;
  const contextForModel = {
    camera: operationalContext.camera_context,
    site: operationalContext.primary_site_context,
    nearby_landmarks: operationalContext.nearby_operational_landmarks,
    verified_contacts: operationalContext.authorities_and_contacts,
    policy: operationalContext.incident_reasoning_policy,
  };

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      temperature: 0.25,
      max_completion_tokens: 500,
      messages: [
        { role: "system", content: SYSTEM_RULES },
        { role: "system", content: `Approved project-file context:\n${JSON.stringify(contextForModel)}` },
        { role: "system", content: `Synchronized current weather and prototype risk:\n${JSON.stringify(weather)}` },
        { role: "system", content: `Current incident state:\n${JSON.stringify(incident)}` },
        ...history,
      ],
    }),
  });

  if (!response.ok) {
    return Response.json({ error: "model_unavailable" }, { status: 502 });
  }
  const result = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
  const reply = result.choices?.[0]?.message?.content?.trim();
  if (!reply) return Response.json({ error: "empty_model_response" }, { status: 502 });
  return Response.json({ reply, live: true });
}
