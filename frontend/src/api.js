const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predict(text) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Prediction failed");
  }
  return res.json();
}
