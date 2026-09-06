/**
 * API client for the EduPulse backend.
 *
 * Every network call in the app goes through here — no component calls
 * fetch() directly. This means: one place to change the base URL, one
 * place to handle errors consistently, and no risk of the frontend
 * "hardcoding" a prediction anywhere (brief 21: "Hard-coded prediction in
 * React" is explicitly called out as a common failure mode).
 */
const BASE_URL = "";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    // The backend isn't reachable at all (server not running, wrong port, etc.)
    throw new ApiError(
      "Could not reach the EduPulse backend. Is the API server running on port 8000?",
      0,
      null
    );
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // Response wasn't JSON (rare) — leave body null
  }

  if (!response.ok) {
    const message =
      (body && body.detail && typeof body.detail === "string" && body.detail) ||
      (body && Array.isArray(body.detail) && body.detail.map((e) => e.msg).join("; ")) ||
      `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status, body);
  }

  return body;
}

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export const api = {
  health: () => request("/health"),

  modelInfo: () => request("/model/info"),

  predict: (payload) =>
    request("/predict", { method: "POST", body: JSON.stringify(payload) }),

  explain: (payload) =>
    request("/explain", { method: "POST", body: JSON.stringify(payload) }),

  modelInsights: () => request("/model/insights"),

  listStudents: ({ riskBand, course, limit } = {}) => {
    const params = new URLSearchParams();
    if (riskBand) params.set("risk_band", riskBand);
    if (course) params.set("course", course);
    if (limit) params.set("limit", limit);
    const qs = params.toString();
    return request(`/students${qs ? `?${qs}` : ""}`);
  },

  getStudent: (id) => request(`/students/${id}`),

  analyticsSummary: () => request("/analytics/summary"),
};
