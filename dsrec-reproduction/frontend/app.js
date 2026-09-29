const form = document.querySelector("#recommend-form");
const userId = document.querySelector("#user-id");
const topK = document.querySelector("#top-k");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const results = document.querySelector("#recommendations");
const resultsTitle = document.querySelector("#results-title");
const modelVersion = document.querySelector("#model-version");

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function renderRecommendations(data) {
  resultsTitle.textContent = `Recommendations for user ${data.user_id}`;
  modelVersion.textContent = `Model: ${data.model_version}`;

  if (!data.recommendations.length) {
    results.innerHTML = '<div class="empty">No recommendations were returned.</div>';
    return;
  }

  results.innerHTML = data.recommendations
    .map(
      (row, index) => `
        <article class="card">
          <div class="rank">#${index + 1}</div>
          <div class="item">Item ${row.item_id}</div>
          <div class="score">Score ${Number(row.score).toFixed(4)}</div>
        </article>
      `,
    )
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  setStatus("Loading recommendations…");

  try {
    const response = await fetch("/v1/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: Number(userId.value),
        top_k: Number(topK.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    renderRecommendations(data);
    setStatus("Recommendations loaded successfully.");
  } catch (error) {
    setStatus(error.message || "Could not load recommendations.", true);
  } finally {
    submit.disabled = false;
  }
});
