// D3 code to draw the diagram

// Convert operator labels to LaTeX (e.g., "gamma(1)" -> "$\\Gamma_{1}$")
function operatorToGreek(op) {
  if (!op) return "";
  if (op.startsWith("gamma")) {
    let m = op.match(/\((\d+)\)/);
    let idx = m ? m[1] : "";
    return "$\\Gamma_{" + idx + "}$";
  }
  if (op.startsWith("phi") || op.startsWith("Phi")) {
    let m = op.match(/\((\d+)\)/);
    let idx = m ? m[1] : "";
    return "$\\Phi_{" + idx + "}$";
  }
  if (op.startsWith("H")) {
    let m = op.match(/\((\d+)\)/);
    let idx = m ? m[1] : "";
    return "$H_{" + idx + "}$";
  }
  return "$" + op + "$";
}

const svg = d3.select("#graph");
const linkGroup = svg.append("g").attr("class", "links");
const nodeGroup = svg.append("g").attr("class", "nodes");
const labelGroup = svg.append("g").attr("class", "labels");

svg.append("defs").append("marker")
  .attr("id", "arrow")
  .attr("viewBox", "0 -5 10 10")
  .attr("refX", 28)
  .attr("refY", 0)
  .attr("markerWidth", 6)
  .attr("markerHeight", 6)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-5L10,0L0,5")
  .attr("fill", "#999");

function openModal(htmlContent) {
  d3.select("#modalContent").html(htmlContent);
  d3.select("#modalOverlay").style("display", "block");
  MathJax.typesetPromise([document.getElementById("modalContent")]);
}
function closeModal() {
  d3.select("#modalOverlay").style("display", "none");
}
document.getElementById("closeBtn").onclick = closeModal;

let kalmanGainHOT = null;
const kgPanel = document.getElementById("kalmanGainPanel");
const kgDiv = document.getElementById("kalmanGainHot");

function fetchGraphStep() {
  fetch("http://localhost:5000/graph_step")
    .then(res => res.json())
    .then(data => {
      updateGraph(data);
      updateKalmanGain(data);
    })
    .catch(e => console.error("Error fetching /graph_step:", e));
}

function updateGraph(data) {
  document.getElementById("currentStepDisplay").textContent = data.step_display || "";
  const nodes = data.nodes || [];
  const edges = data.edges || [];

  // Draw edges
  const linkSel = linkGroup.selectAll(".link").data(edges, d => d.source + "-" + d.target);
  linkSel.exit().remove();
  const linkEnter = linkSel.enter().append("line").attr("class", "link");
  const allLinks = linkEnter.merge(linkSel);
  allLinks
    .attr("x1", d => (nodes.find(n => n.id === d.source) || {x: 0}).x)
    .attr("y1", d => (nodes.find(n => n.id === d.source) || {y: 0}).y)
    .attr("x2", d => (nodes.find(n => n.id === d.target) || {x: 0}).x)
    .attr("y2", d => (nodes.find(n => n.id === d.target) || {y: 0}).y);

  // Draw nodes
  const nodeSel = nodeGroup.selectAll(".node").data(nodes, d => d.id);
  nodeSel.exit().remove();
  const nodeEnter = nodeSel.enter().append("g").attr("class", "node");

  nodeEnter.append("circle")
    .attr("r", 20)
    .on("click", (evt, d) => {
      if (!d.clickable) return;
      const dist = d.distribution || {};
      let mean = dist.mean || "N/A";
      let cov = dist.cov || "N/A";
      openModal(`
        <div class="matrix-container" style="margin:0; box-shadow:none;">
          <h3>${d.display || d.id}</h3>
          <h4>Mean:</h4><div>${mean}</div>
          <h4>Covariance:</h4><div>${cov}</div>
        </div>
      `);
    });

  // Node labels (using foreignObject so MathJax can render)
  nodeEnter.append("foreignObject")
    .attr("class", "nodetext")
    .attr("x", -30)
    .attr("y", -10)
    .attr("width", 60)
    .attr("height", 40)
    .html(d => `
      <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:14px; text-align:center;">
        ${d.display || d.id}
      </div>
    `);

  // Bottom-right labels (for distribution info)
  nodeEnter.append("foreignObject")
    .attr("class", "bottomLabel")
    .attr("x", 15)
    .attr("y", 15)
    .attr("width", 100)
    .attr("height", 30)
    .html(d => `
      <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:12px; color:#444;">
        ${d.bottomLabel || ""}
      </div>
    `);

  const allNodes = nodeEnter.merge(nodeSel);
  allNodes.attr("transform", d => `translate(${d.x},${d.y})`)
    .select("circle")
    .attr("fill", d => d.color || "#ccc");

  // Draw edge operator labels
  const opSel = labelGroup.selectAll(".edgeLabel")
    .data(edges.filter(e => e.operator), d => d.source + "-" + d.target);
  opSel.exit().remove();
  const opEnter = opSel.enter().append("foreignObject")
    .attr("class", "edgeLabel")
    .attr("width", 60)
    .attr("height", 30)
    .on("click", (evt, d) => {
      openModal(`
        <div class="matrix-container" style="margin:0; box-shadow:none;">
          <h3>Operator: ${operatorToGreek(d.operator) || "N/A"}</h3>
          <p>No matrix info available.</p>
        </div>
      `);
    });
  const allOpLabels = opEnter.merge(opSel);
  allOpLabels
    .attr("x", d => {
      const s = nodes.find(n => n.id === d.source);
      const t = nodes.find(n => n.id === d.target);
      if (!s || !t) return 0;
      return s.x + 0.5 * (t.x - s.x) - 20;
    })
    .attr("y", d => {
      const s = nodes.find(n => n.id === d.source);
      const t = nodes.find(n => n.id === d.target);
      if (!s || !t) return 0;
      return s.y + 0.5 * (t.y - s.y) - 10;
    })
    .html(d => `
      <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:14px; text-align:center;">
        ${operatorToGreek(d.operator)}
      </div>
    `);

  MathJax.typesetPromise();
}

function updateKalmanGain(data) {
  if (data.current_step < 2) {
    kgPanel.classList.add("hidden");
    return;
  }
  if (data.kalman_gain) {
    kgPanel.classList.remove("hidden");
    if (!kalmanGainHOT) {
      kalmanGainHOT = new Handsontable(kgDiv, {
        data: data.kalman_gain,
        rowHeaders: true,
        colHeaders: true,
        readOnly: true,
        licenseKey: "non-commercial-and-evaluation"
      });
    } else {
      kalmanGainHOT.updateSettings({ data: data.kalman_gain });
    }
  } else {
    kgPanel.classList.add("hidden");
  }
}

// Step Navigation
document.getElementById("nextStepBtn").onclick = () => {
  fetch("http://localhost:5000/next_step", { method: "POST" })
    .then(res => res.json())
    .then(() => fetchGraphStep())
    .catch(e => console.error("Error next step:", e));
};
document.getElementById("prevStepBtn").onclick = () => {
  fetch("http://localhost:5000/prev_step", { method: "POST" })
    .then(res => res.json())
    .then(() => fetchGraphStep())
    .catch(e => console.error("Error prev step:", e));
};
document.getElementById("backToInputBtn").onclick = () => {
  document.getElementById("diagramPanel").classList.add("hidden");
  document.getElementById("matrixInputPanel").classList.remove("hidden");
  kgPanel.classList.add("hidden");
};
