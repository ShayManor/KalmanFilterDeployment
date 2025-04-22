import copy
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from Kalman import kalman

app = Flask(__name__, static_folder="static")
CORS(app)

current_step = 0
last_kalman_gain = None
kalman_results = {}  # Stores all computed matrices and initial inputs for diagram

grid_scale = 225
shift_x = 125
shift_y = 100


def grid(x, y):
    return shift_x + x * grid_scale, shift_y + y * grid_scale


PINK = "#f4c2c2"  # Chance node
BLUE = "#00afee"  # Deterministic node

BASE_DIAGRAM = {
    "nodes": [
        {"id": "V(0)", "x": grid(0, 2)[0], "y": grid(0, 2)[1],
         "color": PINK, "orig_color": PINK, "clickable": True,
         "distribution": {"mean": "0", "cov": "$R_1$"},
         "bottomLabel": "$\\bigl(0,\\;R_1\\bigr)$", "display": "$v_1$"},
        {"id": "Z(0)", "x": grid(1, 2)[0], "y": grid(1, 2)[1],
         "color": BLUE, "orig_color": BLUE, "clickable": True,
         "distribution": {"mean": "$h_1(\\hat{x}_{1|0})$", "cov": "0"},
         "bottomLabel": "$\\bigl(h_1(\\hat{x}_{1|0}),\\;0\\bigr)$", "display": "$z_1$"},
        {"id": "X(0)", "x": grid(1, 1)[0], "y": grid(1, 1)[1],
         "color": PINK, "orig_color": PINK, "clickable": True,
         "distribution": {"mean": "$\\hat{x}_{1|0}$", "cov": "$P_{1|0}$"},
         "bottomLabel": "$\\bigl(\\hat{x}_{1|0},\\;P_{1|0}\\bigr)$", "display": "$\\hat{x}_{1|0}$"},
        {"id": "X(1)", "x": grid(3, 1)[0], "y": grid(3, 1)[1],
         "color": BLUE, "orig_color": BLUE, "clickable": True,
         "distribution": {"mean": "$\\hat{x}_{2|1}$", "cov": "$P_{2|1}$"},
         "bottomLabel": "$\\bigl(\\hat{x}_{2|1},\\;P_{2|1}\\bigr)$", "display": "$\\hat{x}_{2|1}$"},
        {"id": "W(0)", "x": grid(3, 2)[0], "y": grid(3, 0)[1],
         "color": PINK, "orig_color": PINK, "clickable": True,
         "distribution": {"mean": "0", "cov": "$Q_2$"},
         "bottomLabel": "$\\bigl(0,\\;Q_2\\bigr)$", "display": "$w_2$"}
    ],
    "edges": [
        {"source": "V(0)", "target": "Z(0)"},
        {"source": "W(0)", "target": "X(1)", "operator": "gamma(2)"},
        {"source": "X(0)", "target": "X(1)", "operator": "phi(1)"},
        {"source": "X(0)", "target": "Z(0)", "operator": "H(1)"}
    ]
}


def step(removed_nodes):
    new_nodes = [n for n in BASE_DIAGRAM["nodes"] if n["id"] not in removed_nodes]
    new_edges = [e for e in BASE_DIAGRAM["edges"]
                 if e["source"] not in removed_nodes and e["target"] not in removed_nodes]
    return {"nodes": new_nodes, "edges": new_edges}


def reverse_edge(step_obj, source, target):
    new_step = copy.deepcopy(step_obj)
    for i, edge in enumerate(new_step["edges"]):
        if edge["source"] == source and edge["target"] == target:
            op = edge.get("operator")
            new_step["edges"][i] = {"source": target, "target": source}
            if op:
                new_step["edges"][i]["operator"] = "K1"
    return new_step


STEP_1 = step([])  # Original diagram
STEP_2 = step(["V(0)"])  # Remove measurement noise
STEP_3 = reverse_edge(STEP_2, "X(0)", "Z(0)")  # Reverse X(0)->Z(0) to Z(0)->X(0) with K1
STEP_4 = step(["V(0)", "Z(0)"])  # Remove Z(0), showing just X(0) updated
STEP_5 = step(["V(0)", "Z(0)", "X(0)"])  # Show just X(1) with phi_P1_phiT
STEP_6 = step(["V(0)", "Z(0)", "X(0)", "W(0)"])  # Final step with just X(1) + process noise

STEPS = [STEP_1, STEP_2, STEP_3, STEP_4, STEP_5, STEP_6]

DISPLAY_LABELS = {
    "X(0)": "$x_1$",
    "V(0)": "$v_1$",
    "Z(0)": "$z_1$",
    "X(1)": "$x_2$",
    "W(0)": "$w_2$"
}


def matrix_to_latex(mat):
    """Convert a 2D list or numpy array into a LaTeX bmatrix string."""
    if isinstance(mat, np.ndarray):
        mat = mat.tolist()
    rows = []
    for row in mat:
        if isinstance(row, list):
            formatted = " & ".join(f"{el:.4f}" if isinstance(el, (int, float)) else str(el) for el in row)
        else:
            # Handle 1D arrays or single values
            formatted = f"{row:.4f}" if isinstance(row, (int, float)) else str(row)
        rows.append(formatted)
    body = " \\\\ ".join(rows)
    return f"\\begin{{pmatrix}} {body} \\end{{pmatrix}}"


@app.route("/graph_step", methods=["GET"])
def get_graph_step():
    global current_step, last_kalman_gain, kalman_results
    step_data = copy.deepcopy(STEPS[current_step])

    # Update node displays based on step
    for node in step_data["nodes"]:
        node_id = node["id"]
        if node_id == "X(0)":
            if current_step <= 2:
                node["display"] = "$\\hat{x}_{1|0}$"
            else:
                node["display"] = "$\\hat{x}_{1|1}$"
        elif node_id == "X(1)":
            if current_step < 4:
                node["color"] = BLUE
            else:
                node["color"] = PINK
            if current_step == 3:
                node["display"] = "$\\hat{x}_{2|1}$"
            if current_step > 4:
                node["color"] = PINK
                node["display"] = "$\\hat{x}_{2|1}$"

    # Update each node's distribution using computed (or input) matrices.
    if kalman_results:
        for node in step_data["nodes"]:
            nid = node["id"]
            if nid == "V(0)":
                R_mat = kalman_results.get("R")
                if R_mat is not None:
                    cov_latex = matrix_to_latex(R_mat)
                    node["distribution"] = {"mean": "0", "cov": cov_latex}
                    node["original_distribution"] = {"mean": "0", "cov": "$R_1$"}
                    node["bottomLabel"] = "$\\bigl(0,\\;R_1\\bigr)$"

            elif nid == "Z(0)":
                h_mat = kalman_results.get("h")
                if h_mat is not None:
                    mean_latex = matrix_to_latex(h_mat)
                    node["bottomLabel"] = "$\\bigl(h_1(\\hat{x}_{1|0}),\\;0\\bigr)$"

                    if current_step == 0:
                        node["distribution"] = {"mean": mean_latex, "cov": "0"}
                        node["bottomLabel"] = "$\\bigl(h_1(\\hat{x}_{1|0}),\\;0\\bigr)$"
                    elif current_step == 1:
                        node["color"] = PINK
                        S_mat = kalman_results.get("R")
                        cov_latex = matrix_to_latex(S_mat) if S_mat is not None else "$S_1$"
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["bottomLabel"] = "$\\bigl(h_1(\\hat{x}_{1|0}),\\;R_1\\bigr)$"
                    else:
                        node["color"] = PINK
                        S_mat = kalman_results.get("S")
                        cov_latex = matrix_to_latex(S_mat) if S_mat is not None else "$S_1$"
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["bottomLabel"] = "$\\bigl(h_1(\\hat{x}_{1|0}),\\;S_1\\bigr)$"



            elif nid == "X(0)":
                if current_step < 2:
                    u0 = kalman_results.get("u0")
                    X_mat = kalman_results.get("X")
                    if u0 is not None and X_mat is not None:
                        mean_latex = matrix_to_latex(u0)
                        cov_latex = matrix_to_latex(X_mat)
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["original_distribution"] = {"mean": "$\\hat{x}_{1|0}$", "cov": "$P_{1|0}$"}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{1|0},\\;P_{1|0}\\bigr)$"
                elif current_step == 2:
                    u1 = kalman_results.get("u0")
                    P1_mat = kalman_results.get("P1")
                    if u1 is not None and P1_mat is not None:
                        mean_latex = matrix_to_latex(u1)
                        cov_latex = matrix_to_latex(P1_mat)
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                    node["bottomLabel"] = "$\\bigl(\\hat{x}_{1|0},\\;P_{1|1}\\bigr)$"
                else:
                    u1 = kalman_results.get("u1")
                    P1_mat = kalman_results.get("P1")
                    if u1 is not None and P1_mat is not None:
                        mean_latex = matrix_to_latex(u1)
                        cov_latex = matrix_to_latex(P1_mat)
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{1|1},\\;P_{1|1}\\bigr)$"

            elif nid == "X(1)":
                u_next = kalman_results.get("u_next")

                if current_step < 3:
                    if u_next is not None:
                        mean_latex = matrix_to_latex(u_next)
                        node["distribution"] = {"mean": mean_latex, "cov": "0"}
                        node["original_distribution"] = {"mean": "$\\hat{x}_{2|0}$", "cov": "0"}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{2|0},\\;0\\bigr)$"
                elif current_step == 3:
                    if u_next is not None:
                        mean_latex = matrix_to_latex(u_next)
                        node["distribution"] = {"mean": mean_latex, "cov": "0"}
                        node["original_distribution"] = {"mean": "$\\hat{x}_{2|1}$", "cov": "0"}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{2|1},\\;0\\bigr)$"
                elif current_step == 4:
                    phiP1phiT = kalman_results.get("phi_P1_phiT")
                    if u_next is not None and phiP1phiT is not None:
                        mean_latex = matrix_to_latex(u_next)
                        cov_latex = matrix_to_latex(phiP1phiT)
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["original_distribution"] = {"mean": "$\\hat{x}_{2|1}$", "cov": "$\\Phi_1P_{1|1}\\Phi^T_1$"}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{2|1},\\;\\Phi_1P_{1|1}\\Phi^T_1\\bigr)$"

                else:  # step 5 and beyond
                    B_next = kalman_results.get("B_next")
                    if u_next is not None and B_next is not None:
                        mean_latex = matrix_to_latex(u_next)
                        cov_latex = matrix_to_latex(B_next)
                        node["distribution"] = {"mean": mean_latex, "cov": cov_latex}
                        node["original_distribution"] = {"mean": "$\\hat{x}_{2|1}$", "cov": "$P_{2|1}$"}
                        node["bottomLabel"] = "$\\bigl(\\hat{x}_{2|1},\\;P_{2|1}\\bigr)$"

            elif nid == "W(0)":
                Qk = kalman_results.get("Qk")
                if Qk is not None:
                    cov_latex = matrix_to_latex(Qk)
                    node["distribution"] = {"mean": "0", "cov": cov_latex}
                    node["original_distribution"] = {"mean": "0", "cov": "$Q_2$"}
                    node["bottomLabel"] = "$\\bigl(0,\\;Q_2\\bigr)$"

            # Always set the display label regardless of computations
            if nid in DISPLAY_LABELS:
                node["display"] = DISPLAY_LABELS[nid]

    for edge in step_data["edges"]:
        op = edge.get("operator", "")
        if "H(1)" in op:
            H_mat = kalman_results.get("H")
            if H_mat is not None:
                latex_str = matrix_to_latex(H_mat)
                edge["operator_matrix"] = H_mat.tolist()
        elif "phi(1)" in op:
            Phi_mat = kalman_results.get("Phi")
            if Phi_mat is not None:
                latex_str = matrix_to_latex(Phi_mat)
                edge["operator_matrix"] = Phi_mat.tolist()
        elif "gamma(2)" in op:
            Gamma_mat = kalman_results.get("Gamma")
            if Gamma_mat is not None:
                latex_str = matrix_to_latex(Gamma_mat)
                edge["operator_matrix"] = Gamma_mat.tolist()

    if current_step == 2:
        for edge in step_data["edges"]:
            if edge.get("source") == "Z(0)" and edge.get("target") == "X(0)":
                edge["operator"] = "K1"
                if last_kalman_gain is not None:
                    edge["operator_matrix"] = last_kalman_gain

    # Set clickable property for nodes
    for node in step_data["nodes"]:
        node["clickable"] = True

    step_data["kalman_gain"] = last_kalman_gain
    step_data["current_step"] = current_step
    step_data["step_display"] = f"Step {current_step + 1}"
    return jsonify(step_data)


@app.route("/next_step", methods=["POST"])
def next_step():
    global current_step
    current_step = (current_step + 1) % len(STEPS)
    return jsonify({"current_step": current_step})


@app.route("/prev_step", methods=["POST"])
def prev_step():
    global current_step
    current_step = (current_step - 1) % len(STEPS)
    return jsonify({"current_step": current_step})


@app.route("/run_kalman", methods=["POST"])
def run_kalman_filter():
    global current_step, last_kalman_gain, kalman_results
    data = request.get_json()
    matrices = data["matrices"]
    print(matrices)
    Z = np.array(matrices.get('Z'))
    u = np.array(matrices.get('u'))
    X = np.array(
        [
            [1125, 750, 250, 0, 0, 0],
            [750, 1000, 500, 0, 0, 0],
            [250, 500, 500, 0, 0, 0],
            [0, 0, 0, 1125, 750, 250],
            [0, 0, 0, 750, 1000, 500],
            [0, 0, 0, 250, 500, 500],
        ]
    )
    V = np.zeros((6, 1))
    R = np.array([[25, 0], [0, 0.0087 ** 2]])
    H = np.array([[0.8, 0, 0, -0.6, 0, 0], [0.0012, 0, 0, 0.0016, 0, 0]])
    Phi = np.array([
        [1, 1, 0.5, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0.5],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
    ])
    gamma = np.eye(6)
    Qk = np.array([
        [0.25, 0.5, 0.5, 0, 0, 0],
        [0.5, 1, 1, 0, 0, 0],
        [0.5, 1, 1, 0, 0, 0],
        [0, 0, 0, 0.25, 0.5, 0.5],
        [0, 0, 0, 0.5, 1, 1],
        [0, 0, 0, 0.5, 1, 1],
    ]) * (0.2 ** 2)
    h = np.array([[500], [-0.644]])

    u_updated, B_updated, V_updated, K, S, P1, u1 = kalman(0, Z, u, X, V, R, H, Phi, gamma, Qk, 1, h)

    u_next = u_updated  # time update (predicted next state)
    B_next = B_updated  # predicted covariance matrix
    x20 = np.matmul(Phi, u)
    phi_P1_phiT = np.matmul(Phi, np.matmul(P1, Phi.T))

    kalman_results = {
        "u0": u,
        "X": X,
        "X20": x20,
        "R": R,
        "h": h,
        "Gamma": gamma,
        "u1": u1,
        "P1": P1,
        "S": S,
        "K": K,
        "u_next": u_next,
        "B_next": B_next,
        "phi_P1_phiT": phi_P1_phiT,
        "Qk": Qk,
        "H": H,
        "Phi": Phi
    }
    last_kalman_gain = K.tolist() if K is not None else None
    current_step = 0
    return jsonify({"kalman_gain": last_kalman_gain, "current_step": current_step})


@app.route("/")
def root_index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
