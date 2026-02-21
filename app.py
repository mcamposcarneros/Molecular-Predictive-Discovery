import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
import py3Dmol
from stmol import showmol
import matplotlib.pyplot as plt
import time

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="AI Drug Discovery Platform",
    layout="wide"
)

st.title("AI-Driven Drug Binding Prediction Platform")
st.caption("Virtual Screening | Molecular Analysis | Decision Support")

st.markdown("---")

# =========================================================
# CARGA MODELO
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")

model = load_model()

# =========================================================
# UTILIDADES QUÍMICAS
# =========================================================

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def compute_features(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    descrs = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "QED": Descriptors.qed(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FracCSP3": Descriptors.FractionCSP3(mol)
    }

    return np.concatenate([fp_array, list(descrs.values())]).reshape(1, -1), descrs


def predict_binding(features):
    return model.predict_proba(features)[0][1]


# =========================================================
# DRUG-LIKENESS RULES
# =========================================================

def lipinski(descr):
    violations = 0
    if descr["MolWt"] > 500: violations += 1
    if descr["LogP"] > 5: violations += 1
    if descr["HBA"] > 10: violations += 1
    if descr["HBD"] > 5: violations += 1
    return violations


def veber(descr):
    return descr["TPSA"] <= 140 and descr["RotBonds"] <= 10


# =========================================================
# VISUALIZACIÓN 3D
# =========================================================

def render_3d(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    mblock = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=500, height=500)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'radius': 0.2}})
    view.zoomTo()
    return view


# =========================================================
# RADAR CHART PROPIEDADES
# =========================================================

def radar_plot(descr):
    labels = list(descr.keys())
    values = list(descr.values())

    values_norm = [
        descr["MolWt"] / 800,
        descr["LogP"] / 10,
        descr["HBA"] / 15,
        descr["HBD"] / 10,
        descr["TPSA"] / 200,
        descr["RotBonds"] / 15,
        descr["AromaticRings"] / 10,
        descr["QED"],
        descr["HeavyAtoms"] / 80,
        descr["FracCSP3"]
    ]

    values_norm += values_norm[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values_norm)
    ax.fill(angles, values_norm, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    return fig


# =========================================================
# MODO DE ANÁLISIS
# =========================================================

mode = st.radio(
    "Selecciona modo de análisis",
    ["Molécula individual", "Screening de librería (CSV)"]
)

st.markdown("---")

# =========================================================
# MODO 1 — SINGLE MOLECULE
# =========================================================

DEFAULT_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

if mode == "Molécula individual":

    smiles = st.text_input("Introduce SMILES", value=DEFAULT_SMILES)
    
    if st.button("Analizar molécula"):

        mol = smiles_to_mol(smiles)

        if mol is None:
            st.error("SMILES inválido")
            st.stop()

        with st.spinner("Procesando estructura molecular..."):
            time.sleep(0.5)
            features, descr = compute_features(mol)
            prob = predict_binding(features)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Estructura 2D")
            img = Draw.MolToImage(mol, size=(400,400))
            st.image(img)

        with col2:
            st.subheader("Modelo 3D")
            showmol(render_3d(mol), height=400, width=400)

        st.markdown("---")

        if prob > 0.5:
            st.success(f"BINDER probability: {prob:.2%}")
        else:
            st.error(f"NON-BINDER probability: {prob:.2%}")

        st.subheader("Drug-likeness")

        lip = lipinski(descr)
        veb = veber(descr)

        st.write(f"Lipinski violations: {lip}")
        st.write(f"Veber compliant: {veb}")

        st.subheader("Propiedades fisicoquímicas")
        st.dataframe(pd.DataFrame([descr]))

        st.subheader("Perfil molecular")
        st.pyplot(radar_plot(descr))


# =========================================================
# MODO 2 — BATCH SCREENING (CSV) con vista tipo "single"
# =========================================================
else:

    st.subheader("Screening de librería")
    st.caption("Sube un CSV con columna obligatoria `smiles` (opcional: `name`).")

    file = st.file_uploader("Sube CSV", type="csv")

    # Opciones de screening
    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
    with col_opt1:
        threshold = st.slider("Umbral Binder", 0.0, 1.0, 0.5, 0.01)
    with col_opt2:
        top_k = st.number_input("Top K a mostrar", min_value=5, max_value=5000, value=100, step=5)
    with col_opt3:
        show_3d_batch = st.toggle("Mostrar 3D en detalle", value=True)

    if file:
        df = pd.read_csv(file)

        if "smiles" not in df.columns:
            st.error("El CSV debe tener una columna llamada `smiles`.")
            st.stop()

        # Normalizar columnas opcionales
        if "name" not in df.columns:
            df["name"] = [f"Mol_{i+1}" for i in range(len(df))]

        # Limpiar valores nulos
        df = df.dropna(subset=["smiles"]).copy()
        df["smiles"] = df["smiles"].astype(str).str.strip()

        st.markdown("---")
        st.write(f"Moléculas cargadas: **{len(df):,}**")

        # Ejecutar screening
        results = []
        invalid_idx = []

        progress = st.progress(0)
        status = st.empty()

        for i, (name, smi) in enumerate(zip(df["name"], df["smiles"])):
            status.text(f"Procesando {i+1}/{len(df)} — {name}")

            mol = smiles_to_mol(smi)
            if mol is None:
                results.append({"binding_probability": np.nan})
                invalid_idx.append(i)
            else:
                features, descr = compute_features(mol)
                prob = predict_binding(features)
                results.append({"binding_probability": float(prob)})

            progress.progress((i + 1) / len(df))

        status.empty()
        progress.empty()

        df_res = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        # Clasificación + flags
        df_res["prediction"] = np.where(df_res["binding_probability"] >= threshold, "BINDER", "NON-BINDER")
        df_res["valid_smiles"] = ~df_res["binding_probability"].isna()

        # Ordenar por probabilidad (desc), inválidos al final
        df_res_sorted = df_res.sort_values(
            by=["valid_smiles", "binding_probability"],
            ascending=[False, False]
        ).reset_index(drop=True)

        # Resumen
        n_valid = int(df_res_sorted["valid_smiles"].sum())
        n_invalid = int((~df_res_sorted["valid_smiles"]).sum())
        n_binders = int((df_res_sorted["prediction"] == "BINDER").sum())
        n_non = int((df_res_sorted["prediction"] == "NON-BINDER").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Válidas", f"{n_valid:,}")
        c2.metric("Inválidas", f"{n_invalid:,}")
        c3.metric("BINDERS", f"{n_binders:,}")
        c4.metric("NON-BINDERS", f"{n_non:,}")

        if n_invalid > 0:
            st.warning("Hay SMILES inválidos. Se mantienen en la tabla con probabilidad NaN.")

        st.markdown("---")

        # Tabla TopK
        st.subheader("Ranking (Top K)")
        top_view = df_res_sorted.head(int(top_k)).copy()

        st.dataframe(
            top_view[["name", "smiles", "binding_probability", "prediction", "valid_smiles"]],
            use_container_width=True
        )

        # Descarga resultados completos
        st.download_button(
            "Descargar resultados completos (CSV)",
            df_res_sorted.to_csv(index=False),
            file_name="screening_results.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # =========================================
        # DETALLE TIPO "SINGLE" SOBRE UNA MOLÉCULA
        # =========================================
        st.subheader("Detalle molecular (igual que modo individual)")

        valid_only = df_res_sorted[df_res_sorted["valid_smiles"]].copy()
        if valid_only.empty:
            st.error("No hay ninguna molécula válida para mostrar en detalle.")
            st.stop()

        # Selector: por defecto la mejor (top 1)
        options = (valid_only["name"].astype(str) + " | " +
                   valid_only["binding_probability"].round(4).astype(str) + " | " +
                   valid_only["smiles"].astype(str))

        default_index = 0
        selected = st.selectbox("Elige una molécula del ranking", options.tolist(), index=default_index)

        # Extraer smiles desde el string seleccionado (partimos por ' | ' y cogemos el último)
        selected_smiles = selected.split(" | ")[-1].strip()
        mol = smiles_to_mol(selected_smiles)

        if mol is None:
            st.error("La molécula seleccionada no se pudo reconstruir (SMILES inválido).")
            st.stop()

        # Recalcular todo para que el detalle sea 100% consistente
        with st.spinner("Generando análisis detallado..."):
            features, descr = compute_features(mol)
            prob = predict_binding(features)
            pred = "BINDER" if prob >= threshold else "NON-BINDER"

        # Vista 2D + 3D
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Estructura 2D")
            img = Draw.MolToImage(mol, size=(520, 520))
            st.image(img, use_container_width=True)

        with colB:
            st.markdown("#### Estructura 3D")
            if show_3d_batch:
                showmol(render_3d(mol), height=520, width=520)
            else:
                st.info("Activa 'Mostrar 3D en detalle' para ver la visualización 3D.")

        st.markdown("---")

        # Resultado principal
        if pred == "BINDER":
            st.success(f"### PREDICCIÓN: BINDER\n**Probabilidad de binding:** {prob:.2%} (umbral {threshold:.2f})")
        else:
            st.error(f"### PREDICCIÓN: NON-BINDER\n**Probabilidad de binding:** {prob:.2%} (umbral {threshold:.2f})")

        # Drug-likeness
        st.markdown("### Drug-likeness")
        lip = lipinski(descr)
        veb = veber(descr)

        cL1, cL2, cL3 = st.columns(3)
        cL1.metric("Lipinski violations", str(lip))
        cL2.metric("Veber compliant", "✅" if veb else "❌")
        cL3.metric("QED", f"{descr['QED']:.3f}")

        # Propiedades
        st.markdown("### Propiedades fisicoquímicas")
        df_descr = pd.DataFrame([descr])
        st.dataframe(df_descr, use_container_width=True)

        # Radar
        st.markdown("### Perfil molecular")
        st.pyplot(radar_plot(descr))

        # Descarga detalle (1 molécula)
        detail_df = valid_only.copy()
        detail_row = detail_df[detail_df["smiles"] == selected_smiles].head(1)
        if not detail_row.empty:
            out = detail_row.copy()
            for k, v in descr.items():
                out[k] = v
            out["threshold"] = threshold
            out["detail_prediction"] = pred
            st.download_button(
                "Descargar detalle de esta molécula (CSV)",
                out.to_csv(index=False),
                file_name="selected_molecule_detail.csv",
                mime="text/csv"

            )
