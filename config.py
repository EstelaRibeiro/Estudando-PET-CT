"""
config.py
=========
Configurações globais do projeto: caminhos de arquivos, rótulos de segmentação
e constantes físicas.

Edite este arquivo para apontar para o local correto do dataset em sua máquina.
"""

from pathlib import Path

# =============================================================================
# CAMINHOS DO DATASET ENHANCE.PET 1.6k
# =============================================================================

RAIZ = Path(r"d:\usuarios\estela.ribeiro\Meus Documentos\Datasets\ENHANCE-PET_1.6k")

def get_caminhos(patient_id: str) -> dict:
    """
    Retorna um dicionário com os caminhos dos arquivos NIfTI de um paciente.

    O dataset ENHANCE.PET 1.6k organiza os dados da seguinte forma:
        - CT:                images/CT_0001-1000/<ID>.nii.gz
        - PET:               images/PT/<ID>.nii.gz
        - Máscara Cardíaca:  ground-truth/Cardiac/<ID>.nii.gz

    Args:
        patient_id: Identificador do paciente como string (ex: '0001').

    Returns:
        dict com chaves 'CT', 'PET', 'Mascara_Cardiaca' e os respectivos Path.

    Exemplo:
        >>> from config import get_caminhos
        >>> caminhos = get_caminhos('0001')
        >>> print(caminhos['CT'])
        d:/usuarios/estela.ribeiro/.../images/CT_0001-1000/0001.nii.gz
    """
    return {
        "CT": RAIZ / "images" / "CT_0001-1000" / f"{patient_id}.nii.gz",
        "PET": RAIZ / "images" / "PT" / f"{patient_id}.nii.gz",
        "Mascara_Cardiaca": RAIZ / "ground-truth" / "Cardiac" / f"{patient_id}.nii.gz",
    }


# Caminhos de metadados tabulares
XLSX_PT_PATH    = RAIZ / "PT-details.xlsx"
XLSX_CT_PATH    = RAIZ / "CT-details.xlsx"
LABELS_JSON     = RAIZ / "labels.json"


# =============================================================================
# RÓTULOS DA MÁSCARA CARDÍACA (ground-truth/Cardiac/)
# =============================================================================

ROTULOS_CARDIACOS = {
    0:  "background",
    1:  "heart_myocardium",       # ← FOCO PRINCIPAL: fibrose miocárdica
    2:  "heart_atrium_left",
    3:  "heart_atrium_right",
    4:  "heart_ventricle_left",
    5:  "heart_ventricle_right",
    6:  "aorta",
    7:  "iliac_artery_left",
    8:  "iliac_artery_right",
    9:  "iliac_vena_left",
    10: "iliac_vena_right",
    11: "inferior_vena_cava",
    12: "portal_splenic_vein",
    13: "pulmonary_artery",
}

# Subconjunto das estruturas estritamente cardíacas (excluindo vasos periféricos)
ROTULOS_CORACAO = {
    k: v for k, v in ROTULOS_CARDIACOS.items()
    if k in {1, 2, 3, 4, 5, 6, 13}
}

# Rótulo de interesse principal para análise de fibrose
LABEL_MIOCARDIO = 1

# Cores padrão por estrutura para visualização consistente
CORES_CARDIACAS = {
    1:  "#FF4444",   # Miocárdio — vermelho (principal)
    2:  "#4488FF",   # Átrio esquerdo — azul
    3:  "#44AAFF",   # Átrio direito — azul claro
    4:  "#FF8800",   # Ventrículo esquerdo — laranja
    5:  "#FFBB44",   # Ventrículo direito — amarelo-laranja
    6:  "#CC44CC",   # Aorta — roxo
    7:  "#FF44AA",   # Ilíaca esq — rosa
    8:  "#FF44CC",   # Ilíaca dir — rosa escuro
    9:  "#4444FF",   # Veia ilíaca esq — azul
    10: "#4466FF",   # Veia ilíaca dir — azul
    11: "#22AAFF",   # Veia cava inf — azul claro
    12: "#44CCAA",   # Veia porta/esplênica — verde-azul
    13: "#FF6644",   # Artéria pulmonar — laranja-vermelho
}


# =============================================================================
# CONSTANTES FÍSICAS E CLÍNICAS
# =============================================================================

# Janelas de windowing (WL, WW) em HU para diferentes contextos cardíacos
CT_WINDOW_TECIDO_MOLE  = (40,   400)    # Visualização geral de tecido
CT_WINDOW_CARDIACO     = (100,  700)    # Realce de estruturas cardíacas
CT_WINDOW_VASCULAR     = (100,  600)    # Realce de vasos

# Intervalos de clipping CT para pré-processamento cardíaco
CT_CLIP_CARDIACO = (-200, 500)   # HU: exclui ar e osso denso, foca tecido mole

# Limites de SUV para PET cardíaco
PET_SUV_CLIP     = (0.0, 6.0)   # Para FDG cardíaco (captação miocárdica típica 0–4 SUV)
PET_SUV_THRESHOLD_LESAO = 2.5   # Threshold clínico de suspeita de lesão
