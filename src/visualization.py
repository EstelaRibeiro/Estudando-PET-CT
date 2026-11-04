"""
visualization.py
================
Visualização Avançada de Volumes PET/CT com foco no Coração.

Contexto:
    - Em Raio-X, você exibe uma única imagem 2D.
    - Em PET/CT, você escolhe um PLANO de corte (axial, sagital, coronal) e
      um índice de fatia para visualizar a fatia 2D do volume 3D.
    - A fusão PET/CT sobrepõe o mapa de calor do PET (atividade metabólica)
      sobre a anatomia do CT, usando transparência (alpha blending).
    - Funções cardíacas permitem recortar (crop) o volume ao redor do coração
      e destacar estruturas específicas (miocárdio, ventrículos, etc.).

Planos anatômicos:
    - Axial    (transversal): divide superior/inferior  → fatias ao longo do eixo Z
    - Coronal  (frontal):     divide anterior/posterior → fatias ao longo do eixo Y
    - Sagital  (lateral):     divide esquerda/direita   → fatias ao longo do eixo X

Nomenclatura de estruturas cardíacas:
    Ver ROTULOS_CARDIACOS e CORES_CARDIACAS em config.py.
"""

from typing import Optional, Tuple, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import SimpleITK as sitk
from pathlib import Path

from . import io_utils


# =============================================================================
# SEÇÃO 1: UTILITÁRIOS INTERNOS DE NORMALIZAÇÃO
# =============================================================================

def _normalize_window(
    array: np.ndarray,
    wl: Optional[float] = None,
    ww: Optional[float] = None,
) -> np.ndarray:
    """
    Aplica windowing (nível e largura de janela) ao array e normaliza para [0, 1].

    Windowing é uma técnica de CT para realçar diferentes tecidos:
        - Janela pulmonar:   WL=-600, WW=1500
        - Janela óssea:      WL=400,  WW=1800
        - Janela de tecido:  WL=40,   WW=400

    Args:
        array: Array 2D ou 3D de valores de intensidade.
        wl:    Window Level (centro da janela). Se None, usa a mediana.
        ww:    Window Width (largura da janela). Se None, usa [min, max].

    Returns:
        Array float32 normalizado para [0, 1].
    """
    arr = array.astype(np.float32)

    if wl is None or ww is None:
        vmin, vmax = arr.min(), arr.max()
    else:
        vmin = wl - ww / 2.0
        vmax = wl + ww / 2.0

    arr = np.clip(arr, vmin, vmax)

    # Evita divisão por zero se min == max
    denom = vmax - vmin
    if denom == 0:
        return np.zeros_like(arr)

    return (arr - vmin) / denom


def _get_slice(
    volume: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """
    Extrai uma fatia 2D de um volume 3D no plano anatômico especificado.

    Convenção de eixos (após sitk_to_numpy):
        volume.shape = (z, y, x)
        - Axial    → volume[z, :, :]
        - Coronal  → volume[:, y, :]
        - Sagital  → volume[:, :, x]

    Args:
        volume: Array 3D de shape (z, y, x).
        plane:  'axial', 'coronal' ou 'sagital'.
        index:  Índice da fatia no plano escolhido.

    Returns:
        Array 2D correspondente à fatia.

    Raises:
        ValueError: Se o plano for inválido ou índice fora do intervalo.
    """
    plane = plane.lower()

    # Limita o índice ao intervalo válido
    if plane == "axial":
        index = int(np.clip(index, 0, volume.shape[0] - 1))
        return volume[index, :, :]
    elif plane == "coronal":
        index = int(np.clip(index, 0, volume.shape[1] - 1))
        return volume[:, index, :]
    elif plane == "sagital":
        index = int(np.clip(index, 0, volume.shape[2] - 1))
        return volume[:, :, index]
    else:
        raise ValueError(f"Plano inválido: '{plane}'. Use 'axial', 'coronal' ou 'sagital'.")


# =============================================================================
# SEÇÃO 2: VISUALIZAÇÃO DE UM ÚNICO VOLUME (CT ou PET)
# =============================================================================

def plot_single_volume(
    volume: np.ndarray,
    plane: str = "axial",
    index: Optional[int] = None,
    title: str = "Volume",
    cmap: str = "gray",
    wl: Optional[float] = None,
    ww: Optional[float] = None,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Visualiza uma única fatia 2D de um volume 3D.

    Args:
        volume:    Array 3D (z, y, x).
        plane:     Plano anatômico: 'axial', 'coronal', 'sagital'.
        index:     Índice da fatia. Se None, usa o centro do volume.
        title:     Título do gráfico.
        cmap:      Colormap matplotlib ('gray' para CT, 'hot' para PET).
        wl:        Window Level para CT (ex: 40 para tecido mole).
        ww:        Window Width para CT (ex: 400 para tecido mole).
        figsize:   Tamanho da figura em polegadas.
        save_path: Caminho para salvar a imagem. None para não salvar.

    Returns:
        Objeto matplotlib Figure.

    Exemplo:
        >>> ct_arr = io_utils.sitk_to_numpy(ct_image)
        >>> fig = plot_single_volume(ct_arr, plane='axial', wl=40, ww=400, title='CT Axial')
        >>> plt.show()
    """
    # Índice padrão = centro do volume no plano
    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    if index is None:
        index = volume.shape[dim_map[plane]] // 2

    slice_2d = _get_slice(volume, plane, index)
    norm_slice = _normalize_window(slice_2d, wl=wl, ww=ww)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(norm_slice, cmap=cmap, origin="lower", aspect="equal")
    ax.set_title(f"{title}\n{plane.capitalize()} — Fatia {index}", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_three_planes(
    volume: np.ndarray,
    index_axial: Optional[int] = None,
    index_coronal: Optional[int] = None,
    index_sagital: Optional[int] = None,
    title: str = "Volume — Três Planos",
    cmap: str = "gray",
    wl: Optional[float] = None,
    ww: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Visualiza os três planos anatômicos (axial, coronal, sagital) lado a lado.

    Args:
        volume:          Array 3D (z, y, x).
        index_axial:     Índice do corte axial.
        index_coronal:   Índice do corte coronal.
        index_sagital:   Índice do corte sagital.
        title:           Título principal da figura.
        cmap:            Colormap matplotlib.
        wl, ww:          Windowing para CT.
        figsize:         Tamanho da figura.
        save_path:       Caminho de saída da imagem.

    Returns:
        Objeto matplotlib Figure.
    """
    # Índices padrão = centros de cada dimensão
    iz = index_axial   or volume.shape[0] // 2
    iy = index_coronal or volume.shape[1] // 2
    ix = index_sagital or volume.shape[2] // 2

    planes = [
        ("axial",   iz, f"Axial (z={iz})"),
        ("coronal", iy, f"Coronal (y={iy})"),
        ("sagital", ix, f"Sagital (x={ix})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (plane, idx, label) in zip(axes, planes):
        slice_2d = _get_slice(volume, plane, idx)
        norm_slice = _normalize_window(slice_2d, wl=wl, ww=ww)
        im = ax.imshow(norm_slice, cmap=cmap, origin="lower", aspect="equal")
        ax.set_title(label, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# SEÇÃO 3: FUSÃO / OVERLAY PET sobre CT
# =============================================================================

def plot_pet_ct_fusion(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    plane: str = "axial",
    index: Optional[int] = None,
    ct_wl: float = 40.0,
    ct_ww: float = 400.0,
    pet_cmap: str = "hot",
    pet_alpha: float = 0.5,
    pet_vmin: Optional[float] = None,
    pet_vmax: Optional[float] = None,
    title: str = "Fusão PET/CT",
    figsize: Tuple[int, int] = (7, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Exibe a fusão PET/CT: anatomia CT em escala de cinza com PET superposto
    em colormap de calor (hot, jet, etc.) com transparência ajustável.

    Como funciona a fusão:
        1. CT é mostrado em fundo cinza (anatomia).
        2. PET é mostrado com colormap 'hot' (preto → amarelo → vermelho → branco),
           com alpha (transparência) para deixar a anatomia visível.
        3. O threshold pet_vmin permite suprimir o ruído de fundo do PET.

    Args:
        ct_volume:   Array 3D CT (z, y, x), preferencialmente em HU.
        pet_volume:  Array 3D PET (z, y, x), preferencialmente em SUVbw ou Bq/mL.
        plane:       Plano de corte: 'axial', 'coronal', 'sagital'.
        index:       Índice da fatia. None = centro.
        ct_wl:       Window Level do CT (padrão: tecido mole, 40 HU).
        ct_ww:       Window Width do CT (padrão: tecido mole, 400 HU).
        pet_cmap:    Colormap do PET ('hot', 'jet', 'inferno', 'turbo').
        pet_alpha:   Transparência do PET (0=invisível, 1=opaco).
        pet_vmin:    Valor mínimo do PET para colormap (threshold de background).
        pet_vmax:    Valor máximo do PET para colormap.
        title:       Título da figura.
        figsize:     Tamanho da figura.
        save_path:   Caminho de saída.

    Returns:
        Objeto matplotlib Figure.

    Exemplo:
        >>> fig = plot_pet_ct_fusion(ct_arr, pet_arr, plane='axial',
        ...                          ct_wl=40, ct_ww=400, pet_alpha=0.4)
        >>> plt.show()
    """
def _get_robust_pet_vmax(pet_arr: np.ndarray, user_vmax: Optional[float] = None) -> float:
    """
    Retorna o valor máximo apropriado para normalização visual do PET.
    Evita saturação se o array estiver em Bq/mL (ex: valores até 150.000).
    Se o array estiver em Bq/mL (> 50), usa o percentil 99 dos voxels positivos.
    Se o usuário informou um vmax condizente com a magnitude do array, respeita o valor.
    """
    validos = pet_arr[pet_arr > 0]
    if validos.size == 0:
        return 1.0
    p99 = float(np.percentile(validos, 99.0))
    if user_vmax is not None:
        # Se os dados estiverem na ordem de milhares e o usuário passou vmax pequeno (ex: 5.0),
        # usamos p99 para evitar saturação total
        if pet_arr.max() > 50 and user_vmax < 50:
            return p99
        return float(user_vmax)
    return p99


def plot_pet_ct_fusion(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    plane: str = "axial",
    index: Optional[int] = None,
    mask_slice: Optional[np.ndarray] = None,
    contour_label: int = 1,
    contour_color: str = "lime",
    ct_wl: float = 40.0,
    ct_ww: float = 400.0,
    pet_cmap: str = "hot",
    pet_alpha: float = 0.5,
    pet_vmin: float = 0.0,
    pet_vmax: Optional[float] = None,
    title: str = "Fusão PET/CT",
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Exibe a fusão PET/CT com técnica de máscara transparente:
        - CT anatômico em escala de cinza com windowing.
        - PET exibido com transparência real (voxels <= 0 são mascarados/invisíveis).
        - Escala calibrada automaticamente pelo percentil 99 para evitar saturação.
        - Suporte opcional a contorno nítido de estrutura anatômica (ex: miocárdio).
    """
    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    if index is None:
        index = ct_volume.shape[dim_map[plane]] // 2

    ct_slice  = _get_slice(ct_volume,  plane, index)
    pet_slice = _get_slice(pet_volume, plane, index)

    # Normaliza CT para [0,1]
    ct_norm = _normalize_window(ct_slice, wl=ct_wl, ww=ct_ww)

    # Escala robusta para o PET
    pmax = _get_robust_pet_vmax(pet_volume, pet_vmax)
    pmin = pet_vmin

    # Fundo transparente onde o PET não apresenta captação positiva
    pet_overlay = np.ma.masked_where(pet_slice <= 0, pet_slice)

    fig, ax = plt.subplots(figsize=figsize)

    # Camada 1: CT (fundo cinza)
    ax.imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")

    # Camada 2: PET em mapa de calor apenas onde há captação
    im = ax.imshow(
        pet_overlay,
        cmap=pet_cmap,
        alpha=pet_alpha,
        vmin=pmin,
        vmax=pmax,
        origin="lower",
        aspect="equal",
    )

    # Camada 3: Contorno opcional da máscara
    if mask_slice is not None and np.any(mask_slice == contour_label):
        ax.contour(
            mask_slice == contour_label,
            levels=[0.5],
            colors=[contour_color],
            linewidths=2,
            origin="lower",
        )

    unit = "SUV" if pmax <= 20 else "Bq/mL"
    ax.set_title(
        f"{title}\n{plane.capitalize()} — Fatia {index} (0 a {pmax:.1f} {unit})",
        fontsize=11
    )
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Captação PET ({unit})", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pet_ct_fusion_three_planes(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    index_axial: Optional[int] = None,
    index_coronal: Optional[int] = None,
    index_sagital: Optional[int] = None,
    ct_wl: float = 40.0,
    ct_ww: float = 400.0,
    pet_cmap: str = "hot",
    pet_alpha: float = 0.5,
    pet_vmin: float = 0.0,
    pet_vmax: Optional[float] = None,
    title: str = "Fusão PET/CT — Três Planos",
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Fusão PET/CT nos três planos anatômicos simultaneamente com escala dinâmica
    calibrada e fundo transparente para voxels <= 0.
    """
    iz = index_axial   if index_axial is not None else ct_volume.shape[0] // 2
    iy = index_coronal if index_coronal is not None else ct_volume.shape[1] // 2
    ix = index_sagital if index_sagital is not None else ct_volume.shape[2] // 2

    planes = [
        ("axial",   iz),
        ("coronal", iy),
        ("sagital", ix),
    ]

    pmax = _get_robust_pet_vmax(pet_volume, pet_vmax)
    pmin = pet_vmin

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    last_im = None
    for ax, (plane, idx) in zip(axes, planes):
        ct_slice  = _get_slice(ct_volume,  plane, idx)
        pet_slice = _get_slice(pet_volume, plane, idx)

        ct_norm = _normalize_window(ct_slice, wl=ct_wl, ww=ct_ww)
        pet_overlay = np.ma.masked_where(pet_slice <= 0, pet_slice)

        ax.imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
        last_im = ax.imshow(
            pet_overlay,
            cmap=pet_cmap,
            alpha=pet_alpha,
            vmin=pmin,
            vmax=pmax,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(f"{plane.capitalize()} (i={idx})", fontsize=11)
        ax.axis("off")

    unit = "SUV" if pmax <= 20 else "Bq/mL"
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), fraction=0.02, pad=0.02, label=f"Captação PET ({unit})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# SEÇÃO 4: OVERLAY DE MÁSCARA DE SEGMENTAÇÃO
# =============================================================================

def plot_segmentation_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    plane: str = "axial",
    index: Optional[int] = None,
    bg_cmap: str = "gray",
    bg_wl: Optional[float] = None,
    bg_ww: Optional[float] = None,
    mask_colors: Optional[dict] = None,
    mask_alpha: float = 0.5,
    title: str = "Segmentação",
    figsize: Tuple[int, int] = (7, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Sobrepõe uma máscara de segmentação sobre um volume de fundo (CT ou PET).

    Suporta máscaras binárias (0/1) e multi-classe (0, 1, 2, 3...).

    Args:
        background:  Array 3D de fundo (z, y, x) — geralmente CT.
        mask:        Array 3D de máscara (z, y, x), valores inteiros por classe.
        plane:       Plano de corte.
        index:       Índice da fatia.
        bg_cmap:     Colormap do fundo.
        bg_wl, bg_ww: Windowing do fundo.
        mask_colors: Dict mapeando label → cor. Ex: {1: 'red', 2: 'blue'}.
                     Se None, usa um colormap discreto automático.
        mask_alpha:  Transparência da máscara (0–1).
        title:       Título da figura.
        figsize:     Tamanho da figura.
        save_path:   Caminho de saída.

    Returns:
        Objeto matplotlib Figure.

    Exemplo:
        >>> fig = plot_segmentation_overlay(
        ...     ct_arr, mask_arr, plane='axial',
        ...     mask_colors={1: 'red'}, mask_alpha=0.4,
        ...     bg_wl=40, bg_ww=400, title='Segmentação Tumor'
        ... )
    """
    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    if index is None:
        index = background.shape[dim_map[plane]] // 2

    bg_slice   = _get_slice(background, plane, index)
    mask_slice = _get_slice(mask,       plane, index)

    bg_norm = _normalize_window(bg_slice, wl=bg_wl, ww=bg_ww)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(bg_norm, cmap=bg_cmap, origin="lower", aspect="equal")

    labels = np.unique(mask_slice[mask_slice > 0])  # Exclui fundo (label 0)

    # Cores padrão se não fornecidas
    default_colors = [
        "#FF4444", "#4444FF", "#44FF44", "#FF8800",
        "#AA00FF", "#00FFFF", "#FFFF00", "#FF00AA",
    ]

    for i, label in enumerate(labels):
        if mask_colors and label in mask_colors:
            color = mask_colors[label]
        else:
            color = default_colors[i % len(default_colors)]

        binary = (mask_slice == label).astype(np.float32)

        # Cria RGBA colorido apenas onde a máscara está ativa
        rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
        rgb  = mcolors.to_rgb(color)
        rgba[..., 0] = rgb[0]
        rgba[..., 1] = rgb[1]
        rgba[..., 2] = rgb[2]
        rgba[..., 3] = binary * mask_alpha

        ax.imshow(rgba, origin="lower", aspect="equal")

    ax.set_title(f"{title}\n{plane.capitalize()} — Fatia {index}", fontsize=11)
    ax.axis("off")

    # Legenda manual para as classes
    if len(labels) > 0:
        legend_elements = []
        for i, label in enumerate(labels):
            color = (mask_colors or {}).get(label, default_colors[i % len(default_colors)])
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=mask_alpha, label=f"Classe {int(label)}")
            legend_elements.append(patch)
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# SEÇÃO 5: VISUALIZAÇÃO CARDÍACA — CROP E ZOOM NA REGIÃO DO CORAÇÃO
# =============================================================================

def get_cardiac_bounding_box(
    mask: np.ndarray,
    cardiac_labels: Optional[List[int]] = None,
    margin_voxels: int = 10,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calcula a bounding box 3D que engloba todas as estruturas cardíacas
    presentes na máscara, com margem adicional.

    Args:
        mask:           Array 3D (z, y, x) da máscara cardíaca.
        cardiac_labels: Lista de labels a incluir. Se None, usa todos > 0.
        margin_voxels:  Margem em voxels ao redor das estruturas.

    Returns:
        Tupla ((z_min, z_max), (y_min, y_max), (x_min, x_max)).

    Exemplo:
        >>> from config import ROTULOS_CARDIACOS
        >>> z_r, y_r, x_r = get_cardiac_bounding_box(mask_arr, margin_voxels=15)
        >>> ct_coracao = ct_arr[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
    """
    if cardiac_labels is not None:
        roi = np.zeros_like(mask, dtype=bool)
        for lbl in cardiac_labels:
            roi |= (mask == lbl)
    else:
        roi = mask > 0

    coords = np.where(roi)
    if coords[0].size == 0:
        raise ValueError("Nenhum voxel encontrado para os labels cardíacos especificados.")

    z_min = max(0, int(coords[0].min()) - margin_voxels)
    z_max = min(mask.shape[0], int(coords[0].max()) + margin_voxels + 1)
    y_min = max(0, int(coords[1].min()) - margin_voxels)
    y_max = min(mask.shape[1], int(coords[1].max()) + margin_voxels + 1)
    x_min = max(0, int(coords[2].min()) - margin_voxels)
    x_max = min(mask.shape[2], int(coords[2].max()) + margin_voxels + 1)

    return (z_min, z_max), (y_min, y_max), (x_min, x_max)


def plot_cardiac_zoom(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    pet_volume: Optional[np.ndarray] = None,
    cardiac_labels: Optional[List[int]] = None,
    label_names: Optional[Dict[int, str]] = None,
    label_colors: Optional[Dict[int, str]] = None,
    mask_colors: Optional[Dict[int, str]] = None,
    margin_voxels: int = 15,
    ct_wl: float = 100.0,
    ct_ww: float = 700.0,
    pet_alpha: float = 0.4,
    pet_cmap: str = "hot",
    pet_vmax: Optional[float] = None,
    mask_alpha: float = 0.45,
    title: str = "Região Cardíaca — Três Planos",
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Visualiza a região do coração nos três planos anatômicos, com crop automático
    ao redor das estruturas cardíacas da máscara.

    Etapas internas:
        1. Detecta bounding box do coração via máscara.
        2. Recorta CT (e PET se fornecido) nessa região.
        3. Exibe três planos com overlay colorido por estrutura.

    Args:
        ct_volume:      Array 3D CT (z, y, x), em HU.
        mask_volume:    Array 3D da máscara cardíaca (z, y, x), labels inteiros.
        pet_volume:     Array 3D PET (z, y, x), em SUV ou Bq/mL. Opcional.
        cardiac_labels: Labels a incluir na bbox. None = todos não-zero.
        label_names:    Dict {label: nome}. Ex: ROTULOS_CARDIACOS de config.py.
        label_colors:   Dict {label: cor hex}. Ex: CORES_CARDIACAS de config.py.
        mask_colors:    Alias para label_colors.
        margin_voxels:  Margem ao redor da bounding box cardíaca.
        ct_wl:          Window Level CT (padrão: janela cardíaca, 100 HU).
        ct_ww:          Window Width CT (padrão: janela cardíaca, 700 HU).
        pet_alpha:      Transparência do PET (0–1).
        pet_cmap:       Colormap do PET ('hot', 'inferno', etc.).
        pet_vmax:       Valor máximo do PET para normalização de cor.
        mask_alpha:     Transparência das máscaras de segmentação.
        title:          Título da figura.
        figsize:        Tamanho da figura em polegadas.
        save_path:      Caminho para salvar a imagem.
    """
    effective_colors = mask_colors if mask_colors is not None else label_colors

    default_colors = [
        "#FF4444", "#4488FF", "#44AAFF", "#FF8800",
        "#FFBB44", "#CC44CC", "#FF44AA", "#44CCAA",
    ]

    z_r, y_r, x_r = get_cardiac_bounding_box(
        mask_volume, cardiac_labels=cardiac_labels, margin_voxels=margin_voxels
    )
    ct_crop   = ct_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
    mask_crop = mask_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
    pet_crop  = (
        pet_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
        if pet_volume is not None else None
    )

    # Escala robusta pelo percentil 99 para evitar saturação
    pmax = _get_robust_pet_vmax(pet_crop if pet_crop is not None else pet_volume, pet_vmax)
    pmin = 0.0

    # Localizar fatia axial centralizada no miocárdio (label 1) se disponível
    voxels_mio_crop = np.sum(mask_crop == 1, axis=(1, 2))
    if voxels_mio_crop.max() > 0:
        iz = int(np.argmax(voxels_mio_crop))
    else:
        iz = ct_crop.shape[0] // 2

    iy = ct_crop.shape[1] // 2
    ix = ct_crop.shape[2] // 2

    planes = [
        ("axial",   iz, f"Axial (z={z_r[0]+iz})"),
        ("coronal", iy, f"Coronal (y={y_r[0]+iy})"),
        ("sagital", ix, f"Sagital (x={x_r[0]+ix})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    last_pet_im = None
    for ax, (plane, idx, plabel) in zip(axes, planes):
        ct_sl   = _get_slice(ct_crop,   plane, idx)
        mask_sl = _get_slice(mask_crop, plane, idx)
        ct_norm = _normalize_window(ct_sl, wl=ct_wl, ww=ct_ww)
        ax.imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")

        if pet_crop is not None:
            pet_sl   = _get_slice(pet_crop, plane, idx)
            pet_overlay = np.ma.masked_where(pet_sl <= 0, pet_sl)
            last_pet_im = ax.imshow(
                pet_overlay,
                cmap=pet_cmap,
                alpha=pet_alpha,
                vmin=pmin,
                vmax=pmax,
                origin="lower",
                aspect="equal",
            )

        # Contorno das estruturas presentes
        present_labels = np.unique(mask_sl[mask_sl > 0])
        for i, lbl in enumerate(present_labels):
            color  = (effective_colors or {}).get(int(lbl), default_colors[i % len(default_colors)])
            ax.contour(
                mask_sl == lbl,
                levels=[0.5],
                colors=[color],
                linewidths=2,
                origin="lower",
            )

        ax.set_title(plabel, fontsize=11)
        ax.axis("off")

    all_labels = np.unique(mask_crop[mask_crop > 0])
    patches = []
    for i, lbl in enumerate(all_labels):
        color = (effective_colors or {}).get(int(lbl), default_colors[i % len(default_colors)])
        name  = (label_names or {}).get(int(lbl), f"label {int(lbl)}")
        patches.append(mpatches.Patch(color=color, label=f"{int(lbl)}: {name}"))

    if patches:
        fig.legend(
            handles=patches, loc="lower center",
            ncol=min(len(patches), 5), fontsize=8,
            bbox_to_anchor=(0.5, -0.05), framealpha=0.8,
        )

    if last_pet_im is not None:
        unit = "SUV" if pmax <= 20 else "Bq/mL"
        fig.colorbar(last_pet_im, ax=axes.tolist(), fraction=0.02, pad=0.02, label=f"Captação PET ({unit})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# Alias de conveniência
plot_cardiac_zoom_3planes = plot_cardiac_zoom


def plot_alignment_verification_panel(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    mask_volume: np.ndarray,
    label: int = 1,
    structure_name: str = "Miocardio",
    slice_idx: Optional[int] = None,
    ct_vmin: float = -1000.0,
    ct_vmax: float = 400.0,
    pet_alpha: float = 0.5,
    pet_cmap: str = "hot",
    pet_vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (18, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Recria com fidelidade exata a visualização de alinhamento de referência
    (baseada em old/05_verificar_alinhamento_pet_ct_mascara.py):
        Coluna 1: CT anatômico (vmin=-1000, vmax=400)
        Coluna 2: CT + Contorno do Miocárdio (ax.contour em verde limão)
        Coluna 3: CT + PET reamostrado (fundo transparente masked_where) + Contorno do Miocárdio

    Calcula automaticamente a fatia axial Z que contém o pico de voxels da estrutura
    e o percentil 99 dos voxels positivos do PET para garantir escala e contraste ideais.
    """
    if slice_idx is None:
        voxels_por_fatia = np.sum(mask_volume == label, axis=(1, 2))
        if voxels_por_fatia.max() > 0:
            slice_idx = int(np.argmax(voxels_por_fatia))
        else:
            slice_idx = ct_volume.shape[0] // 2

    ct_slice = _get_slice(ct_volume, "axial", slice_idx)
    pet_slice = _get_slice(pet_volume, "axial", slice_idx)
    mask_slice = _get_slice(mask_volume, "axial", slice_idx)

    pmax = _get_robust_pet_vmax(pet_volume, pet_vmax)
    pet_overlay = np.ma.masked_where(pet_slice <= 0, pet_slice)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. CT Puro
    axes[0].imshow(ct_slice, cmap="gray", vmin=ct_vmin, vmax=ct_vmax, origin="lower", aspect="equal")
    axes[0].set_title("CT", fontsize=12)
    axes[0].axis("off")

    # 2. CT + Contorno
    axes[1].imshow(ct_slice, cmap="gray", vmin=ct_vmin, vmax=ct_vmax, origin="lower", aspect="equal")
    axes[1].contour(mask_slice == label, levels=[0.5], colors=["lime"], linewidths=2, origin="lower")
    axes[1].set_title(f"CT + contorno ({structure_name})", fontsize=12)
    axes[1].axis("off")

    # 3. CT + PET Reamostrado + Contorno
    axes[2].imshow(ct_slice, cmap="gray", vmin=ct_vmin, vmax=ct_vmax, origin="lower", aspect="equal")
    im = axes[2].imshow(
        pet_overlay,
        cmap=pet_cmap,
        alpha=pet_alpha,
        vmin=0,
        vmax=pmax,
        origin="lower",
        aspect="equal",
    )
    axes[2].contour(mask_slice == label, levels=[0.5], colors=["lime"], linewidths=2, origin="lower")
    axes[2].set_title(f"CT + PET reamostrado + label ({structure_name})", fontsize=12)
    axes[2].axis("off")

    unit = "SUV" if pmax <= 20 else "Bq/mL"
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label=f"Intensidade PET ({unit})")

    main_title = title or f"Alinhamento PET/CT/Máscara — Fatia Axial Z={slice_idx}"
    fig.suptitle(main_title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cardiac_fusion_panel(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    mask_volume: Optional[np.ndarray] = None,
    target_labels: Optional[List[int]] = None,
    mask_colors: Optional[Dict[int, str]] = None,
    plane: str = "axial",
    slice_idx: Optional[int] = None,
    margin_voxels: int = 15,
    ct_wl: float = 100.0,
    ct_ww: float = 700.0,
    pet_alpha: float = 0.5,
    pet_cmap: str = "hot",
    pet_vmin: float = 0.0,
    pet_vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (18, 5),
    title: str = "Painel de Fusão Cardíaca PET/CT",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Exibe um painel comparativo em 4 colunas com foco na anatomia cardíaca:
        1. CT puro com windowing cardíaco (100 HU / 700 HU)
        2. PET puro com fundo mascarado (colormap 'hot')
        3. Fusão PET/CT (CT grayscale + PET translúcido com máscara de captação)
        4. Contorno nítido das estruturas cardíacas segmentadas da máscara
    """
    default_colors = [
        "#FF4444", "#4488FF", "#44AAFF", "#FF8800",
        "#FFBB44", "#CC44CC", "#FF44AA", "#44CCAA",
    ]

    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    axis = dim_map.get(plane.lower(), 0)

    # Localizar fatia no pico do miocárdio (label 1) se disponível
    if slice_idx is None:
        if mask_volume is not None:
            if plane.lower() == "axial":
                voxels_mio = np.sum(mask_volume == 1, axis=(1, 2))
            elif plane.lower() == "coronal":
                voxels_mio = np.sum(mask_volume == 1, axis=(0, 2))
            else:
                voxels_mio = np.sum(mask_volume == 1, axis=(0, 1))

            if voxels_mio.max() > 0:
                slice_idx = int(np.argmax(voxels_mio))
            else:
                coords = np.where(mask_volume > 0)
                slice_idx = int(coords[axis].mean()) if coords[0].size > 0 else ct_volume.shape[axis] // 2
        else:
            slice_idx = ct_volume.shape[axis] // 2

    ct_sl = _get_slice(ct_volume, plane, slice_idx)
    pet_sl = _get_slice(pet_volume, plane, slice_idx)
    mask_sl = _get_slice(mask_volume, plane, slice_idx) if mask_volume is not None else None

    ct_norm = _normalize_window(ct_sl, wl=ct_wl, ww=ct_ww)

    pmax = _get_robust_pet_vmax(pet_volume, pet_vmax)
    pmin = pet_vmin
    pet_overlay = np.ma.masked_where(pet_sl <= 0, pet_sl)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    unit = "SUV" if pmax <= 20 else "Bq/mL"
    fig.suptitle(f"{title} — Plano {plane.capitalize()} (fatia {slice_idx})", fontsize=13, fontweight="bold")

    axes[0].imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
    axes[0].set_title("1. CT Anatômico (HU)", fontsize=11)
    axes[0].axis("off")

    im_pet = axes[1].imshow(pet_overlay, cmap=pet_cmap, origin="lower", aspect="equal", vmin=pmin, vmax=pmax)
    axes[1].set_title(f"2. PET Metabólico (0 a {pmax:.1f} {unit})", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
    axes[2].imshow(pet_overlay, cmap=pet_cmap, alpha=pet_alpha, origin="lower", aspect="equal", vmin=pmin, vmax=pmax)
    if mask_sl is not None and np.any(mask_sl == 1):
        axes[2].contour(mask_sl == 1, levels=[0.5], colors=["lime"], linewidths=2, origin="lower")
    axes[2].set_title("3. Fusão PET / CT", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
    if mask_sl is not None:
        labels_na_fatia = np.unique(mask_sl[mask_sl > 0])
        for idx, lbl in enumerate(labels_na_fatia):
            if target_labels is not None and lbl not in target_labels:
                continue
            color = (mask_colors or {}).get(int(lbl), default_colors[idx % len(default_colors)])
            axes[3].contour(mask_sl == lbl, levels=[0.5], colors=[color], linewidths=2, origin="lower")
    axes[3].set_title("4. Segmentação Cardíaca", fontsize=11)
    axes[3].axis("off")

    fig.colorbar(im_pet, ax=axes.tolist(), fraction=0.015, pad=0.02, label=f"Captação PET ({unit})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_myocardium_highlight(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    pet_volume: Optional[np.ndarray] = None,
    label_mio: int = 1,
    plane: str = "axial",
    margin_voxels: int = 20,
    ct_wl: float = 100.0,
    ct_ww: float = 700.0,
    pet_alpha: float = 0.5,
    pet_cmap: str = "hot",
    pet_vmin: float = 0.0,
    pet_vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 4),
    n_slices: int = 5,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Exibe uma sequência de fatias do miocárdio com crop cardíaco, delineamento
    em contorno e escala do PET calibrada dinamicamente pelo percentil 99.
    """
    z_r, y_r, x_r = get_cardiac_bounding_box(
        mask_volume, cardiac_labels=[label_mio], margin_voxels=margin_voxels
    )
    ct_crop   = ct_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
    mask_crop = mask_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
    pet_crop  = (
        pet_volume[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1]]
        if pet_volume is not None else None
    )

    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    n_total = ct_crop.shape[dim_map[plane]]
    indices = np.linspace(0, n_total - 1, n_slices, dtype=int)

    pmax = _get_robust_pet_vmax(pet_crop if pet_crop is not None else pet_volume, pet_vmax)
    pmin = pet_vmin

    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    fig.suptitle(
        f"Miocárdio (label={label_mio}) — {n_slices} fatias {plane}",
        fontsize=13, fontweight="bold"
    )

    last_im = None
    for ax, idx in zip(axes, indices):
        ct_sl   = _get_slice(ct_crop,   plane, int(idx))
        mask_sl = _get_slice(mask_crop, plane, int(idx))
        ct_norm = _normalize_window(ct_sl, wl=ct_wl, ww=ct_ww)
        ax.imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")

        if pet_crop is not None:
            pet_sl = _get_slice(pet_crop, plane, int(idx))
            pet_overlay = np.ma.masked_where(pet_sl <= 0, pet_sl)
            last_im = ax.imshow(
                pet_overlay,
                cmap=pet_cmap,
                alpha=pet_alpha,
                vmin=pmin,
                vmax=pmax,
                origin="lower",
                aspect="equal",
            )

        if np.any(mask_sl == label_mio):
            ax.contour(mask_sl == label_mio, levels=[0.5], colors=["lime"], linewidths=2, origin="lower")

        global_idx = (z_r[0] + idx) if plane == "axial" else int(idx)
        ax.set_title(f"i={global_idx}", fontsize=9)
        ax.axis("off")

    if last_im is not None:
        unit = "SUV" if pmax <= 20 else "Bq/mL"
        fig.colorbar(last_im, ax=axes, fraction=0.015, pad=0.02, label=f"PET ({unit})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_short_axis_triplet(
    ct_slice: np.ndarray,
    pet_slice: np.ndarray,
    mask_slice: Optional[np.ndarray] = None,
    contour_label: int = 1,
    ct_wl: float = 100.0,
    ct_ww: float = 700.0,
    pet_cmap: str = "hot",
    pet_alpha: float = 0.55,
    pet_vmin: float = 0.0,
    pet_vmax: Optional[float] = None,
    contour_color: str = "lime",
    title: str = "Short Axis Cardíaco — CT, PET e Fusão",
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota lado a lado o CT, o PET e a Fusão (Overlay) em formato Short Axis (Eixo Curto).

    Visualização padrão em cardiologia nuclear para avaliar a integridade estrutural,
    perfusão/viabilidade metabólica e alinhamento do anel miocárdico ("rosquinha").

    Args:
        ct_slice:      Fatia 2D do CT reorientada em Short Axis.
        pet_slice:     Fatia 2D do PET reorientada em Short Axis.
        mask_slice:    Fatia 2D da máscara cardíaca reorientada em Short Axis (opcional).
        contour_label: Rótulo da estrutura a ser contornada (padrão: 1 - miocárdio).
        ct_wl:         Nível de janela do CT (Window Level, padrão: 100 HU).
        ct_ww:         Largura de janela do CT (Window Width, padrão: 700 HU).
        pet_cmap:      Colormap para exibição do PET (padrão: 'hot').
        pet_alpha:     Transparência da sobreposição do PET na fusão (0.0 a 1.0).
        pet_vmin:      Valor mínimo para corte de intensidade do PET.
        pet_vmax:      Valor máximo para intensidade do PET (calculado se None).
        contour_color: Cor do contorno da máscara na fusão (padrão: 'lime').
        title:         Título principal da figura.
        figsize:       Dimensões da figura (largura, altura) em polegadas.
        save_path:     Caminho de saída para salvar o gráfico em arquivo (opcional).

    Returns:
        Figura do matplotlib contendo os 3 painéis lado a lado.
    """
    pmax = _get_robust_pet_vmax(pet_slice, pet_vmax)
    pmin = pet_vmin
    unit = "SUV" if pmax <= 30 else "Bq/mL"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # 1. CT puro (Short Axis)
    ct_norm = _normalize_window(ct_slice, wl=ct_wl, ww=ct_ww)
    axes[0].imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
    axes[0].set_title("CT Estrutural (Short Axis)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. PET puro (Short Axis)
    pet_vis = np.clip(pet_slice, pmin, pmax)
    im_pet = axes[1].imshow(pet_vis, cmap=pet_cmap, vmin=pmin, vmax=pmax, origin="lower", aspect="equal")
    axes[1].set_title(f"PET Metabólico ({unit})", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    cbar_pet = fig.colorbar(im_pet, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_pet.set_label(f"Intensidade ({unit})", fontsize=9)

    # 3. Fusão PET/CT + Contorno do Miocárdio
    axes[2].imshow(ct_norm, cmap="gray", origin="lower", aspect="equal")
    pet_overlay = np.ma.masked_where(pet_slice <= 0, pet_slice)
    im_fus = axes[2].imshow(
        pet_overlay, cmap=pet_cmap, alpha=pet_alpha, vmin=pmin, vmax=pmax, origin="lower", aspect="equal"
    )
    if mask_slice is not None and np.any(mask_slice == contour_label):
        axes[2].contour(
            mask_slice == contour_label,
            levels=[0.5],
            colors=[contour_color],
            linewidths=2,
            origin="lower",
        )
    axes[2].set_title("Fusão PET/CT + Contorno Miocárdico", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    cbar_fus = fig.colorbar(im_fus, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_fus.set_label(f"Overlay ({unit})", fontsize=9)

    # Anotações cardeais anatômicas (convenção radiológica: Septal à esq, Lateral à dir, Anterior em cima)
    for ax in axes:
        ax.text(0.5, 0.96, "ANT", transform=ax.transAxes, ha="center", va="top",
                fontsize=8, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
        ax.text(0.5, 0.04, "INF", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
        ax.text(0.04, 0.5, "SEP", transform=ax.transAxes, ha="left", va="center",
                fontsize=8, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
        ax.text(0.96, 0.5, "LAT", transform=ax.transAxes, ha="right", va="center",
                fontsize=8, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig

