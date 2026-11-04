"""
stats.py
========
Caracterização Física e Estatística de Volumes PET/CT — Foco Cardíaco.

Contexto:
    Módulo de análise quantitativa para volumes 3D de PET/CT, com ênfase na
    avaliação de estruturas cardíacas e análise de fibrose miocárdica.

    Funcionalidades:
        - Estatísticas globais de volume (intensidade, dimensões físicas).
        - Histogramas de intensidade (HU para CT, SUV para PET).
        - Estatísticas de VOI (Volume of Interest) por estrutura cardíaca.
        - Análise por estrutura usando os labels de ROTULOS_CARDIACOS (config.py).
        - Perfil axial de SUV do miocárdio (detectar regiões de fibrose).
        - Verificação de consistência geométrica PET/CT.
"""

from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =============================================================================
# SEÇÃO 1: ESTATÍSTICAS GLOBAIS DO VOLUME
# =============================================================================

def volume_statistics(volume: np.ndarray, spacing_mm: Optional[Tuple] = None) -> Dict:
    """
    Calcula estatísticas físicas e de intensidade de um volume 3D.

    Args:
        volume:     Array 3D (z, y, x) com valores de intensidade.
        spacing_mm: Tupla (sz, sy, sx) com o tamanho de cada voxel em mm.
                    Se fornecido, calcula dimensões físicas em cm e volume em mL.
                    Nota: 1 mL = 1 cm³ = 1000 mm³.

    Returns:
        dict com chaves:
            shape, n_voxels, dtype,
            min, max, mean, median, std, percentile_5, percentile_95,
            n_nonzero, fraction_nonzero,
            (se spacing) -> physical_size_mm, physical_size_cm, volume_mL
    """
    arr = volume.astype(np.float32)

    stats = {
        # Dimensões e estrutura
        "shape":            arr.shape,            # (z, y, x)
        "n_voxels":         arr.size,
        "dtype":            str(volume.dtype),

        # Intensidade
        "min":              float(arr.min()),
        "max":              float(arr.max()),
        "mean":             float(arr.mean()),
        "median":           float(np.median(arr)),
        "std":              float(arr.std()),
        "percentile_5":     float(np.percentile(arr, 5)),
        "percentile_95":    float(np.percentile(arr, 95)),

        # Esparsidade
        "n_nonzero":        int(np.count_nonzero(arr)),
        "fraction_nonzero": float(np.count_nonzero(arr) / arr.size),
    }

    if spacing_mm is not None:
        sz, sy, sx = float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])

        # Tamanho físico total do volume
        phys_z = arr.shape[0] * sz
        phys_y = arr.shape[1] * sy
        phys_x = arr.shape[2] * sx

        # Volume do voxel em mm³ e total em mL (1 mL = 1000 mm³)
        voxel_vol_mm3 = sz * sy * sx
        total_vol_mL  = arr.size * voxel_vol_mm3 / 1000.0

        stats["voxel_volume_mm3"]   = voxel_vol_mm3
        stats["physical_size_mm"]   = (phys_z, phys_y, phys_x)
        stats["physical_size_cm"]   = (phys_z/10, phys_y/10, phys_x/10)
        stats["total_volume_mL"]    = total_vol_mL

    return stats


def print_volume_statistics(stats: Dict, name: str = "Volume") -> None:
    """
    Imprime de forma formatada o dicionário gerado por volume_statistics().

    Args:
        stats: Dicionário retornado por volume_statistics().
        name:  Nome descritivo para o cabeçalho.
    """
    w = 55
    print(f"\n{'='*w}")
    print(f"  Estatísticas: {name}")
    print(f"{'='*w}")
    print(f"  Shape (z, y, x):      {stats['shape']}")
    print(f"  Nº de voxels:         {stats['n_voxels']:,}")
    print(f"  Tipo de dado:         {stats['dtype']}")
    print(f"  Mín / Máx:            {stats['min']:.4f} / {stats['max']:.4f}")
    print(f"  Média ± Desvio:       {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Mediana:              {stats['median']:.4f}")
    print(f"  Percentis 5/95:       {stats['percentile_5']:.4f} / {stats['percentile_95']:.4f}")
    print(f"  Voxels não-zero:      {stats['n_nonzero']:,}  ({stats['fraction_nonzero']*100:.1f}%)")

    if "physical_size_mm" in stats:
        pz, py, px = stats["physical_size_mm"]
        print(f"\n  --- Dimensões Físicas ---")
        print(f"  Tamanho (mm):         Z={pz:.1f}  Y={py:.1f}  X={px:.1f}")
        print(f"  Tamanho (cm):         Z={pz/10:.1f}  Y={py/10:.1f}  X={px/10:.1f}")
        print(f"  Vol. do voxel:        {stats['voxel_volume_mm3']:.4f} mm³")
        print(f"  Vol. total do FOV:    {stats['total_volume_mL']:.1f} mL")
    print(f"{'='*w}")


# =============================================================================
# SEÇÃO 2: HISTOGRAMA DE INTENSIDADE
# =============================================================================

def plot_intensity_histogram(
    volume: np.ndarray,
    n_bins: int = 256,
    title: str = "Histograma de Intensidade",
    xlabel: str = "Intensidade",
    color: str = "#4C9BE8",
    log_scale: bool = False,
    clip_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota o histograma de intensidade de um volume 3D.

    O histograma mostra a distribuição dos valores de voxel. É fundamental para:
        - CT: identificar picos característicos (ar, pulmão, tecido, osso).
        - PET: verificar a distribuição de captação (SUV) e threshold de lesões.
        - Planejamento de normalização para DL.

    Args:
        volume:      Array 3D (z, y, x).
        n_bins:      Número de bins do histograma.
        title:       Título do gráfico.
        xlabel:      Rótulo do eixo X (ex: "HU" para CT, "SUVbw" para PET).
        color:       Cor das barras.
        log_scale:   Se True, escala Y em log (útil quando há muito fundo).
        clip_range:  Tupla (min, max) para filtrar o histograma (ex: (-1000, 1000)).
        figsize:     Tamanho da figura.
        save_path:   Caminho de saída.

    Returns:
        Objeto matplotlib Figure.
    """
    arr = volume.flatten().astype(np.float32)

    if clip_range is not None:
        arr = arr[(arr >= clip_range[0]) & (arr <= clip_range[1])]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(arr, bins=n_bins, color=color, edgecolor="none", alpha=0.85)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Número de Voxels" + (" (log)" if log_scale else ""), fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Linhas de referência para CT (opcionais, ignoradas se não fizer sentido)
    if "HU" in xlabel or "Hounsfield" in xlabel:
        referencias = {
            "Ar (-1000)": -1000, "Água (0)": 0,
            "Tecido mole (~40)": 40, "Osso (~700)": 700
        }
        for label, val in referencias.items():
            if arr.min() <= val <= arr.max():
                ax.axvline(val, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=label)
        ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dual_histogram(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    label_a: str = "Volume A",
    label_b: str = "Volume B",
    n_bins: int = 200,
    clip_range_a: Optional[Tuple[float, float]] = None,
    clip_range_b: Optional[Tuple[float, float]] = None,
    title: str = "Comparação de Histogramas",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota dois histogramas lado a lado (ou sobrepostos) para comparação.

    Útil para comparar CT e PET, ou volume antes/depois de pré-processamento.

    Args:
        volume_a, volume_b: Arrays 3D (z, y, x).
        label_a, label_b:   Rótulos para a legenda.
        n_bins:             Número de bins.
        clip_range_a/b:     Tuplas (min, max) para cada histograma.
        title:              Título geral.
        figsize:            Tamanho da figura.
        save_path:          Caminho de saída.

    Returns:
        Objeto matplotlib Figure.
    """
    arr_a = volume_a.flatten().astype(np.float32)
    arr_b = volume_b.flatten().astype(np.float32)

    if clip_range_a:
        arr_a = arr_a[(arr_a >= clip_range_a[0]) & (arr_a <= clip_range_a[1])]
    if clip_range_b:
        arr_b = arr_b[(arr_b >= clip_range_b[0]) & (arr_b <= clip_range_b[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.hist(arr_a, bins=n_bins, color="#4C9BE8", edgecolor="none", alpha=0.85)
    ax1.set_title(label_a, fontsize=11)
    ax1.set_xlabel("Intensidade"); ax1.set_ylabel("Voxels")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(arr_b, bins=n_bins, color="#E84C4C", edgecolor="none", alpha=0.85)
    ax2.set_title(label_b, fontsize=11)
    ax2.set_xlabel("Intensidade"); ax2.set_ylabel("Voxels")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# SEÇÃO 3: ESTATÍSTICAS DE ROI / VOI (Região de Interesse 3D)
# =============================================================================

def roi_statistics(
    volume: np.ndarray,
    mask: np.ndarray,
    label: int = 1,
    spacing_mm: Optional[Tuple] = None,
) -> Dict:
    """
    Calcula estatísticas de intensidade dentro de uma máscara de segmentação (ROI/VOI).

    VOI = Volume of Interest (equivalente 3D do ROI 2D).

    Uso típico em PET/CT:
        - Calcular SUV médio, SUV máximo e SUV pico dentro de uma lesão.
        - Estimar o volume da lesão em mL a partir da máscara.
        - Calcular MTV (Metabolic Tumor Volume) e TLG (Total Lesion Glycolysis).

    Args:
        volume:     Array 3D de intensidade (z, y, x).
        mask:       Array 3D de máscara (z, y, x), mesmo shape do volume.
        label:      Valor inteiro da classe de interesse (ex: 1 para tumor).
        spacing_mm: Tupla (sz, sy, sx) em mm. Se fornecido, calcula volume em mL.

    Returns:
        dict com chaves: n_voxels_roi, min, max, mean, median, std,
                         percentile_5, percentile_95, (se spacing) volume_mL.

    Raises:
        ValueError: Se o volume e a máscara tiverem shapes diferentes.

    Exemplo:
        >>> roi_stats = roi_statistics(pet_suv, tumor_mask, label=1, spacing_mm=(3.27, 3.27, 3.27))
        >>> print(f"SUV máximo: {roi_stats['max']:.2f}")
        >>> print(f"Volume do tumor: {roi_stats['volume_mL']:.1f} mL")
    """
    if volume.shape != mask.shape:
        raise ValueError(
            f"Volume e máscara têm shapes diferentes: {volume.shape} vs {mask.shape}"
        )

    roi_voxels = volume[mask == label].astype(np.float32)

    if roi_voxels.size == 0:
        print(f"[AVISO] Nenhum voxel encontrado para label={label}.")
        return {"n_voxels_roi": 0}

    stats = {
        "label":            label,
        "n_voxels_roi":     int(roi_voxels.size),
        "min":              float(roi_voxels.min()),
        "max":              float(roi_voxels.max()),
        "mean":             float(roi_voxels.mean()),
        "median":           float(np.median(roi_voxels)),
        "std":              float(roi_voxels.std()),
        "percentile_5":     float(np.percentile(roi_voxels, 5)),
        "percentile_95":    float(np.percentile(roi_voxels, 95)),
    }

    if spacing_mm is not None:
        sz, sy, sx = float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])
        voxel_vol_mm3 = sz * sy * sx
        stats["voxel_volume_mm3"] = voxel_vol_mm3
        stats["volume_mL"]        = roi_voxels.size * voxel_vol_mm3 / 1000.0

        # Métricas específicas de PET (aplicáveis quando volume é SUVbw)
        # MTV = Volume metabólico (mL)
        # TLG = Glicolise total da lesão (SUVmean × MTV)
        stats["MTV_mL"] = stats["volume_mL"]
        stats["TLG"]    = float(roi_voxels.mean()) * stats["volume_mL"]

    return stats


def print_roi_statistics(stats: Dict, name: str = "ROI") -> None:
    """
    Imprime formatado o dicionário retornado por roi_statistics().

    Args:
        stats: Dicionário de roi_statistics().
        name:  Nome descritivo.
    """
    w = 55
    print(f"\n{'='*w}")
    print(f"  Estatísticas ROI/VOI: {name}  (label={stats.get('label','?')})")
    print(f"{'='*w}")
    print(f"  Voxels na ROI:        {stats['n_voxels_roi']:,}")
    print(f"  Mín / Máx:            {stats.get('min', 'N/A'):.4f} / {stats.get('max', 'N/A'):.4f}")
    print(f"  Média ± Desvio:       {stats.get('mean', 'N/A'):.4f} ± {stats.get('std', 'N/A'):.4f}")
    print(f"  Mediana:              {stats.get('median', 'N/A'):.4f}")
    print(f"  Percentis 5/95:       {stats.get('percentile_5', 'N/A'):.4f} / {stats.get('percentile_95', 'N/A'):.4f}")

    if "volume_mL" in stats:
        print(f"\n  --- Métricas de Volume (PET) ---")
        print(f"  Vol. do voxel:        {stats['voxel_volume_mm3']:.4f} mm³")
        print(f"  Volume da ROI:        {stats['volume_mL']:.2f} mL")
        print(f"  MTV (vol. metabólico):{stats['MTV_mL']:.2f} mL")
        print(f"  TLG (SUVmean×MTV):    {stats['TLG']:.2f}")
    print(f"{'='*w}")


# =============================================================================
# SEÇÃO 4: VERIFICAÇÃO DE CONSISTÊNCIA GEOMÉTRICA
# =============================================================================

def check_geometric_consistency(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    spacing_a: Tuple,
    spacing_b: Tuple,
    origin_a: Tuple,
    origin_b: Tuple,
    name_a: str = "Volume A",
    name_b: str = "Volume B",
    tolerance_mm: float = 0.5,
) -> Dict:
    """
    Verifica se dois volumes (ex: PET e CT) são geometricamente compatíveis
    para fusão direta (sem reamostramento).

    Dois volumes são compatíveis se:
        1. Têm o mesmo shape (z, y, x).
        2. Têm o mesmo spacing (tamanho de voxel).
        3. Têm a mesma origem (ponto de referência no espaço físico).

    Na prática, o PET tem resolução MUITO MENOR que o CT, então geralmente
    o PET precisa ser reamostrado para o grid do CT antes de qualquer fusão.

    Args:
        volume_a, volume_b: Arrays 3D.
        spacing_a, spacing_b: Tuplas de spacing (sz, sy, sx) em mm.
        origin_a, origin_b:   Tuplas de origem (oz, oy, ox) em mm.
        name_a, name_b:       Nomes para relatório.
        tolerance_mm:         Tolerância em mm para comparar spacings e origens.

    Returns:
        dict com chaves:
            same_shape, same_spacing, same_origin, is_compatible,
            shape_a, shape_b, spacing_a, spacing_b, origin_a, origin_b.
    """
    same_shape   = volume_a.shape == volume_b.shape
    same_spacing = all(
        abs(sa - sb) <= tolerance_mm
        for sa, sb in zip(spacing_a, spacing_b)
    )
    same_origin = all(
        abs(oa - ob) <= tolerance_mm
        for oa, ob in zip(origin_a, origin_b)
    )
    is_compatible = same_shape and same_spacing and same_origin

    result = {
        "is_compatible": is_compatible,
        "same_shape":    same_shape,
        "same_spacing":  same_spacing,
        "same_origin":   same_origin,
        f"shape_{name_a}":   volume_a.shape,
        f"shape_{name_b}":   volume_b.shape,
        f"spacing_{name_a}": spacing_a,
        f"spacing_{name_b}": spacing_b,
        f"origin_{name_a}":  origin_a,
        f"origin_{name_b}":  origin_b,
    }

    # Relatório no console
    status = "✅ COMPATÍVEIS" if is_compatible else "⚠️  INCOMPATÍVEIS — co-registro necessário"
    print(f"\n{'='*55}")
    print(f"  Verificação de Consistência Geométrica: {status}")
    print(f"{'='*55}")
    print(f"  {'':20s}  {name_a:>12s}   {name_b:>12s}")
    print(f"  {'Shape':20s}  {str(volume_a.shape):>12s}   {str(volume_b.shape):>12s}  {'✓' if same_shape else '✗'}")
    print(f"  {'Spacing (mm)':20s}  {str(tuple(f'{v:.2f}' for v in spacing_a)):>12s}   {str(tuple(f'{v:.2f}' for v in spacing_b)):>12s}  {'✓' if same_spacing else '✗'}")
    print(f"  {'Origin (mm)':20s}  {str(tuple(f'{v:.1f}' for v in origin_a)):>12s}   {str(tuple(f'{v:.1f}' for v in origin_b)):>12s}  {'✓' if same_origin else '✗'}")
    print(f"{'='*55}")

    return result


# =============================================================================
# SEÇÃO 5: ESTATÍSTICAS CARDÍACAS — ANÁLISE POR ESTRUTURA
# =============================================================================

def cardiac_structures_statistics(
    volume: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    label_names: Optional[Dict] = None,
    spacing_mm: Optional[Tuple] = None,
    modality: str = "PET (SUV)",
    ct_volume: Optional[np.ndarray] = None,
    pet_volume: Optional[np.ndarray] = None,
    mask_volume: Optional[np.ndarray] = None,
    cardiac_labels: Optional[Dict] = None,
) -> Dict[int, Dict]:
    """
    Calcula estatísticas de intensidade para CADA estrutura cardíaca
    presente na máscara, retornando um dicionário indexado por label.

    Suporta dois modos de uso:
        1. Análise conjunta de CT e PET:
           cardiac_structures_statistics(ct_volume=ct, pet_volume=pet, mask_volume=mask, cardiac_labels=labels)
        2. Análise de volume individual:
           cardiac_structures_statistics(volume=arr, mask=mask, label_names=labels)

    Args:
        volume:         Array 3D individual (se usado em modo simples).
        mask:           Array 3D da máscara.
        label_names:    Dicionário {label_int: nome_str}.
        spacing_mm:     Tupla com espaçamento físico (sz, sy, sx) em mm.
        modality:       Identificador textual da modalidade.
        ct_volume:      Array 3D de CT em HU (modo conjunto).
        pet_volume:     Array 3D de PET em SUV (modo conjunto).
        mask_volume:    Alias para mask (modo conjunto).
        cardiac_labels: Alias para label_names (modo conjunto).

    Returns:
        Dict {label_int: estatísticas_da_estrutura}.
    """
    # Resolver aliases
    actual_mask = mask_volume if mask_volume is not None else mask
    if actual_mask is None:
        raise ValueError("A máscara (mask ou mask_volume) deve ser fornecida.")

    actual_labels = cardiac_labels if cardiac_labels is not None else label_names
    if actual_labels is None:
        actual_labels = {}

    labels_presentes = sorted([
        lbl for lbl in np.unique(actual_mask)
        if lbl > 0 and lbl in actual_labels
    ])

    resultados = {}
    w = 60

    is_joint_mode = ct_volume is not None and pet_volume is not None

    print(f"\n{'='*w}")
    print(f"  Estatísticas por Estrutura Cardíaca — {'PET/CT Conjunto' if is_joint_mode else modality}")
    print(f"{'='*w}")

    for lbl in labels_presentes:
        nome = actual_labels.get(lbl, f"label_{lbl}")
        lbl_int = int(lbl)

        if is_joint_mode:
            stats_ct = roi_statistics(ct_volume, actual_mask, label=lbl_int, spacing_mm=spacing_mm)
            stats_pet = roi_statistics(pet_volume, actual_mask, label=lbl_int, spacing_mm=spacing_mm)

            vol_ml = stats_pet.get("volume_mL", 0.0)
            res_item = {
                "label": lbl_int,
                "label_name": nome,
                "CT_HU": stats_ct,
                "PET_SUV": stats_pet,
                "volume_ml": vol_ml,
                "n_voxels": stats_pet.get("n_voxels_roi", 0),
            }
            resultados[lbl_int] = res_item

            print(
                f"  [{lbl_int:2d}] {nome:<28s} "
                f"vol={vol_ml:>6.1f}mL | "
                f"SUV méd={stats_pet.get('mean', 0.0):>5.2f} (máx={stats_pet.get('max', 0.0):>5.2f}) | "
                f"HU méd={stats_ct.get('mean', 0.0):>6.1f}"
            )
        else:
            vol_target = volume if volume is not None else (ct_volume if ct_volume is not None else pet_volume)
            stats_roi = roi_statistics(
                volume=vol_target,
                mask=actual_mask,
                label=lbl_int,
                spacing_mm=spacing_mm,
            )
            stats_roi["label_name"] = nome
            resultados[lbl_int] = stats_roi

            vol_str = f"  | {stats_roi.get('volume_mL', 0.0):.1f} mL" if "volume_mL" in stats_roi else ""
            print(
                f"  [{lbl_int:2d}] {nome:<30s}  "
                f"n={stats_roi['n_voxels_roi']:>8,}  "
                f"média={stats_roi.get('mean', 0.0):>6.3f}  "
                f"máx={stats_roi.get('max', 0.0):>6.3f}{vol_str}"
            )

    print(f"{'='*w}")
    return resultados


def plot_cardiac_suv_profile(
    pet_volume: np.ndarray,
    mask_volume: np.ndarray,
    label_mio: int = 1,
    plane: str = "axial",
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Perfil de SUV Médio — Miocárdio",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plota o perfil de SUV médio do miocárdio ao longo de um eixo anatômico.

    O que este gráfico mostra:
        - Eixo X: índice de fatia (posição anatômica, ex: da base ao ápice).
        - Eixo Y: SUV médio do miocárdio naquela fatia (apenas voxels do label_mio).

    Interpretação para fibrose miocárdica:
        - Perfil UNIFORME: miocárdio metabolicamente homogêneo → sem fibrose evidente.
        - QUEDA LOCAL: região com captação reduzida → suspeita de fibrose, cicatriz
          ou hipoperfusão nessa porção do miocárdio.
        - AUSÊNCIA DE SINAL: fatia sem voxels do miocárdio (fora da extensão do coração).

    Args:
        pet_volume:   Array 3D PET (z, y, x), em SUV.
        mask_volume:  Array 3D da máscara (z, y, x).
        label_mio:    Label do miocárdio (padrão: 1).
        plane:        Eixo de análise: 'axial' (base→ápice), 'coronal', 'sagital'.
        figsize:      Tamanho da figura.
        title:        Título do gráfico.
        save_path:    Caminho de saída.

    Returns:
        Objeto matplotlib Figure.

    Exemplo:
        >>> fig = plot_cardiac_suv_profile(pet_suv_arr, mask_arr, label_mio=1)
        >>> plt.show()
    """
    dim_map = {"axial": 0, "coronal": 1, "sagital": 2}
    axis    = dim_map[plane]
    n_slices = pet_volume.shape[axis]

    suv_means  = []
    suv_stds   = []
    n_voxels   = []
    indices    = []

    for i in range(n_slices):
        if axis == 0:
            pet_sl  = pet_volume[i, :, :]
            mask_sl = mask_volume[i, :, :]
        elif axis == 1:
            pet_sl  = pet_volume[:, i, :]
            mask_sl = mask_volume[:, i, :]
        else:
            pet_sl  = pet_volume[:, :, i]
            mask_sl = mask_volume[:, :, i]

        roi_voxels = pet_sl[mask_sl == label_mio].astype(np.float32)

        if roi_voxels.size > 0:
            suv_means.append(float(roi_voxels.mean()))
            suv_stds.append(float(roi_voxels.std()))
            n_voxels.append(roi_voxels.size)
            indices.append(i)

    indices   = np.array(indices)
    means     = np.array(suv_means)
    stds      = np.array(suv_stds)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(indices, means - stds, means + stds, alpha=0.2, color="#E84C4C", label="±1 DP")
    ax.plot(indices, means, color="#E84C4C", linewidth=2, label="SUV médio")
    ax.axhline(means.mean(), linestyle="--", color="gray", linewidth=1,
               label=f"Média global: {means.mean():.2f}")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"Índice de fatia ({plane})", fontsize=11)
    ax.set_ylabel("SUV médio no miocárdio", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Anotação da queda máxima (possível região de fibrose)
    if len(means) > 0:
        idx_min = indices[np.argmin(means)]
        ax.axvline(idx_min, color="navy", linestyle=":", linewidth=1.2,
                   label=f"SUV mín: fatia {idx_min}")
        ax.annotate(
            f"Mín: {means.min():.2f}\n(fatia {idx_min})",
            xy=(idx_min, means.min()),
            xytext=(idx_min + max(len(indices)//10, 2), means.min()),
            arrowprops=dict(arrowstyle="->", color="navy"),
            fontsize=8, color="navy",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
