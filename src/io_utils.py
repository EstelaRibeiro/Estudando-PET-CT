"""
io_utils.py
===========
Utilitários de Leitura e Carregamento de Volumes PET/CT.

Contexto para quem vem de Raio-X (2D):
    - Em Raio-X, você lida com imagens 2D (H x W) em formatos como PNG, JPEG ou DICOM.
    - Em PET/CT, o dado fundamental é um VOLUME 3D (H x W x D), onde D é a profundidade
      (número de fatias/slices). Cada elemento é chamado de VOXEL (volumetric pixel).
    - Os dois formatos principais são DICOM (.dcm) e NIfTI (.nii / .nii.gz).

Módulos suportados:
    - SimpleITK: leitura unificada de DICOM (série) e NIfTI, gerenciamento de metadados espaciais.
    - pydicom: acesso detalhado a tags DICOM individuais (necessário para extrair campos PET
      como RadiopharmaceuticalInformationSequence).
    - nibabel: leitura alternativa de NIfTI, muito usada em neuroimagem.
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List, Any

import numpy as np
import SimpleITK as sitk
import pydicom
import nibabel as nib
import pandas as pd

try:
    from config import XLSX_PT_PATH, XLSX_CT_PATH, LABELS_JSON, get_caminhos
except ImportError:
    RAIZ_PADRAO = Path(r"d:\usuarios\estela.ribeiro\Meus Documentos\Datasets\ENHANCE-PET_1.6k")
    XLSX_PT_PATH = RAIZ_PADRAO / "PT-details.xlsx"
    XLSX_CT_PATH = RAIZ_PADRAO / "CT-details.xlsx"
    LABELS_JSON  = RAIZ_PADRAO / "label.json"
    def get_caminhos(pid: str) -> dict:
        return {
            "CT": RAIZ_PADRAO / "images" / "CT_0001-1000" / f"{pid}.nii.gz",
            "PET": RAIZ_PADRAO / "images" / "PT" / f"{pid}.nii.gz",
            "Mascara_Cardiaca": RAIZ_PADRAO / "ground-truth" / "Cardiac" / f"{pid}.nii.gz",
        }


# =============================================================================
# SEÇÃO 1: LEITURA SIMPLEITK (Interface unificada para DICOM e NIfTI)
# =============================================================================

def load_sitk_image(path: Union[str, Path]) -> sitk.Image:
    """
    Carrega um volume 3D usando SimpleITK.

    Funciona para:
        - Arquivo NIfTI: forneça o caminho do arquivo (.nii ou .nii.gz).
        - Série DICOM:   forneça o caminho da PASTA contendo os arquivos .dcm.

    Args:
        path: Caminho para o arquivo NIfTI ou para a pasta DICOM.

    Returns:
        sitk.Image: Objeto de imagem SimpleITK com metadados espaciais embutidos
                    (spacing, origin, direction).

    Raises:
        FileNotFoundError: Se o caminho não existir.
        RuntimeError:      Se nenhum arquivo DICOM for encontrado na pasta.

    Exemplo:
        >>> ct = load_sitk_image("/dados/paciente01/CT/")
        >>> pet = load_sitk_image("/dados/paciente01/PET.nii.gz")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

    if path.is_dir():
        # --- Leitura de série DICOM ---
        # SimpleITK usa um "reader" especial que une fatias individuais em um volume 3D.
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))

        if not dicom_names:
            raise RuntimeError(
                f"Nenhuma série DICOM encontrada em: {path}\n"
                "Verifique se a pasta contém arquivos .dcm."
            )

        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()  # Preserva metadados de cada fatia
        reader.LoadPrivateTagsOn()                # Carrega tags privadas (ex: escala SUV)
        image = reader.Execute()
        return image

    else:
        # --- Leitura de arquivo único (NIfTI, MetaImage, etc.) ---
        image = sitk.ReadImage(str(path))
        return image


def get_spatial_metadata(image: sitk.Image) -> Dict:
    """
    Extrai os metadados espaciais fundamentais de um volume SimpleITK.

    Por que isso é importante?
        Em 2D, você raramente precisa saber o tamanho físico dos pixels.
        Em 3D, o SPACING (tamanho do voxel em mm) é CRÍTICO para:
            - Calcular volumes reais de lesões.
            - Garantir que PET e CT estejam no mesmo espaço antes de fusão.
            - Normalizar o input de redes neurais.

    Args:
        image: Imagem SimpleITK carregada.

    Returns:
        dict com chaves:
            - 'size':       Dimensões em voxels (x, y, z)
            - 'spacing':    Tamanho de cada voxel em mm (x, y, z)
            - 'origin':     Coordenada física do voxel (0,0,0) em mm
            - 'direction':  Matriz 3x3 de orientação (cosenos diretores)
            - 'n_components': Número de componentes por voxel (1 para escalar)
            - 'pixel_type': Tipo do dado (ex: 32-bit float)
    """
    direction_flat = image.GetDirection()
    direction_matrix = np.array(direction_flat).reshape(3, 3)

    return {
        "size":         image.GetSize(),          # (nx, ny, nz) em voxels
        "spacing":      image.GetSpacing(),        # (sx, sy, sz) em mm/voxel
        "origin":       image.GetOrigin(),         # (ox, oy, oz) em mm
        "direction":    direction_matrix,           # Matriz de orientação 3x3
        "n_components": image.GetNumberOfComponentsPerPixel(),
        "pixel_type":   sitk.GetPixelIDValueAsString(image.GetPixelID()),
    }


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """
    Converte um sitk.Image para um array NumPy, reordenando os eixos.

    Atenção ao ordering de eixos:
        SimpleITK usa (x, y, z) = (largura, altura, profundidade).
        NumPy usa a convenção (z, y, x) = (profundidade, altura, largura),
        que é mais intuitiva para indexação slice-por-slice.

    Args:
        image: Imagem SimpleITK.

    Returns:
        np.ndarray com shape (z, y, x), dtype preservado do original.
    """
    # GetArrayFromImage já retorna no formato (z, y, x)
    return sitk.GetArrayFromImage(image)


# =============================================================================
# SEÇÃO 2: LEITURA DICOM COM PYDICOM (Acesso detalhado a tags)
# =============================================================================

def load_dicom_series_pydicom(folder: Union[str, Path]) -> Tuple[np.ndarray, List[pydicom.Dataset]]:
    """
    Carrega uma série DICOM usando pydicom, retornando o volume e os datasets.

    Use esta função quando precisar acessar tags DICOM específicas que SimpleITK
    não expõe diretamente (ex: RadiopharmaceuticalInformationSequence do PET).

    Args:
        folder: Pasta contendo os arquivos .dcm da série.

    Returns:
        Tupla (volume, datasets) onde:
            - volume:   np.ndarray de shape (n_slices, height, width), dtype float32.
            - datasets: Lista de pydicom.Dataset, um por arquivo, ordenado por posição Z.

    Raises:
        FileNotFoundError: Se a pasta não existir ou não houver arquivos DICOM.
    """
    folder = Path(folder)
    dcm_files = sorted(folder.glob("*.dcm"))

    if not dcm_files:
        # Tenta sem extensão explícita (alguns arquivos DICOM não têm .dcm)
        dcm_files = [
            f for f in folder.iterdir()
            if f.is_file() and _is_dicom(f)
        ]
        dcm_files = sorted(dcm_files)

    if not dcm_files:
        raise FileNotFoundError(f"Nenhum arquivo DICOM encontrado em: {folder}")

    datasets = [pydicom.dcmread(str(f)) for f in dcm_files]

    # Ordena fatias pela posição Z (Image Position Patient, tag 0020,0032)
    datasets = _sort_slices_by_z(datasets)

    # Monta o volume 3D
    volume = np.stack(
        [ds.pixel_array.astype(np.float32) for ds in datasets],
        axis=0
    )  # shape: (n_slices, height, width)

    return volume, datasets


def _is_dicom(filepath: Path) -> bool:
    """Verifica se um arquivo é DICOM lendo os primeiros bytes (magic number)."""
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b"DICM"
    except Exception:
        return False


def _sort_slices_by_z(datasets: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    Ordena fatias DICOM pela posição Z (coordenada de Image Position Patient).
    Isso é essencial: os arquivos no disco podem não estar em ordem anatômica.
    """
    def get_z(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except (AttributeError, IndexError):
            return 0.0

    return sorted(datasets, key=get_z)


# =============================================================================
# SEÇÃO 3: EXTRAÇÃO DE METADADOS CT (Unidades Hounsfield)
# =============================================================================

def get_ct_metadata(datasets: List[pydicom.Dataset]) -> Dict:
    """
    Extrai metadados físicos relevantes de uma série CT.

    Unidades Hounsfield (HU):
        - A escala HU é padronizada internacionalmente:
            * Ar:        ~ -1000 HU
            * Pulmão:    ~ -500  HU
            * Gordura:   ~ -100  HU
            * Água:         0    HU
            * Músculo:   ~ +40   HU
            * Osso:      ~ +400 a +1000 HU
        - O pixel_array DICOM armazena valores brutos. Para converter para HU:
            HU = pixel_value * RescaleSlope + RescaleIntercept

    Args:
        datasets: Lista de pydicom.Dataset da série CT.

    Returns:
        dict com chaves: 'rescale_slope', 'rescale_intercept', 'spacing',
                         'rows', 'cols', 'n_slices', 'modality'.
    """
    ds = datasets[0]  # Metadados comuns a todas as fatias

    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # Spacing: (row_spacing, col_spacing) em mm, da tag PixelSpacing
    row_sp, col_sp = [float(v) for v in ds.PixelSpacing]

    # Espessura entre fatias (slice thickness)
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

    return {
        "modality":          getattr(ds, "Modality", "CT"),
        "rescale_slope":     slope,
        "rescale_intercept": intercept,
        "spacing":           (row_sp, col_sp, slice_thickness),  # (mm, mm, mm)
        "rows":              int(ds.Rows),
        "cols":              int(ds.Columns),
        "n_slices":          len(datasets),
    }


def apply_ct_rescale(volume: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """
    Aplica a transformação linear para converter pixel values em Unidades Hounsfield.

    Args:
        volume:    Array 3D com pixel values brutos.
        slope:     RescaleSlope (tag DICOM 0028,1053).
        intercept: RescaleIntercept (tag DICOM 0028,1052).

    Returns:
        Array 3D em Unidades Hounsfield (float32).
    """
    return volume.astype(np.float32) * slope + intercept


# =============================================================================
# SEÇÃO 4: EXTRAÇÃO DE METADADOS PET (SUV / Bq/mL)
# =============================================================================

def get_pet_metadata(datasets: List[pydicom.Dataset]) -> Dict:
    """
    Extrai metadados físicos de uma série PET, incluindo os parâmetros necessários
    para calcular o Standardized Uptake Value (SUV).

    O que é SUV?
        - O PET mede concentração de atividade em Bq/mL (Becquerel por mililitro).
        - Para comparação entre pacientes/scanners, normaliza-se pelo SUVbw:
            SUVbw = (Bq/mL) / (Dose_injetada_Bq / Peso_paciente_g)
        - O SUV remove variações de dose e peso corporal, tornando os valores
          comparáveis. Lesões ativas tipicamente têm SUV > 2.5.

    Args:
        datasets: Lista de pydicom.Dataset da série PET.

    Returns:
        dict com chaves: 'rescale_slope', 'rescale_intercept', 'units',
                         'radionuclide_total_dose', 'patient_weight_kg',
                         'radiopharmaceutical', 'spacing', 'n_slices'.
    """
    ds = datasets[0]

    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # Unidade dos pixel values (deve ser "BQML" para atividade absoluta)
    units = getattr(ds, "Units", "UNKNOWN")

    # Peso do paciente (em kg na tag, convertemos para g na aplicação do SUV)
    patient_weight_kg = float(getattr(ds, "PatientWeight", 0.0))

    # Informações do radiofármaco (tag complexa: sequence)
    radio_info = {}
    if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
        rseq = ds.RadiopharmaceuticalInformationSequence[0]
        radio_info = {
            "radiopharmaceutical": getattr(rseq, "Radiopharmaceutical", "N/A"),
            "radionuclide_total_dose": float(
                getattr(rseq, "RadionuclideTotalDose", 0.0)
            ),  # em Bq
        }

    # Spacing
    row_sp, col_sp = [float(v) for v in ds.PixelSpacing]
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

    return {
        "modality":                getattr(ds, "Modality", "PT"),
        "rescale_slope":           slope,
        "rescale_intercept":       intercept,
        "units":                   units,
        "patient_weight_kg":       patient_weight_kg,
        "spacing":                 (row_sp, col_sp, slice_thickness),
        "n_slices":                len(datasets),
        **radio_info,
    }


def compute_suvbw(
    volume_bqml: np.ndarray,
    total_dose_bq: float,
    patient_weight_kg: float,
) -> np.ndarray:
    """
    Converte um volume PET de Bq/mL para SUVbw (body-weight normalized).

    Fórmula:
        SUVbw[voxel] = (Bq/mL[voxel]) / (TotalDose_Bq / WeightPatient_g)

    Args:
        volume_bqml:       Array 3D com atividade em Bq/mL.
        total_dose_bq:     Dose total injetada em Becquerel (Bq).
        patient_weight_kg: Peso do paciente em kg.

    Returns:
        Array 3D com valores SUVbw (adimensional, g/mL).

    Raises:
        ValueError: Se dose ou peso forem zero (divisão por zero).
    """
    if total_dose_bq <= 0:
        raise ValueError("total_dose_bq deve ser maior que zero.")
    if patient_weight_kg <= 0:
        raise ValueError("patient_weight_kg deve ser maior que zero.")

    weight_g = patient_weight_kg * 1000.0  # kg → g
    suv = volume_bqml / (total_dose_bq / weight_g)
    return suv.astype(np.float32)


def parse_time_to_minutes(time_val: Any) -> Optional[float]:
    """
    Converte um valor de horário (datetime.time, datetime.datetime, string HH:MM:SS ou float)
    em minutos desde a meia-noite (0.0 a 1440.0).
    """
    if time_val is None:
        return None
    if isinstance(time_val, float) and np.isnan(time_val):
        return None
    from datetime import time as dt_time, datetime as dt_datetime
    if isinstance(time_val, (dt_time, dt_datetime)):
        return time_val.hour * 60.0 + time_val.minute + time_val.second / 60.0 + getattr(time_val, "microsecond", 0) / 60000000.0
    s = str(time_val).strip()
    if not s or s.lower() == "nan":
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            h = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            sec = float(parts[2]) if len(parts) > 2 else 0.0
            return h * 60.0 + m + sec / 60.0
        except ValueError:
            return None
    if len(s) == 6 and s.isdigit():
        try:
            h = float(s[:2])
            m = float(s[2:4])
            sec = float(s[4:6])
            return h * 60.0 + m + sec / 60.0
        except ValueError:
            return None
    try:
        val_f = float(s)
        if 0.0 <= val_f <= 1.0:
            return val_f * 1440.0
    except ValueError:
        pass
    return None


def calculate_suvbw_from_pt_metadata(
    volume_bqml: np.ndarray,
    pt_metadata: Dict[str, Any],
    decay_correction_mode: str = "acquisition",
    half_life_f18_min: float = 109.77,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calcula o volume 3D em SUVbw (Standardized Uptake Value, normalizado por peso corporal)
    a partir do volume PET original em Bq/mL e dos metadados de PT-details.xlsx.

    Fórmula:
        SUVbw = C_PET (Bq/mL) * (Peso [g] / Dose_Efetiva [Bq])
        onde:
            Peso [g] = Peso [kg] * 1000
            Dose_Efetiva:
                - Hipótese 'acquisition': Dose administrada decaída até o horário de aquisição:
                  Dose_Aquisicao = Dose_Injetada * 2^(-Δt / T_half)
                - Hipótese 'injection': Dose administrada no horário de injeção (caso a imagem
                  já tenha sido corrigida pelo scanner para o horário da administração).

    Args:
        volume_bqml:           Array 3D do PET com intensidades originais em Bq/mL.
        pt_metadata:           Dicionário com os campos de PT-details.xlsx.
        decay_correction_mode: Modo de decaimento ('acquisition' ou 'injection').
        half_life_f18_min:     Meia-vida do 18F em minutos (padrão: 109.77 min).

    Returns:
        Tupla (volume_suv, dict_info) com o volume em SUVbw e os parâmetros calculados.
    """
    if volume_bqml is None:
        raise ValueError("volume_bqml não pode ser None.")

    # 1. Extração do peso
    weight_kg = None
    for k in ["Weight [kg]", "Weight", "weight", "Peso", "peso"]:
        if k in pt_metadata and pt_metadata[k] is not None:
            val = pt_metadata[k]
            if not (isinstance(val, float) and np.isnan(val)):
                try:
                    weight_kg = float(val)
                    if weight_kg > 0:
                        break
                except (ValueError, TypeError):
                    pass

    # 2. Extração da dose administrada (em MBq)
    dose_mbq = None
    for k in ["Dose [MBq]", "Dose", "dose", "TotalDose", "Dose Administrada [MBq]"]:
        if k in pt_metadata and pt_metadata[k] is not None:
            val = pt_metadata[k]
            if not (isinstance(val, float) and np.isnan(val)):
                try:
                    dose_mbq = float(val)
                    if dose_mbq > 0:
                        break
                except (ValueError, TypeError):
                    pass

    if weight_kg is None or dose_mbq is None:
        info_fail = {
            "suv_applied": False,
            "weight_kg": weight_kg,
            "dose_mbq": dose_mbq,
            "reason": "Peso [kg] ou Dose [MBq] ausentes ou inválidos em pt_metadata.",
        }
        return volume_bqml.copy().astype(np.float32), info_fail

    weight_g = weight_kg * 1000.0
    dose_injecao_bq = dose_mbq * 1e6

    # 3. Horários de injeção e aquisição
    start_time_raw = None
    for k in ["Radiopharmaceutical Start Time", "Radiopharmaceutical,Start Time", "Start Time", "Injecao", "Radio Start Time"]:
        if k in pt_metadata and pt_metadata[k] is not None and str(pt_metadata[k]).strip() != "":
            start_time_raw = pt_metadata[k]
            break

    acq_time_raw = None
    for k in ["Acquisition Time", "AcquisitionTime", "Series Time", "SeriesTime", "Aquisicao"]:
        if k in pt_metadata and pt_metadata[k] is not None and str(pt_metadata[k]).strip() != "":
            acq_time_raw = pt_metadata[k]
            break

    t_start_min = parse_time_to_minutes(start_time_raw)
    t_acq_min = parse_time_to_minutes(acq_time_raw)

    intervalo_min = None
    fator_decaimento = 1.0

    if t_start_min is not None and t_acq_min is not None:
        delta = t_acq_min - t_start_min
        if delta < 0:
            delta += 1440.0  # cruzamento de meia-noite
        intervalo_min = delta
        fator_decaimento = 2.0 ** (-intervalo_min / half_life_f18_min)

    dose_aquisicao_bq = dose_injecao_bq * fator_decaimento

    # Fatores para as duas hipóteses
    fator_suv_inj = weight_g / dose_injecao_bq
    fator_suv_acq = weight_g / dose_aquisicao_bq

    # Seleção do modo
    if decay_correction_mode.lower() == "acquisition" and intervalo_min is not None:
        fator_utilizado = fator_suv_acq
        dose_utilizada_bq = dose_aquisicao_bq
        modo_escolhido = "acquisition (dose decaída na aquisição)"
    else:
        fator_utilizado = fator_suv_inj
        dose_utilizada_bq = dose_injecao_bq
        modo_escolhido = "injection (dose na injeção)"

    volume_suv = (volume_bqml * fator_utilizado).astype(np.float32)

    info = {
        "suv_applied": True,
        "weight_kg": weight_kg,
        "dose_mbq": dose_mbq,
        "dose_injecao_bq": dose_injecao_bq,
        "start_time": str(start_time_raw) if start_time_raw is not None else None,
        "acq_time": str(acq_time_raw) if acq_time_raw is not None else None,
        "intervalo_min": intervalo_min,
        "fator_decaimento": fator_decaimento,
        "dose_aquisicao_bq": dose_aquisicao_bq,
        "effective_dose_bq": dose_utilizada_bq,
        "suv_factor": fator_utilizado,
        "fator_suv_hipotese_a": fator_suv_inj,
        "fator_suv_hipotese_b": fator_suv_acq,
        "decay_mode": modo_escolhido,
    }

    return volume_suv, info


# =============================================================================
# SEÇÃO 5: LEITURA NIfTI COM NIBABEL
# =============================================================================

def load_nifti_nibabel(path: Union[str, Path]) -> Tuple[np.ndarray, nib.nifti1.Nifti1Header]:
    """
    Carrega um arquivo NIfTI usando nibabel (alternativa ao SimpleITK).

    Nibabel é amplamente utilizado em pipelines de neuroimagem (ex: FreeSurfer,
    FSL) e algumas competições de segmentação médica (ex: Medical Segmentation Decathlon).

    Args:
        path: Caminho para arquivo .nii ou .nii.gz.

    Returns:
        Tupla (data, header) onde:
            - data:   np.ndarray 3D (ou 4D se for série temporal), float32.
            - header: nib.nifti1.Nifti1Header com metadados (zooms = spacing, etc.).

    Exemplo:
        >>> data, header = load_nifti_nibabel("mascara.nii.gz")
        >>> print(header.get_zooms())  # spacing em mm
    """
    path = Path(path)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.header


def get_nifti_affine(path: Union[str, Path]) -> np.ndarray:
    """
    Retorna a matriz affine 4x4 de um NIfTI.

    A affine codifica: spacing, origin e direction em uma única matriz.
    É a forma canônica de descrever a geometria espacial em NIfTI.

    Returns:
        np.ndarray de shape (4, 4).
    """
    img = nib.load(str(path))
    return img.affine


# =============================================================================
# SEÇÃO 6: UTILIDADES GERAIS
# =============================================================================

def print_volume_summary(name: str, image: sitk.Image) -> None:
    """
    Imprime um resumo legível dos metadados espaciais de um volume.

    Args:
        name:  Nome descritivo para identificação (ex: "CT", "PET").
        image: Imagem SimpleITK.
    """
    meta = get_spatial_metadata(image)
    arr  = sitk_to_numpy(image)

    print(f"\n{'='*55}")
    print(f"  Volume: {name}")
    print(f"{'='*55}")
    print(f"  Dimensões (voxels): {meta['size']}  →  (x, y, z)")
    print(f"  Array shape (numpy): {arr.shape}    →  (z, y, x)")
    print(f"  Spacing (mm/voxel): {tuple(f'{v:.4f}' for v in meta['spacing'])}")
    print(f"  Origem (mm):        {tuple(f'{v:.2f}' for v in meta['origin'])}")
    print(f"  Tipo do pixel:      {meta['pixel_type']}")
    print(f"  Intensidade mín:    {arr.min():.4f}")
    print(f"  Intensidade máx:    {arr.max():.4f}")
    print(f"  Intensidade média:  {arr.mean():.4f}")
    print(f"{'='*55}")


# =============================================================================
# SEÇÃO 7: METADADOS AUXILIARES (EXCEL E JSON) E CARREGAMENTO DE EXAMES
# =============================================================================

def load_pt_details(xlsx_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Carrega a planilha de detalhes e metadados de aquisição PET (PT-details.xlsx).

    Contém informações essenciais de protocolo, dose, peso e parâmetros de scanner
    do dataset ENHANCE.PET 1.6k.

    Args:
        xlsx_path: Caminho para PT-details.xlsx. Se None, usa o caminho padrão de config.py.

    Returns:
        pd.DataFrame com os metadados dos exames PET.
    """
    path = Path(xlsx_path) if xlsx_path is not None else XLSX_PT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arquivo PT-details não encontrado em: {path}")

    return pd.read_excel(path)


def load_ct_details(xlsx_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Carrega a planilha de detalhes e metadados de aquisição CT (CT-details.xlsx).

    Contém parâmetros de tomografia (kVp, mA, espessura de corte, kernel, etc.)
    do dataset ENHANCE.PET 1.6k.

    Args:
        xlsx_path: Caminho para CT-details.xlsx. Se None, usa o caminho padrão de config.py.

    Returns:
        pd.DataFrame com os metadados dos exames CT.
    """
    path = Path(xlsx_path) if xlsx_path is not None else XLSX_CT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arquivo CT-details não encontrado em: {path}")

    return pd.read_excel(path)


def get_patient_metadata(
    patient_id: str,
    df_details: pd.DataFrame,
    id_column_candidates: Tuple[str, ...] = ("Patient", "ID", "PatientID", "id", "Subject", "Case", "Nome"),
) -> Dict[str, Any]:
    """
    Filtra os metadados de um paciente específico em um DataFrame de detalhes.

    Tenta converter e casar formatos como '0001', '1', 1 ou 'PETCT_0001'.

    Args:
        patient_id: Identificador do paciente (ex: '0001').
        df_details: DataFrame carregado de PT-details ou CT-details.
        id_column_candidates: Possíveis nomes para a coluna de identificação.

    Returns:
        Dict com as colunas e valores do paciente encontrado, ou vazio se não encontrado.
    """
    # Identificar a coluna de ID
    target_col = None
    for col in id_column_candidates:
        if col in df_details.columns:
            target_col = col
            break

    if target_col is None:
        # Pega a primeira coluna como fallback
        target_col = df_details.columns[0]

    series = df_details[target_col].astype(str).str.strip()

    # Formatos possíveis de busca
    pid_clean = str(patient_id).strip()
    pid_int = str(int(pid_clean)) if pid_clean.isdigit() else pid_clean

    mask = (series == pid_clean) | (series == pid_int) | (series.str.endswith(pid_clean))
    matched = df_details[mask]

    if not matched.empty:
        return matched.iloc[0].to_dict()

    return {}


def load_labels_json(json_path: Optional[Union[str, Path]] = None) -> Dict[int, str]:
    """
    Carrega o arquivo label.json que mapeia os índices numéricos das máscaras
    para seus respectivos nomes anatômicos.

    Args:
        json_path: Caminho para label.json. Se None, usa o caminho padrão de config.py.

    Returns:
        Dict mapeando índice inteiro -> nome da classe (ex: {1: "heart_myocardium"}).
    """
    path = Path(json_path) if json_path is not None else LABELS_JSON
    if not path.exists():
        alt_name = "label.json" if path.name == "labels.json" else "labels.json"
        alt_path = path.parent / alt_name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Arquivo de rótulos não encontrado em: {path} (nem em {alt_path})")

    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re
        # Remove comentários de linha única (// ...)
        cleaned = re.sub(r"//.*", "", content)
        # Remove trailing commas antes de } ou ]
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            data = json.loads(cleaned)
        except Exception:
            # Fallback para eval seguro de dict se for formato python/js
            import ast
            data = ast.literal_eval(cleaned)

    # Normalizar chaves para inteiros
    labels_dict = {}
    if isinstance(data, dict):
        if "labels" in data and isinstance(data["labels"], (dict, list)):
            data = data["labels"]
        elif "Cardiac" in data and isinstance(data["Cardiac"], (dict, list)):
            data = data["Cardiac"]
        elif "cardiac" in data and isinstance(data["cardiac"], (dict, list)):
            data = data["cardiac"]

    if isinstance(data, dict):
        for k, v in data.items():
            # Caso 1: k é o índice numérico (ex: "1": "heart_myocardium")
            try:
                lbl_idx = int(k)
                if isinstance(v, dict):
                    name = v.get("name", v.get("label", v.get("structure", str(k))))
                else:
                    name = str(v)
                labels_dict[lbl_idx] = str(name)
                continue
            except (ValueError, TypeError):
                pass

            # Caso 2: v é o índice numérico (ex: "heart_myocardium": 1)
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
                labels_dict[int(v)] = str(k)
            elif isinstance(v, dict):
                idx = v.get("id", v.get("label", v.get("index", v.get("value"))))
                if idx is not None:
                    try:
                        labels_dict[int(idx)] = str(k)
                    except (ValueError, TypeError):
                        pass

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                lbl_id = item.get("id", item.get("label", idx))
                lbl_name = item.get("name", str(item))
                try:
                    labels_dict[int(lbl_id)] = str(lbl_name)
                except (ValueError, TypeError):
                    labels_dict[idx] = str(lbl_name)
            else:
                labels_dict[idx] = str(item)

    return labels_dict


def load_patient_exam(
    patient_id: str = "0001",
    caminhos: Optional[Dict[str, Path]] = None,
    load_metadata_tables: bool = True,
    pt_details_path: Optional[Union[str, Path]] = None,
    ct_details_path: Optional[Union[str, Path]] = None,
    labels_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Carrega todos os dados associados a um paciente do ENHANCE.PET 1.6k:
        - Volume CT (sitk.Image)
        - Volume PET (sitk.Image)
        - Máscara Cardíaca (sitk.Image)
        - Metadados clínicos/técnicos de PT-details.xlsx e CT-details.xlsx
        - Mapeamento de rótulos de label.json

    Args:
        patient_id: Identificador do paciente (ex: '0001').
        caminhos: Dicionário com chaves 'CT', 'PET', 'Mascara_Cardiaca'. Se None,
                  usa get_caminhos(patient_id).
        load_metadata_tables: Se True, tenta ler PT-details.xlsx, CT-details.xlsx e label.json.
        pt_details_path: Caminho alternativo para PT-details.xlsx.
        ct_details_path: Caminho alternativo para CT-details.xlsx.
        labels_json_path: Caminho alternativo para label.json.

    Returns:
        Dicionário com:
            - 'patient_id': str
            - 'ct_image': sitk.Image
            - 'pet_image': sitk.Image
            - 'mask_image': sitk.Image (se existir)
            - 'pt_details': dict com metadados PET
            - 'ct_details': dict com metadados CT
            - 'labels': dict de rótulos anatômicos
            - 'paths': dict com os caminhos dos arquivos
    """
    if caminhos is None:
        caminhos = get_caminhos(patient_id)

    res: Dict[str, Any] = {
        "patient_id": patient_id,
        "paths": caminhos,
        "ct_image": None,
        "pet_image": None,
        "mask_image": None,
        "pt_details": {},
        "ct_details": {},
        "labels": {},
    }

    # 1. Carregar volumes NIfTI
    if "CT" in caminhos and Path(caminhos["CT"]).exists():
        res["ct_image"] = load_sitk_image(caminhos["CT"])
    else:
        raise FileNotFoundError(f"Arquivo CT não encontrado para paciente {patient_id}: {caminhos.get('CT')}")

    if "PET" in caminhos and Path(caminhos["PET"]).exists():
        res["pet_image"] = load_sitk_image(caminhos["PET"])
    else:
        raise FileNotFoundError(f"Arquivo PET não encontrado para paciente {patient_id}: {caminhos.get('PET')}")

    if "Mascara_Cardiaca" in caminhos and Path(caminhos["Mascara_Cardiaca"]).exists():
        res["mask_image"] = load_sitk_image(caminhos["Mascara_Cardiaca"])

    # 2. Carregar metadados auxiliares se solicitado
    if load_metadata_tables:
        # PT Details
        try:
            df_pt = load_pt_details(pt_details_path)
            res["pt_details"] = get_patient_metadata(patient_id, df_pt)
        except Exception as e:
            res["pt_details"] = {"error": str(e)}

        # CT Details
        try:
            df_ct = load_ct_details(ct_details_path)
            res["ct_details"] = get_patient_metadata(patient_id, df_ct)
        except Exception as e:
            res["ct_details"] = {"error": str(e)}

        # Label JSON
        try:
            res["labels"] = load_labels_json(labels_json_path)
        except Exception as e:
            # Fallback para rótulos de config.py se existir
            try:
                from config import ROTULOS_CARDIACOS
                res["labels"] = ROTULOS_CARDIACOS
            except ImportError:
                res["labels"] = {"error": str(e)}

    return res
