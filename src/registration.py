"""
registration.py
===============
Co-registro e Reamostramento Espacial PET/CT com SimpleITK.

Contexto:
    Em Raio-X (2D), as imagens já vêm registradas pelo posicionamento físico
    do paciente. Em PET/CT, embora o scanner adquira as duas modalidades em
    sequência, pode haver desalinhamento por:
        1. Movimento do paciente entre aquisições.
        2. Diferença de resolução (CT: ~0.5-1.5mm, PET: ~3-5mm por voxel).
        3. Diferentes grids de amostragem (spacing e origin).

    Antes de alimentar uma rede neural com volumes PET+CT concatenados,
    é INDISPENSÁVEL que ambos estejam no mesmo grid espacial.

Pipeline de co-registro:
    1. Reamostrar PET para o grid do CT (mesmo spacing, origin, size, direction).
    2. Opcionalmente, realizar registro rígido (translação + rotação) para
       corrigir pequenos deslocamentos de movimento.
    3. Registrar a máscara de segmentação no mesmo espaço.

Referências:
    - SimpleITK Notebooks: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks
    - Elastix registration: https://elastix.lumc.nl/
"""

from typing import Optional, Tuple, Dict, Union
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# =============================================================================
# SEÇÃO 1: REAMOSTRAMENTO (RESAMPLING)
# =============================================================================

def resample_to_reference(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """
    Reamostra (resample) uma imagem para o grid espacial de uma imagem de referência.

    O que é reamostramento?
        Imagine que o CT tem voxels de 1×1×3mm e o PET tem voxels de 4×4×4mm.
        Para trabalhar com ambos no mesmo espaço, precisamos "reprojetar" os
        voxels do PET para o grid do CT — criando uma versão do PET com o mesmo
        número de voxels e tamanho físico do CT.

    Interpoladores disponíveis:
        - sitkLinear:               Trilinear (padrão — suave, bom para PET/CT).
        - sitkNearestNeighbor:      Vizinho mais próximo (para MÁSCARAS — preserva labels inteiros).
        - sitkBSpline:              B-Spline (qualidade superior, mais lento).
        - sitkGaussian:             Gaussiano (suavizador).
        - sitkLabelGaussian:        Para máscaras com múltiplas classes.

    Args:
        moving_image:       Imagem a ser reamostrada (ex: PET de baixa resolução).
        reference_image:    Imagem de referência que define o grid de destino (ex: CT).
        interpolator:       Estratégia de interpolação.
        default_pixel_value: Valor para regiões fora do campo de visão do moving_image.

    Returns:
        sitk.Image reamostrada com o mesmo size, spacing, origin e direction
        da reference_image.

    Exemplo:
        >>> pet_resamp = resample_to_reference(pet_image, ct_image)
        >>> mask_resamp = resample_to_reference(
        ...     mask_image, ct_image,
        ...     interpolator=sitk.sitkNearestNeighbor
        ... )
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    # Usa a transformação identidade (sem deformação, apenas resampling de grid)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputPixelType(moving_image.GetPixelID())

    return resampler.Execute(moving_image)


def resample_to_spacing(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float],
    interpolator=sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """
    Reamostra um volume para um novo spacing isotropico ou anisotrópico.

    Útil para:
        - Uniformizar o spacing entre diferentes pacientes antes do treinamento.
        - Criar volumes isotropicos (ex: spacing 1×1×1mm) para redes 3D.
        - Reduzir resolução para diminuir custo computacional.

    Args:
        image:       Imagem SimpleITK de entrada.
        new_spacing: Tupla (sx, sy, sz) em mm/voxel (convenção SimpleITK: x,y,z).
        interpolator: Estratégia de interpolação.
        default_pixel_value: Valor de preenchimento para bordas.

    Returns:
        sitk.Image com o novo spacing e size ajustado proporcionalmente.

    Exemplo:
        >>> # Isotropizar CT para 1mm³:
        >>> ct_iso = resample_to_spacing(ct_image, (1.0, 1.0, 1.0))
    """
    original_spacing = image.GetSpacing()   # (sx, sy, sz) em mm
    original_size    = image.GetSize()      # (nx, ny, nz) em voxels

    # Calcula o novo tamanho em voxels para manter o tamanho físico
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetOutputPixelType(image.GetPixelID())

    print(f"  Reamostramento:")
    print(f"    Spacing: {original_spacing} → {new_spacing}")
    print(f"    Size:    {original_size} → {tuple(new_size)}")

    return resampler.Execute(image)


# =============================================================================
# SEÇÃO 2: REGISTRO RÍGIDO (RIGID REGISTRATION)
# =============================================================================

def rigid_registration(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    metric: str = "MattesMutualInformation",
    n_histogram_bins: int = 50,
    sampling_fraction: float = 0.01,
    n_iterations: int = 200,
    learning_rate: float = 1.0,
    verbose: bool = True,
) -> Tuple[sitk.Image, sitk.Transform, Dict]:
    """
    Realiza registro rígido (6 graus de liberdade: 3 translações + 3 rotações)
    entre uma imagem fixada (CT) e uma imagem móvel (PET).

    Quando usar registro rígido vs. deformável:
        - Rígido:      Adequado para corrigir movimentos do paciente (respiração,
                       pequenas mudanças de posição) em estruturas ósseas.
        - Deformável:  Necessário quando há deformação de tecidos moles
                       (ex: bexiga cheia vs. vazia, respiração abdominal).

    Métricas de similaridade:
        - MattesMutualInformation: Robusta para imagens multi-modais (CT+PET).
          Compara distribuições de intensidade, não valores absolutos.
        - MeanSquares: Para imagens da mesma modalidade.
        - Correlation: Para imagens com intensidades relacionadas linearmente.

    Args:
        fixed_image:       Imagem de referência (CT — maior resolução, fixo).
        moving_image:      Imagem a ser alinhada (PET — move para se alinhar ao CT).
        metric:            Métrica de similaridade ('MattesMutualInformation', etc.).
        n_histogram_bins:  Número de bins do histograma para MI.
        sampling_fraction: Fração dos voxels usada em cada iteração (1% = 0.01).
        n_iterations:      Número máximo de iterações do otimizador.
        learning_rate:     Taxa de aprendizado do otimizador de gradiente.
        verbose:           Imprime progresso durante o registro.

    Returns:
        Tupla (registered_image, transform, info_dict) onde:
            - registered_image: PET reamostrado no grid do CT, alinhado.
            - transform:        sitk.Transform com a transformação ótima encontrada.
            - info_dict:        dict com métricas finais e parâmetros da transformação.

    Exemplo:
        >>> pet_registered, tf, info = rigid_registration(ct_image, pet_resamp)
        >>> sitk.WriteImage(pet_registered, "pet_registered.nii.gz")
        >>> sitk.WriteTransform(tf, "transform.tfm")
    """
    # 1. Garante que o PET esteja no mesmo espaço (grid) do CT antes do registro
    #    (o registro vai ajustar o alinhamento fino, mas precisa começar no mesmo grid)
    moving_resampled = resample_to_reference(moving_image, fixed_image)

    # 2. Configura o framework de registro
    registration_method = sitk.ImageRegistrationMethod()

    # 2a. Métrica de similaridade
    if metric == "MattesMutualInformation":
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=n_histogram_bins
        )
        registration_method.SetMetricSamplingStrategy(
            registration_method.RANDOM
        )
        registration_method.SetMetricSamplingPercentage(sampling_fraction)
    elif metric == "MeanSquares":
        registration_method.SetMetricAsMeanSquares()
    elif metric == "Correlation":
        registration_method.SetMetricAsCorrelation()
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")

    # 2b. Interpolador para avaliação da métrica
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 2c. Otimizador: Gradient Descent com tamanho de passo adaptativo
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=n_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 2d. Estratégia multi-resolução (Gaussian pyramid):
    #     Começa com imagens suavizadas (resolução grossa) e refina.
    #     Evita mínimos locais e acelera o registro.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 2e. Transformação rígida com inicialização por centros de massa
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_resampled,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 3. Callback para monitoramento
    if verbose:
        def _print_iteration():
            print(
                f"  Iter {registration_method.GetOptimizerIteration():4d} | "
                f"Métrica: {registration_method.GetMetricValue():.6f} | "
                f"Posição: {registration_method.GetOptimizerPosition()}"
            )

        registration_method.AddCommand(
            sitk.sitkIterationEvent, _print_iteration
        )

    # 4. Executa o registro
    print("\n[Registro Rígido] Iniciando otimização...")
    final_transform = registration_method.Execute(fixed_image, moving_resampled)
    print("[Registro Rígido] Concluído!")

    # 5. Aplica a transformação final ao PET
    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    info = {
        "final_metric_value":   registration_method.GetMetricValue(),
        "optimizer_stop_cond":  registration_method.GetOptimizerStopConditionDescription(),
        "n_iterations_run":     registration_method.GetOptimizerIteration(),
        "transform_parameters": final_transform.GetParameters(),
        "transform_fixed_params": final_transform.GetFixedParameters(),
    }

    return registered_image, final_transform, info


# =============================================================================
# SEÇÃO 3: TRANSFORMAÇÃO DE MÁSCARAS
# =============================================================================

def apply_transform_to_mask(
    mask_image: sitk.Image,
    reference_image: sitk.Image,
    transform: Optional[sitk.Transform] = None,
) -> sitk.Image:
    """
    Aplica uma transformação (ou apenas reamostramento) a uma máscara de segmentação.

    Diferença crucial para imagens de intensidade:
        - Imagens de intensidade → interpolação LINEAR (suave, contínua).
        - Máscaras binárias/categóricas → interpolação NEAREST NEIGHBOR
          (preserva valores inteiros de label, sem criar valores intermediários).

    Args:
        mask_image:       Máscara de segmentação (sitk.Image, dtype inteiro ou float).
        reference_image:  Grid de referência (ex: CT reamostrado).
        transform:        Transformação sitk a aplicar. Se None, apenas reamostra o grid.

    Returns:
        sitk.Image da máscara no espaço do reference_image.
    """
    if transform is None:
        transform = sitk.Transform()  # Transformação identidade

    resampled_mask = sitk.Resample(
        mask_image,
        reference_image,
        transform,
        sitk.sitkNearestNeighbor,  # CRÍTICO: preserva labels inteiros
        0.0,                         # Background = 0
        mask_image.GetPixelID(),
    )

    return resampled_mask


# =============================================================================
# SEÇÃO 4: SALVAR E CARREGAR TRANSFORMAÇÕES
# =============================================================================

def save_transform(transform: sitk.Transform, output_path: Union[str, Path]) -> None:
    """
    Salva uma transformação SimpleITK em arquivo .tfm.

    O arquivo .tfm é texto legível contendo os parâmetros da transformação.
    Você pode carregá-lo depois com load_transform() para reproduzir o registro.

    Args:
        transform:    Transformação SimpleITK (ex: resultado de rigid_registration).
        output_path:  Caminho de saída (ex: "paciente01_transform.tfm").
    """
    sitk.WriteTransform(transform, str(output_path))
    print(f"Transformação salva em: {output_path}")


def load_transform(path: Union[str, Path]) -> sitk.Transform:
    """
    Carrega uma transformação SimpleITK de um arquivo .tfm.

    Args:
        path: Caminho para o arquivo .tfm.

    Returns:
        sitk.Transform carregada.
    """
    return sitk.ReadTransform(str(path))


def save_image(image: sitk.Image, output_path: Union[str, Path]) -> None:
    """
    Salva um sitk.Image em formato NIfTI (.nii.gz recomendado) ou outro suportado.

    Args:
        image:       Imagem a salvar.
        output_path: Caminho de saída (ex: "pet_registered.nii.gz").
    """
    sitk.WriteImage(image, str(output_path))
    print(f"Imagem salva em: {output_path}")


# =============================================================================
# SEÇÃO 5: PIPELINE COMPLETO DE CO-REGISTRO PET/CT
# =============================================================================

def full_pet_ct_registration_pipeline(
    ct_image: sitk.Image,
    pet_image: sitk.Image,
    mask_image: Optional[sitk.Image] = None,
    do_rigid_registration: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    patient_id: str = "patient",
    verbose: bool = True,
) -> Dict:
    """
    Pipeline completo de co-registro PET/CT:
        1. Reamostra PET para o grid do CT.
        2. (Opcional) Realiza registro rígido para alinhar PET ao CT.
        3. (Opcional) Reamostra/transforma a máscara de segmentação.
        4. (Opcional) Salva todos os resultados em output_dir.

    Args:
        ct_image:               Imagem CT (sitk.Image, imagem fixa).
        pet_image:              Imagem PET (sitk.Image, imagem móvel).
        mask_image:             Máscara de segmentação (sitk.Image ou None).
        do_rigid_registration:  Se True, realiza otimização de registro rígido.
                                Se False, faz apenas reamostramento de grid.
        output_dir:             Pasta de saída. Se None, não salva arquivos.
        patient_id:             Identificador do paciente para nomes de arquivo.
        verbose:                Imprime progresso.

    Returns:
        dict com chaves:
            'pet_resampled':    PET reamostrado (sem registro rígido).
            'pet_registered':   PET após registro (ou None se não aplicado).
            'mask_resampled':   Máscara reamostrada (ou None).
            'transform':        Transformação rígida (ou None).
            'registration_info': Info do registro (ou None).

    Exemplo:
        >>> results = full_pet_ct_registration_pipeline(
        ...     ct_image, pet_image, mask_image,
        ...     do_rigid_registration=True,
        ...     output_dir="resultados_registro/",
        ...     patient_id="0001"
        ... )
    """
    results = {
        "pet_resampled":    None,
        "pet_registered":   None,
        "mask_resampled":   None,
        "transform":        None,
        "registration_info": None,
    }

    if verbose:
        print("\n" + "="*55)
        print(f"  Pipeline de Co-registro PET/CT: {patient_id}")
        print("="*55)

    # --- Passo 1: Resampling do PET para o grid do CT ---
    if verbose:
        print("\n[1/3] Reamostramento PET → grid do CT...")
    pet_resampled = resample_to_reference(pet_image, ct_image, interpolator=sitk.sitkLinear)
    results["pet_resampled"] = pet_resampled

    # --- Passo 2: Registro Rígido (opcional) ---
    if do_rigid_registration:
        if verbose:
            print("\n[2/3] Registro Rígido...")
        pet_registered, transform, reg_info = rigid_registration(
            fixed_image=ct_image,
            moving_image=pet_resampled,
            verbose=verbose,
        )
        results["pet_registered"]    = pet_registered
        results["transform"]         = transform
        results["registration_info"] = reg_info
    else:
        if verbose:
            print("\n[2/3] Registro Rígido ignorado (apenas resampling de grid).")
        results["pet_registered"] = pet_resampled

    # --- Passo 3: Transformação da máscara (opcional) ---
    if mask_image is not None:
        if verbose:
            print("\n[3/3] Reamostramento da Máscara...")
        tf_for_mask = results.get("transform", None)
        mask_resampled = apply_transform_to_mask(mask_image, ct_image, transform=tf_for_mask)
        results["mask_resampled"] = mask_resampled
    else:
        if verbose:
            print("\n[3/3] Nenhuma máscara fornecida.")

    # --- Salvar resultados ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_image(results["pet_registered"], output_dir / f"{patient_id}_PET_registrado.nii.gz")

        if results["transform"] is not None:
            save_transform(results["transform"], output_dir / f"{patient_id}_transform.tfm")

        if results["mask_resampled"] is not None:
            save_image(results["mask_resampled"], output_dir / f"{patient_id}_mascara_registrada.nii.gz")

    if verbose:
        print("\n✅ Pipeline concluído com sucesso!")
        print("="*55)

    return results


# Alias de conveniência
run_registration_pipeline = full_pet_ct_registration_pipeline
