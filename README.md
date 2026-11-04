# Pipeline de Análise Cardíaca PET/CT — ENHANCE.PET 1.6k

> **Objetivo:** Pipeline modular e didático para manipular, co-registrar, quantificar e visualizar exames volumétricos (3D) de PET/CT com foco na **região cardíaca**, especialmente no **miocárdio (`label 1: heart_myocardium`)**.

---

## Visão Geral

Este repositório implementa um pipeline completo baseado no dataset **[ENHANCE.PET 1.6k](https://registry.opendata.aws/enhance-pet-1-6k/)** (disponível no AWS Open Data), cobrindo:

1. **Leitura e I/O 3D:** Carregamento de volumes NIfTI e metadados com SimpleITK e pydicom.
2. **Leitura de Metadados Auxiliares:** Integração tabular e parâmetros de aquisição de `PT-details.xlsx`, `CT-details.xlsx` e mapeamento anatômico de `label.json`.
3. **Co-registro e Resampling Espacial:** Reamostramento do volume PET e da Máscara Cardíaca para a referência anatômica do CT (mesmo grid de voxels, origem e direção).
4. **Quantificação Estatística de ROI:** Extração de volume real (mL), métricas de SUV (*Standardized Uptake Value*) e densidade tomográfica (HU) por estrutura cardíaca.

> ℹ **Escopo Metodológico:** Foco exclusivo em I/O físico, co-registro espacial, estatísticas de ROI e visualização interpretativa. O projeto não requer geração de tensores de Deep Learning ou data augmentation sintético.

---

## Estrutura do Repositório

```
estudando_pet_ct/
├── config.py                 # Caminhos reais, constantes físicas e dicionário de rótulos
├── requirements.txt          # Dependências Python (SimpleITK, pandas, openpyxl, etc.)
├── README.md                 # Guia geral do projeto
├── src/                      # Módulos Python reutilizáveis
│   ├── __init__.py           # Exportação dos submódulos
│   ├── io_utils.py           # Leitura NIfTI, DICOM, Excel (PT/CT) e JSON
│   ├── registration.py       # Resampling de volumes/máscaras e registro rígido
│   ├── stats.py              # Estatísticas de ROI, volume físico e perfil axial SUV
│   └── visualization.py      # Bounding box cardíaco, zoom 3D, fusão PET/CT e overlays
└── notebooks/                # Espaço reservado para cadernos interativos Jupyter
    └── README.md
```

---

## Configuração e Caminhos dos Dados

Os caminhos reais do dataset estão configurados centralmente em [`config.py`](config.py):

```python
from pathlib import Path

RAIZ = Path(r"d:\usuarios\estela.ribeiro\Meus Documentos\Datasets\ENHANCE-PET_1.6k")

# Mapeamento padrão por paciente (ex: '0001'):
# CT:                images/CT_0001-1000/<ID>.nii.gz
# PET:               images/PT/<ID>.nii.gz
# Máscara Cardíaca:  ground-truth/Cardiac/<ID>.nii.gz
# Metadados:         PT-details.xlsx, CT-details.xlsx, label.json
```

### Rótulos Anatômicos Cardíacos (`ground-truth/Cardiac/`)
- **1:** `heart_myocardium` (**Foco Principal: Fibrose Miocárdica**)
- **2:** `heart_atrium_left`
- **3:** `heart_atrium_right`
- **4:** `heart_ventricle_left`
- **5:** `heart_ventricle_right`
- **6:** `aorta`
- **7–12:** Vasos ilíacos, cava inferior e sistema porta
- **13:** `pulmonary_artery`

---

## Como Executar

### 1. Instalação do Ambiente

```bash
pip install -r requirements.txt
```

