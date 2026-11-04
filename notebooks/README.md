# Notebooks de Estudo

Esta pasta contém uma sequência didática de notebooks (`.ipynb`) preparados para acompanhar o aprendizado prático passo a passo.

---

## 📚 Sequência Recomendada de Estudos

| Notebook | Tópico Abordado | Módulos Relacionados |
|---|---|---|
| [**01 — Leitura, Metadados e Formatos**](01_fundamentos_leitura_e_metadados.ipynb) | Carregamento de NIfTI, inspeção de headers, metadados espaciais (spacing, origin, direction), leitura de planilhas de aquisição (`PT-details.xlsx` e `CT-details.xlsx`) e rótulos (`label.json`). | [`src/io_utils.py`](../src/io_utils.py) |
| [**02 — Reamostragem e Co-registro Espacial**](02_coregistro_e_resampling.ipynb) | Harmonização do grid do PET e da Máscara para a grade do CT com SimpleITK. Diferença prática entre interpolação linear e vizinho mais próximo. | [`src/registration.py`](../src/registration.py) |
| [**03 — Quantificação, Estatísticas de VOI e Calibração SUV**](03_quantificacao_estatisticas_e_suv.ipynb) | Cálculo de volume real em mililitros ($\text{mL}$), conversão de $\text{Bq/mL}$ para $\text{SUV}_{\text{bw}}$, correção de meia-vida do $^{18}\text{F}$ e detecção de heterogeneidade metabólica. | [`src/stats.py`](../src/stats.py) |
| [**04 — Visualização Multi-planar e Fusão PET/CT**](04_visualizacao_e_fusao_pet_ct.ipynb) | Janelamento tomográfico (HU windowing), cortes ortogonais (Axial, Coronal, Sagital), fusão térmica com transparência (alpha blending) e contorno do miocárdio. | [`src/visualization.py`](../src/visualization.py) |

---

## 🚀 Como Executar os Notebooks

1. Certifique-se de que o ambiente Python possui as dependências instaladas:
   ```bash
   pip install -r requirements.txt
   pip install jupyter
   ```
2. Inicie o Jupyter no diretório raiz do projeto:
   ```bash
   jupyter notebook
   ```
3. Abra a pasta `notebooks/` e execute as células sequencialmente. Todos os cadernos possuem código autoexplicativo, verificações defensivas e explicações detalhadas em células Markdown.
