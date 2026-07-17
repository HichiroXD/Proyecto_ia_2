# Proyecto: Clasificación de Discurso de Odio en Twitter mediante DistilBERT

## Integrantes

- Eduardo Sáez

---

## Descripción del Problema

La detección automática de discurso de odio en redes sociales constituye un desafío relevante para la moderación de contenido debido al gran volumen de publicaciones generadas diariamente y a la diversidad de formas en que este tipo de lenguaje puede manifestarse. Los sistemas de moderación manual resultan insuficientes para analizar millones de publicaciones en tiempo real, por lo que los modelos de aprendizaje profundo representan una alternativa para asistir en esta tarea.

Este proyecto busca desarrollar un modelo capaz de clasificar publicaciones de Twitter según la categoría de discurso de odio presente en el texto de cada publicación, utilizando un modelo Transformer preentrenado ajustado mediante *fine-tuning*.

Es importante delimitar el alcance del proyecto. Aunque el dataset seleccionado es multimodal e incluye imágenes asociadas a cada tweet, este trabajo se enfocará exclusivamente en el contenido textual (`tweet_text`). De esta forma, se evaluará la capacidad de un modelo de lenguaje para identificar patrones lingüísticos asociados al discurso de odio sin utilizar información visual.

El modelo no pretende comprender el contexto completo de una publicación ni sustituir sistemas de moderación humana, sino aprender patrones estadísticos presentes en un conjunto de datos previamente anotado para estimar la categoría más probable de una publicación textual.

---

## Descripción del Dataset y Fuente

Se utilizará el dataset **MMHS150K (Multi-Modal Hate Speech Dataset)** disponible en Kaggle.

El conjunto de datos contiene aproximadamente **150.000 publicaciones de Twitter**, donde cada publicación está asociada a un tweet, una imagen y anotaciones realizadas por tres evaluadores independientes.

Para este proyecto únicamente se utilizarán los siguientes recursos:

- `MMHS150K_GT.json`
- `splits/train_ids.txt`
- `splits/val_ids.txt`
- `splits/test_ids.txt`

Estos archivos proporcionan las particiones oficiales de entrenamiento, validación y prueba definidas por los autores del dataset.

Aunque el dataset incluye imágenes (`img_resized`) y el texto obtenido mediante OCR (`img_txt`), dichos recursos no serán utilizados en esta primera versión del proyecto.

**Nota sobre calidad de datos:** durante la construcción inicial del dataset se detectó que, en la primera descarga, `train_ids.txt` y `test_ids.txt` contenían exactamente los mismos 10,000 IDs (error de copiado de archivos durante la descarga). Esto se identificó mediante verificación de intersección de conjuntos y se corrigió volviendo a descargar el dataset, confirmando splits disjuntos con los tamaños oficiales: `train` (134,823), `val` (5,000) y `test` (10,000). El proceso de validación queda documentado en `01_build_dataset.ipynb`.

Otro hallazgo relevante: los splits `val` y `test` están **balanceados casi perfectamente (~50% Hate / ~50% NotHate)**, mientras que `train` conserva la distribución natural del dataset (~78% NotHate / ~22% Hate). Esto es consistente con una práctica común de diseño de datasets: balancear los conjuntos de evaluación para obtener métricas más interpretables, dejando el desbalance real únicamente en el conjunto de entrenamiento.

Fuente: https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech

### Variables utilizadas

#### Feature principal

- `tweet_text`: contenido textual de la publicación.

Se realizará una limpieza mínima del texto:

- Eliminación de URLs.
- Eliminación del prefijo `RT`.
- Normalización de espacios.

No se eliminarán emojis, puntuación ni mayúsculas.

#### Variable objetivo

Cada publicación posee tres anotaciones independientes correspondientes
a las categorías:

| Valor | Categoría |
|------:|-----------|
| 0     | NotHate   |
| 1     | Racist    |
| 2     | Sexist    |
| 3     | Homophobe |
| 4     | Religion  |
| 5     | OtherHate |

La etiqueta final se obtendrá mediante **votación mayoritaria** entre los tres anotadores. Se calcularán y compararán **dos esquemas de etiquetado** durante la construcción del dataset y el EDA:

**Esquema multiclase (6 categorías):** se aplica votación mayoritaria directamente sobre las 6 categorías originales. En los casos donde los tres anotadores entreguen valores distintos entre sí (ej. `[1, 3, 5]`) y por lo tanto no exista una categoría mayoritaria, la muestra será **descartada** del dataset final para este esquema.

**Esquema binario (Hate / NotHate):** se binariza primero cada voto (`0` → NotHate, `1-5` → Hate) y luego se aplica votación mayoritaria. Con solo dos categorías posibles y tres votos, siempre existe una mayoría, por lo que **no se descarta ninguna muestra** en este esquema.

Ambos esquemas se calcularon y compararon en el EDA (`02_eda.ipynb`) obteniendo los siguientes resultados sobre las 149,823 muestras totales:

| Esquema | Muestras descartadas | Distribución de clases |
|---|---|---|
| Multiclase (6 clases) | 11,714 (7.82%) | NotHate 112,844 · Racist 11,926 · OtherHate 5,811 · Homophobe 3,870 · Sexist 3,495 · **Religion 163 (0.13%)** |
| Binario (Hate/NotHate) | 0 (0.00%) | NotHate 112,852 (75.4%) · Hate 36,971 (24.6%) |

**Decisión final: se utilizará el esquema binario (Hate / NotHate)** para el entrenamiento del modelo.

**Justificación:** el esquema multiclase presenta un desbalance severo en varias categorías, en particular `Religion`, que representa apenas el 0.13% del dataset (163 muestras). Entrenar un clasificador de 6 clases con una categoría minoritaria de ese tamaño, en el plazo disponible para este proyecto, no permitiría obtener un modelo capaz de generalizar sobre dicha clase. El esquema binario, en cambio, presenta un desbalance moderado (~3:1) que es perfectamente manejable mediante técnicas estándar de regularización, como ponderación de clases (`class weights`) en la función de pérdida.

---

## Justificación del Modelo Seleccionado

Se utilizará **DistilBERT**, una versión compacta de BERT basada en *knowledge distillation*.

### Ventajas

1. Menor costo computacional que BERT Base.
2. Excelente desempeño en clasificación de texto.
3. Compatible con Hugging Face Transformers.
4. Adecuado para *fine-tuning* sobre datasets de tamaño medio.

### Limitaciones del modelo

1. Menor capacidad representacional que modelos Transformer de mayor tamaño (ej. BERT Base, RoBERTa).

### Limitaciones del alcance del proyecto

1. Al utilizar únicamente `tweet_text`, el modelo no accede a la información visual del meme asociado a cada publicación.
2. Parte del discurso de odio puede estar codificado exclusivamente en la imagen, lo que puede limitar el *recall* del modelo en esos casos.

---

## Metodología Aplicada (Plan de Acción Paso a Paso)

1. Construcción del dataset.
2. Análisis Exploratorio de Datos (EDA).
3. Preprocesamiento del texto.
4. Fine-tuning de DistilBERT.
5. Control de overfitting mediante Early Stopping, Weight Decay y Dropout.
6. Evaluación mediante Accuracy, Precision, Recall, F1-score y matriz de confusión.

---

## Estructura del Repositorio

```
Proyecto/
│
├── data/
│   ├── raw/
│   │   ├── MMHS150K_GT.json
│   │   └── splits/
│   │       ├── train_ids.txt
│   │       ├── val_ids.txt
│   │       └── test_ids.txt
│   │
│   └── processed/
│       ├── dataset.csv
│       └── resultados_test.json
│
├── notebooks/
│   ├── 01_build_dataset.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_train.ipynb
│   └── 04_evaluate.ipynb
│
├── README.md
└── environment.yml
```

---

## Dependencias

Las dependencias del proyecto se encuentran especificadas en el archivo [`environment.yml`](./environment.yml), el cual permite recrear el entorno de trabajo mediante:

```bash
conda env create -f environment.yml
```

---

## Estado del Proyecto

- [x] Definición del problema.
- [x] Selección del dataset.
- [x] Selección del modelo.
- [x] Construcción del dataset.
- [x] EDA.
- [x] Entrenamiento.
- [x] Evaluación.
- [x] Conclusiones.

---

## Resultados Obtenidos

El modelo fue evaluado sobre el conjunto de **test** (10,000 tweets, balanceado ~50% NotHate / 50% Hate), que no fue utilizado en ninguna etapa de entrenamiento ni validación.

### Métricas finales

| Métrica   | Valor  |
|-----------|-------:|
| Accuracy  | 68.14% |
| Precision | 69.01% |
| Recall    | 65.89% |
| F1-score  | 67.41% |

Dado que el set de test está balanceado, un modelo sin capacidad predictiva real rondaría el 50% de accuracy. El 68.14% obtenido confirma que el modelo capturó señal real del lenguaje asociado al discurso de odio, aunque con margen de mejora.

### Matriz de confusión

|                    | Predicho NotHate | Predicho Hate |
|--------------------|-----------------:|--------------:|
| **Real NotHate**   | 3,519            | 1,480         |
| **Real Hate**      | 1,706            | 3,295         |

Los falsos positivos (1,480) y falsos negativos (1,706) están en magnitudes similares, lo que indica que el modelo no está sesgado fuertemente hacia sobre-predecir o sub-predecir la clase `Hate`.

### Curva de entrenamiento y control de overfitting

Durante el fine-tuning, el `EarlyStoppingCallback` detuvo el entrenamiento en la época 3 de 4 planificadas, tras detectar que la pérdida de validación aumentaba de forma sostenida mientras la pérdida de entrenamiento continuaba bajando (evidencia clara de sobreajuste a partir de la época 2). El modelo final corresponde a la **época 1**, que obtuvo el mejor F1 de validación (0.685), gracias a `load_best_model_at_end=True`.

| Época | Train Loss | Val Loss | F1 (val) |
|-------|-----------:|---------:|---------:|
| 1     | 0.595      | 0.641    | **0.685** |
| 2     | 0.592      | 0.671    | 0.641    |
| 3     | 0.566      | 0.777    | 0.599    |

### Análisis de errores

Se inspeccionaron ejemplos concretos de falsos positivos y falsos negativos para entender las limitaciones reales del modelo más allá de las métricas agregadas:

- **Falsos positivos** (NotHate clasificado como Hate): en su mayoría corresponden a tweets con lenguaje informal o reapropiado entre usuarios (insultos usados en tono de broma o cercanía) que contienen términos fuertes sin intención de odio real.
- **Falsos negativos** (Hate clasificado como NotHate): varios casos corresponden a tweets donde una palabra fuerte se usa en un tono claramente casual o coloquial, pero fue etiquetada como `Hate` por los anotadores humanos. Esto sugiere que parte del error no es atribuible únicamente al modelo, sino a la ambigüedad inherente de la tarea: incluso anotadores humanos difieren en la interpretación de ciertos términos según el contexto social del hablante.

Este patrón indica que, si bien DistilBERT captura mucho más contexto que un enfoque basado en palabras clave, el modelo aún se apoya de forma significativa en la presencia de términos específicos más que en una comprensión completa del contexto conversacional — consistente con la limitación de capacidad representacional señalada en la sección de "Justificación del Modelo".

---

## Conclusiones

Este proyecto permitió entrenar y evaluar un modelo Transformer preentrenado (DistilBERT) para la detección de discurso de odio en publicaciones de Twitter, utilizando exclusivamente el contenido textual de cada tweet.

**Sobre la decisión del esquema de etiquetado:** el análisis exploratorio mostró que el esquema multiclase original (6 categorías) presentaba un desbalance severo, con la categoría `Religion` representando apenas el 0.13% del dataset. Esto llevó a optar por un esquema binario (Hate/NotHate), decisión que resultó adecuada para completar un ciclo de entrenamiento y evaluación completo en el plazo disponible.

**Sobre el desempeño del modelo:** un F1-score de 0.674 sobre un test set balanceado indica que el modelo aprendió patrones lingüísticos reales asociados al discurso de odio, superando claramente un clasificador aleatorio (50%), aunque sin alcanzar un desempeño sobresaliente. El análisis de errores reveló que buena parte de las equivocaciones del modelo ocurren en casos genuinamente ambiguos — donde incluso anotadores humanos podrían discrepar — más que en errores evidentes, lo cual sugiere que el techo de desempeño en esta tarea específica, usando solo texto, tiene limitaciones no triviales.

**Sobre el control de overfitting:** el uso de `early stopping` fue determinante: sin él, el modelo habría completado las 4 épocas planificadas entrenando sobre una versión sobreajustada y con peor capacidad de generalización que la obtenida en la época 1.

**Limitaciones y trabajo futuro:** como se documentó desde el diseño del proyecto, el modelo no tiene acceso al contenido visual de los memes asociados a cada tweet. El análisis de errores sugiere que incorporar el texto OCR (`img_txt`) del meme como información adicional podría ayudar a resolver casos ambiguos donde el texto del tweet por sí solo es insuficiente para determinar la intención del mensaje. Explorar modelos de mayor capacidad (BERT Base, RoBERTa) o técnicas de data augmentation para las clases minoritarias del esquema multiclase también quedan como líneas de trabajo futuro.