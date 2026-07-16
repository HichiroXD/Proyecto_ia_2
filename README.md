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

Fuente:
https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech

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

Ambos esquemas se calcularán y compararán en el EDA (porcentaje de muestras descartadas, balance de clases resultante en cada caso, etc). La decisión final sobre qué esquema utilizar para el entrenamiento del modelo **queda pendiente** hasta contar con estos resultados, y será documentada y justificada en esta sección una vez definida.

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
│       └── dataset.csv
│
├── notebooks/
│   ├── 01_build_dataset.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_train.ipynb
│   └── 04_evaluate.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
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
- [ ] Construcción del dataset.
- [ ] EDA.
- [ ] Entrenamiento.
- [ ] Evaluación.
- [ ] Conclusiones.

---

## Resultados Obtenidos

> Esta sección será completada una vez finalizado el entrenamiento del
> modelo.

---

## Conclusiones

> Esta sección será completada una vez finalizado el análisis de
> resultados.
