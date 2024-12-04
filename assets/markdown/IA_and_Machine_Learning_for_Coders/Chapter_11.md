# Usando Métodos Convolucionales y Recurrentes para Modelos de Secuencias
Los últimos capítulos te introdujeron a los datos de secuencias. Aprendiste a predecirlos primero utilizando métodos estadísticos, luego métodos básicos de aprendizaje automático con una red neuronal profunda. También exploraste cómo ajustar los hiperparámetros del modelo utilizando Keras Tuner. En este capítulo, explorarás técnicas adicionales que pueden mejorar aún más tu capacidad para predecir datos de secuencias usando redes neuronales convolucionales, así como redes neuronales recurrentes.

## Convoluciones para Datos de Secuencias
En el Capítulo 3 se introdujeron las convoluciones, donde un filtro 2D se pasaba sobre una imagen para modificarla y, potencialmente, extraer características. Con el tiempo, la red neuronal aprendía qué valores de filtro eran efectivos para ajustar las modificaciones hechas a los píxeles a sus etiquetas, extrayendo características de la imagen. La misma técnica puede aplicarse a datos numéricos en series temporales, pero con una modificación: la convolución será unidimensional en lugar de bidimensional.

Considera, por ejemplo, la serie de números en la Figura 11-1.

![Figura 11-1. Una secuencia de números](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.1.png)

Una convolución 1D podría operar sobre estos números de la siguiente manera. Considera la convolución como un filtro 1 × 3 con valores de filtro de –0.5, 1 y –0.5, respectivamente. En este caso, el primer valor de la secuencia se perderá, y el segundo valor se transformará de 8 a –1.5, como se muestra en la Figura 11-2.

![Figura 11-2. Usando una convolución con la secuencia de números](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.2.png)

El filtro luego avanzará a través de los valores, calculando nuevos valores a medida que avanza. Así, por ejemplo, en el siguiente paso 15 se transformará en 3, como se muestra en la Figura 11-3.

![Figura 11-3. Otro paso en la convolución 1D](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.3.png)

Usando este método, es posible extraer los patrones entre los valores y aprender los filtros que los extraen exitosamente, de manera similar a como las convoluciones en los píxeles de imágenes pueden extraer características. En este caso, no hay etiquetas, pero se podrían aprender las convoluciones que minimicen la pérdida general.

### Programando Convoluciones
Antes de programar convoluciones, deberás ajustar el generador de conjuntos de datos con ventanas que utilizaste en el capítulo anterior. Esto se debe a que al programar las capas convolucionales, debes especificar la dimensionalidad. El conjunto de datos con ventanas era unidimensional, pero no estaba definido como un tensor 1D. Esto simplemente requiere agregar una declaración tf.expand_dims al inicio de la función windowed_dataset, como esta:

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```

Ahora que tienes un conjunto de datos ajustado, puedes agregar una capa convolucional antes de las capas densas que usaste previamente:

```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.Dense(28, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, verbose=1)
```

En la capa Conv1D, tienes varios parámetros:

- filters: Es el número de filtros que quieres que la capa aprenda. Generará este número y los ajustará con el tiempo para que se adapten a tus datos conforme aprende.
- kernel_size: Es el tamaño del filtro; anteriormente demostramos un filtro con los valores –0.5, 1, –0.5, que tendría un tamaño de núcleo de 3.
- strides: Es el tamaño del “paso” que el filtro tomará mientras escanea la lista. Típicamente es 1.
- padding: Determina el comportamiento de la lista en relación a qué extremo se eliminan datos. Un filtro 3 × 1 “perderá” el primer y último valor de la lista porque no puede calcular el valor anterior para el primero ni el valor posterior para el último. Típicamente con datos de secuencia usarás causal, que solo tomará datos de los pasos de tiempo actuales y anteriores, nunca futuros.
- activation: Es la función de activación. En este caso, relu significa rechazar efectivamente valores negativos que salen de la capa.
- input_shape: Como siempre, es la forma de entrada de los datos que se pasan a la red. Dado que esta es la primera capa, debes especificarla.

Entrenando con esto, obtendrás un modelo como antes, pero para obtener predicciones del modelo, dado que la capa de entrada ha cambiado de forma, deberás modificar tu código de predicción.

Además, en lugar de predecir cada valor uno por uno, basándote en la ventana anterior, puedes obtener una única predicción para toda una serie si has formateado correctamente la serie como un conjunto de datos. Para simplificar un poco las cosas, aquí tienes una función auxiliar que puede predecir toda una serie basada en el modelo, con un tamaño de ventana especificado:

```python
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
```

Si deseas usar el modelo para predecir esta serie, simplemente pasa la serie con un nuevo eje para manejar las Conv1D necesarias para la capa con el eje extra. Puedes hacerlo así:

```python
forecast = model_forecast(model, series[..., np.newaxis], window_size)
```

Y puedes dividir esta predicción en solo las predicciones para el conjunto de validación usando la división de tiempo predeterminada:

```python
results = forecast[split_time - window_size:-1, -1, 0]
```

Un gráfico de los resultados contra la serie se muestra en la Figura 11-4.

El MAE en este caso es 4.89, que es ligeramente peor que para la predicción anterior. Esto podría deberse a que no hemos ajustado adecuadamente la capa convolucional, o podría ser que las convoluciones simplemente no ayudan. Este es el tipo de experimentación que necesitarás hacer con tus datos.

Toma en cuenta que estos datos tienen un elemento aleatorio, por lo que los valores cambiarán entre sesiones. Si estás utilizando el código del Capítulo 10 y luego ejecutas este código por separado, obviamente tendrás fluctuaciones aleatorias que afectarán tus datos y, por lo tanto, tu MAE.

![Figura 11-4. Red neuronal convolucional con predicción de datos de secuencia temporal](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.4.png)

Cuando usas convoluciones, siempre surge la pregunta: ¿Por qué elegir los parámetros que elegimos? ¿Por qué 128 filtros? ¿Por qué tamaño 3 × 1? La buena noticia es que puedes experimentar con ellos utilizando Keras Tuner, como se mostró anteriormente. Exploraremos eso a continuación.

### Experimentando con los Hiperparámetros de Conv1D
En la sección anterior, viste una convolución 1D codificada con parámetros fijos como el número de filtros, el tamaño del kernel, el número de pasos (strides), etc. Al entrenar la red neuronal con estos parámetros, el MAE aumentó ligeramente, por lo que no obtuvimos beneficios al usar la capa Conv1D. Esto puede no ser siempre el caso, dependiendo de tus datos, pero podría deberse a hiperparámetros subóptimos. En esta sección, aprenderás cómo Keras Tuner puede optimizarlos por ti.

En este ejemplo, experimentarás con los hiperparámetros para el número de filtros, el tamaño del kernel y el tamaño del paso, manteniendo los demás parámetros estáticos:

```python
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=hp.Int('units', min_value=128, max_value=256, step=64), 
        kernel_size=hp.Int('kernels', min_value=3, max_value=9, step=3),
        strides=hp.Int('strides', min_value=1, max_value=3, step=1),
        padding='causal', activation='relu', input_shape=[None, 1]
    ))  
    model.add(tf.keras.layers.Dense(28, input_shape=[window_size], activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(momentum=0.5, lr=1e-5))
    return model
```

Los valores de filtro comenzarán en 128 y luego aumentarán hasta 256 en incrementos de 64. El tamaño del kernel comenzará en 3 y aumentará hasta 9 en pasos de 3, y los pasos comenzarán en 1 y aumentarán hasta 3.

Hay muchas combinaciones de valores aquí, por lo que el experimento tomará algo de tiempo para ejecutarse. También podrías probar otros cambios, como usar un valor inicial mucho más pequeño para los filtros, para ver su impacto.

Aquí tienes el código para realizar la búsqueda:

```python
tuner = RandomSearch(
    build_model, 
    objective='loss', 
    max_trials=500, 
    executions_per_trial=3, 
    directory='my_dir', 
    project_name='cnn-tune'
)
tuner.search_space_summary()
tuner.search(dataset, epochs=100, verbose=2)
```

Cuando ejecuté el experimento, descubrí que 128 filtros, con un tamaño de kernel de 9 y un paso de 1, dieron los mejores resultados. Entonces, en comparación con el modelo inicial, la gran diferencia fue cambiar el tamaño del kernel, lo que tiene sentido con un conjunto de datos tan grande. Con un tamaño de kernel de 3, solo los vecinos inmediatos tienen un impacto, mientras que con un tamaño de 9, los vecinos más lejanos también afectan el resultado al aplicar el filtro. Esto justificaría un experimento adicional, comenzando con estos valores y probando tamaños de filtro más grandes y quizás menos filtros. ¡Te dejo eso a ti para ver si puedes mejorar aún más el modelo!

Insertando estos valores en la arquitectura del modelo, obtendrás esto:

```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=9,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.Dense(28, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1),
])
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100,  verbose=1)
```

Después de entrenar con esto, el modelo tuvo una precisión mejorada en comparación con la CNN ingenua creada anteriormente y con la red neuronal densa original, como se muestra en la Figura 11-5.

![Figura 11-5. Predicciones optimizadas de la CNN](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.5.png)

Esto resultó en un MAE de 4.39, que es una ligera mejora respecto al 4.47 que obtuvimos sin usar la capa convolucional. Más experimentación con los hiperparámetros de la CNN podría mejorar aún más este resultado.

Más allá de las convoluciones, las técnicas exploradas en los capítulos sobre procesamiento de lenguaje natural con RNNs, incluyendo LSTMs, pueden ser poderosas al trabajar con datos de secuencia. Por su propia naturaleza, las RNN están diseñadas para mantener el contexto, por lo que los valores previos pueden afectar a los posteriores. Explorarás cómo usarlas para modelar secuencias a continuación. Pero primero, pasemos de un conjunto de datos sintético y comencemos a analizar datos reales. En este caso, consideraremos datos meteorológicos.

## Usando Datos Meteorológicos de la NASA
Un gran recurso para datos meteorológicos en series temporales es el análisis de temperatura de superficie del Instituto Goddard de Estudios Espaciales (GISS) de la NASA. Si sigues el enlace Station Data, en el lado derecho de la página puedes elegir una estación meteorológica para obtener datos. Por ejemplo, seleccioné el aeropuerto de Seattle Tacoma (SeaTac) y fui llevado a la página de la Figura 11-6.

![Figura 11-6. Datos de temperatura de superficie del GISS](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.6.png)

En la parte inferior de esta página, puedes ver un enlace para descargar datos mensuales como CSV. Selecciona esto, y se descargará un archivo llamado station.csv a tu dispositivo. Si lo abres, verás que es una cuadrícula de datos con un año en cada fila y un mes en cada columna, como en la Figura 11-7.

![Figura 11-7. Explorando los datos](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.7.png)

Como estos datos están en formato CSV, es bastante fácil procesarlos en Python, pero como con cualquier conjunto de datos, debes prestar atención al formato. Al leer un CSV, generalmente se lee línea por línea, y a menudo cada línea tiene un punto de datos de interés. En este caso, hay al menos 12 puntos de datos de interés por línea, por lo que deberás considerar esto al leer los datos.

### Leyendo Datos GISS en Python
El código para leer los datos GISS se muestra aquí:

```python
def get_data():
    data_file = "/home/ljpm/Desktop/bookpython/station.csv"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures = []
    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series
```

Esto abrirá el archivo en la ruta indicada (la tuya obviamente será diferente) y leerá el archivo completo como un conjunto de líneas, donde el separador de línea es el carácter de nueva línea (\n). Luego recorrerá cada línea, ignorando la primera, y las dividirá usando la coma como separador en un nuevo arreglo llamado linedata. Los elementos del 1 al 13 en este arreglo indicarán los valores para los meses de enero a diciembre como cadenas. Estos valores se convierten a flotantes y se añaden al arreglo llamado temperatures. Una vez completado, se convertirá en un arreglo de Numpy llamado series, y se creará otro arreglo de Numpy llamado time del mismo tamaño que series. Como se crea usando np.arange, el primer elemento será 1, el segundo 2, etc. Por lo tanto, esta función devolverá el tiempo en pasos del 1 al número de puntos de datos, y la serie como los datos para ese tiempo.

Ahora, si deseas una serie temporal normalizada, simplemente puedes ejecutar este código:

```python
time, series = get_data()
mean = series.mean(axis=0)
series -= mean
std = series.std(axis=0)
series /= std
```

Esto puede dividirse en conjuntos de entrenamiento y validación como antes. Elige tu tiempo de división basado en el tamaño de los datos; en este caso, tenía aproximadamente 840 elementos, por lo que dividí en 792 (reservando cuatro años de datos para validación):

```python
split_time = 792
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

Debido a que los datos ahora son un arreglo de Numpy, puedes usar el mismo código que antes para crear un conjunto de datos con ventanas para entrenar una red neuronal:

```python
window_size = 24
batch_size = 12
shuffle_buffer_size = 48
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
valid_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)
```

Esto debería usar la misma función windowed_dataset que la red convolucional mencionada anteriormente, agregando una nueva dimensión. Al usar RNNs, GRUs y LSTMs, necesitarás que los datos estén en esa forma.

## Usando RNNs para Modelado de Secuencias
Ahora que tienes los datos del CSV de la NASA en un conjunto de datos con ventanas, es relativamente fácil crear un modelo para entrenar un predictor para estos. (¡Es un poco más difícil entrenar uno bueno!) Comencemos con un modelo simple e ingenuo usando RNNs. Aquí está el código:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1)
])
```

En este caso, se utiliza la capa SimpleRNN de Keras. Las RNNs son una clase de redes neuronales poderosas para explorar modelos de secuencias. Las viste por primera vez en el Capítulo 7 cuando explorabas el procesamiento de lenguaje natural. No entraré en detalles sobre cómo funcionan aquí, pero si estás interesado y te saltaste ese capítulo, échale un vistazo ahora. Es importante mencionar que una RNN tiene un bucle interno que itera sobre los pasos de tiempo de una secuencia mientras mantiene un estado interno de los pasos de tiempo que ha visto hasta ahora. Una SimpleRNN tiene la salida de cada paso de tiempo alimentada al siguiente paso de tiempo.

Puedes compilar y ajustar el modelo con los mismos hiperparámetros de antes, o usar Keras Tuner para ver si puedes encontrar mejores. Para simplificar, puedes usar estos ajustes:

```python
optimizer = tf.keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(dataset, epochs=100, verbose=1, validation_data=valid_dataset)
```

Incluso cien épocas son suficientes para tener una idea de cómo puede predecir valores. La Figura 11-8 muestra los resultados.

![Figura 11-8. Resultados de la SimpleRNN](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.8.png)

Como puedes ver, los resultados fueron bastante buenos. Puede estar un poco desviado en los picos, y cuando el patrón cambia inesperadamente (como en los pasos de tiempo 815 y 828), pero en general no está mal. Ahora veamos qué sucede si lo entrenamos durante 1,500 épocas (Figura 11-9).

![Figura 11-9. RNN entrenada durante 1,500 épocas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.9.png)

No hay mucha diferencia, excepto que algunos de los picos se suavizan. Si miras la historia de la pérdida tanto en el conjunto de validación como en el de entrenamiento, se ve algo como la Figura 11-10.

![Figura 11-10. Pérdida de entrenamiento y validación para la SimpleRNN](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.10.png)

Como puedes ver, hay una buena coincidencia entre la pérdida de entrenamiento y la pérdida de validación, pero a medida que aumentan las épocas, el modelo comienza a sobreajustarse al conjunto de entrenamiento. Tal vez un mejor número de épocas sería alrededor de quinientas.

Una razón para esto podría ser el hecho de que los datos, al ser datos meteorológicos mensuales, son altamente estacionales. Otra es que hay un conjunto de entrenamiento muy grande y un conjunto de validación relativamente pequeño. A continuación, exploraremos el uso de un conjunto de datos climáticos más grande.

### Explorando un Conjunto de Datos Más Grande
El KNMI Climate Explorer permite explorar datos climáticos granulares de muchas ubicaciones alrededor del mundo. Descargué un conjunto de datos que contiene lecturas diarias de temperatura del centro de Inglaterra desde 1772 hasta 2020. Estos datos están estructurados de manera diferente a los datos de GISS, con la fecha como una cadena, seguida de varios espacios, y luego la lectura.

He preparado los datos, eliminando los encabezados y los espacios extra. De esta manera, es fácil leerlos con código como este:

```python
def get_data():
    data_file = "tdaily_cet.dat.txt"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    temperatures=[]
    for line in lines:
        if line:
            linedata = line.split(' ')
            temperatures.append(float(linedata[1]))
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series
```

Este conjunto de datos tiene 90,663 puntos de datos, así que, antes de entrenar tu modelo, asegúrate de dividirlo adecuadamente. Usé un tiempo de división de 80,000, dejando 10,663 registros para validación. Además, actualiza el tamaño de la ventana, el tamaño del lote y el tamaño del buffer de aleatorización de manera apropiada. Aquí tienes un ejemplo:

```python
window_size = 60
batch_size = 120
shuffle_buffer_size = 240
```

Todo lo demás puede permanecer igual. Como puedes ver en la Figura 11-11, después de entrenar durante cien épocas, el gráfico de las predicciones contra el conjunto de validación se ve bastante bien.

![Figura 11-11. Gráfico de predicciones contra los datos reales](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.11.png)

Hay muchos datos aquí, así que vamos a acercarnos a los últimos cien días de datos (Figura 11-12).

![Figura 11-12. Resultados para los últimos cien días de datos](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.12.png)

Aunque el gráfico sigue generalmente la curva de los datos, y está obteniendo las tendencias más o menos correctas, está bastante desviado, particularmente en los extremos, por lo que hay espacio para mejorar.

También es importante recordar que normalizamos los datos, por lo que aunque nuestra pérdida y MAE puedan verse bajos, eso es porque están basados en la pérdida y MAE de los valores normalizados, que tienen una varianza mucho más baja que los reales. Así que, la Figura 11-13, mostrando una pérdida de menos de 0.1, podría inducirte a una falsa sensación de seguridad.

![Figura 11-13. Pérdida y pérdida de validación para el conjunto de datos grande](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.13.png)

Para desnormalizar los datos, puedes hacer la inversa de la normalización: primero multiplicas por la desviación estándar, y luego sumas la media. En ese punto, si lo deseas, puedes calcular el MAE real para el conjunto de predicciones como se hizo anteriormente.

## Utilizando otros métodos recurrentes
Además de la capa SimpleRNN, TensorFlow tiene otros tipos de capas recurrentes, como unidades recurrentes con compuertas (GRUs) y capas de memoria a corto y largo plazo (LSTMs), discutidas en el Capítulo 7. Al usar la arquitectura basada en TFRecord para tus datos que has utilizado a lo largo de este capítulo, se vuelve relativamente simple integrar estos tipos de RNN si deseas experimentar. 

Por ejemplo, si consideras la RNN básica que creaste anteriormente:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(100, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1)
])
```

Reemplazar esto con una GRU es tan fácil como:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(100, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.GRU(100),
    tf.keras.layers.Dense(1)
])
```

Con un LSTM, es similar:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1)
])
```

Vale la pena experimentar con estos tipos de capas, así como con diferentes hiperparámetros, funciones de pérdida y optimizadores. No existe una solución universal, por lo que lo que funcione mejor para ti en cualquier situación dependerá de tus datos y de tus requisitos para realizar predicciones con ellos.

## Using Dropout
Si encuentras sobreajuste en tus modelos, donde el MAE o la pérdida para los datos de entrenamiento es mucho mejor que para los datos de validación, puedes usar dropout. Como se discutió en el Capítulo 3 en el contexto de visión por computadora, con dropout, las neuronas vecinas se desactivan aleatoriamente (se ignoran) durante el entrenamiento para evitar un sesgo de familiaridad.

Al usar RNNs, también hay un parámetro de recurrent dropout que puedes usar. ¿Cuál es la diferencia? Recuerda que al usar RNNs típicamente tienes un valor de entrada, y la neurona calcula un valor de salida y un valor que se pasa al siguiente paso de tiempo. Dropout desactivará aleatoriamente los valores de entrada. Recurrent dropout desactivará aleatoriamente los valores recurrentes que se pasan al siguiente paso.

Por ejemplo, considera la arquitectura básica de red neuronal recurrente mostrada en la Figura 11-14.

![Figura 11-14. Red neuronal recurrente](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.14.png)

Aquí puedes ver las entradas a las capas en diferentes pasos de tiempo (x). El tiempo actual es t, y los pasos mostrados son t – 2 hasta t + 1. Las salidas relevantes en los mismos pasos de tiempo (y) también se muestran. Los valores recurrentes que se pasan entre los pasos de tiempo están indicados por líneas punteadas y etiquetados como r.

Usar dropout desactivará aleatoriamente las entradas x. Usar recurrent dropout desactivará aleatoriamente los valores recurrentes r.

Puedes aprender más sobre cómo funciona el recurrent dropout desde una perspectiva matemática más profunda en el artículo "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks" de Yarin Gal y Zoubin Ghahramani.

Una cosa a considerar al usar recurrent dropout es lo que Gal discute en su investigación sobre incertidumbre en aprendizaje profundo, donde demuestra que el mismo patrón de unidades de dropout debe aplicarse en cada paso de tiempo, y que una máscara de dropout constante similar debe aplicarse en cada paso de tiempo. Aunque dropout es típicamente aleatorio, el trabajo de Gal se incorporó en Keras, por lo que al usar tf.keras se mantiene la consistencia recomendada por su investigación.

Para agregar dropout y recurrent dropout, simplemente usas los parámetros relevantes en tus capas. Por ejemplo, agregar estos a la GRU simple de antes se vería así:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(100, input_shape=[None, 1], return_sequences=True, 
                        dropout=0.1, recurrent_dropout=0.1),
    tf.keras.layers.GRU(100, dropout=0.1, recurrent_dropout=0.1),
    tf.keras.layers.Dense(1),
])
```

Cada parámetro toma un valor entre 0 y 1, indicando la proporción de valores que se desactivarán. Un valor de 0.1 desactivará el 10% de los valores correspondientes.

Las RNNs que usan dropout a menudo tardan más en converger, así que asegúrate de entrenarlas durante más épocas para probar esto. La Figura 11-15 muestra los resultados de entrenar la GRU anterior con dropout y recurrent dropout en cada capa configurados a 0.1 durante 1,000 épocas.

![Figura 11-15. Entrenando una GRU con dropout](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.15.png)

Como puedes ver, la pérdida y el MAE disminuyeron rápidamente hasta alrededor de la época 300, después de lo cual continuaron disminuyendo, pero de manera bastante ruidosa. A menudo verás ruido como este en la pérdida al usar dropout, y es una indicación de que puedes querer ajustar la cantidad de dropout, así como los parámetros de la función de pérdida, como la tasa de aprendizaje.

Las predicciones con esta red se moldearon bastante bien, como puedes ver en la Figura 11-16, pero hay espacio para mejorar, ya que los picos de las predicciones son mucho más bajos que los picos reales.

![Figura 11-16. Predicciones usando una GRU con dropout](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.16.png)

Como has visto en este capítulo, predecir datos de secuencias temporales usando redes neuronales es una propuesta difícil, pero ajustar sus hiperparámetros (particularmente con herramientas como Keras Tuner) puede ser una forma poderosa de mejorar tu modelo y sus predicciones posteriores.

## Uso de RNN bidireccionales
Otra técnica a considerar al clasificar secuencias es usar entrenamiento bidireccional. Esto puede parecer contraintuitivo al principio, ya que podrías preguntarte cómo los valores futuros podrían impactar a los pasados. Pero recuerda que los valores de series temporales pueden contener estacionalidad, donde los valores se repiten con el tiempo, y al usar una red neuronal para hacer predicciones, lo que hacemos es una sofisticada búsqueda de patrones. Dado que los datos se repiten, una señal de cómo los datos pueden repetirse podría encontrarse en los valores futuros. Al usar entrenamiento bidireccional, podemos entrenar la red para intentar detectar patrones que van desde el tiempo t al tiempo t + x, así como del tiempo t + x al tiempo t.

Afortunadamente, programar esto es sencillo. Por ejemplo, considera la GRU de la sección anterior. Para hacerla bidireccional, simplemente envuelves cada capa GRU en una llamada a tf.keras.layers.Bidirectional. Esto efectivamente entrena dos veces en cada paso: una vez con los datos de la secuencia en su orden original y otra con ellos en orden inverso. Los resultados se combinan antes de proceder al siguiente paso. Aquí tienes un ejemplo:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(100, input_shape=[None, 1], return_sequences=True, 
                            dropout=0.1, recurrent_dropout=0.1)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(100, dropout=0.1, recurrent_dropout=0.1)),
    tf.keras.layers.Dense(1),
])
```

Un gráfico de los resultados del entrenamiento con una GRU bidireccional con dropout en la serie temporal se muestra en la Figura 11-17. Como puedes ver, aquí no hay una diferencia importante, y el MAE termina siendo similar. Sin embargo, con una serie de datos más grande, podrías notar una diferencia considerable en la precisión. Además, ajustar los parámetros de entrenamiento, particularmente window_size para capturar múltiples temporadas, puede tener un gran impacto.

![Figura 11-17. Entrenamiento con una GRU bidireccional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.17.png)

Esta red tiene un MAE (en los datos normalizados) de aproximadamente 0.48, principalmente porque no parece manejar bien los picos altos. Reentrenarla con una ventana más grande y bidireccionalidad produce mejores resultados: tiene un MAE significativamente menor de aproximadamente 0.28 (Figura 11-18).

![Figura 11-18. Resultados con ventana más grande y GRU bidireccional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure11.18.png)

Como puedes ver, puedes experimentar con diferentes arquitecturas de red y diferentes hiperparámetros para mejorar tus predicciones generales. Las elecciones ideales dependen mucho de los datos, por lo que las habilidades que has aprendido en este capítulo te ayudarán con tus conjuntos de datos específicos.

## Summary
En este capítulo, exploraste diferentes tipos de redes para construir modelos que predicen datos de series temporales. Partiste de la simple DNN del Capítulo 10, agregaste convoluciones y experimentaste con tipos de redes recurrentes como RNNs simples, GRUs y LSTMs. Viste cómo ajustar hiperparámetros y la arquitectura de la red puede mejorar la precisión de tu modelo, y practicaste trabajando con algunos conjuntos de datos del mundo real, incluyendo uno masivo con cientos de años de lecturas de temperatura. ¡Ahora estás listo para comenzar a construir redes para una variedad de conjuntos de datos, con un buen entendimiento de lo que necesitas saber para optimizarlas!
