# Creación de modelos de ML para predecir secuencias
El Capítulo 9 introdujo los datos secuenciales y los atributos de una serie temporal, como la estacionalidad, la tendencia, la autocorrelación y el ruido. Creaste una serie sintética para realizar predicciones y exploraste cómo hacer pronósticos estadísticos básicos. En los próximos capítulos, aprenderás a usar Machine Learning (ML) para realizar pronósticos. Pero antes de comenzar a crear modelos, necesitas entender cómo estructurar los datos de series temporales para entrenar modelos predictivos, creando lo que llamaremos un dataset con ventanas.

Para entender por qué necesitas hacer esto, considera la serie temporal que creaste en el Capítulo 9. Puedes ver su gráfico en la Figura 10-1.

![Figura 10-1. Serie temporal sintética](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.1.png)

Si en algún momento quieres predecir un valor en el tiempo t, querrás predecirlo como una función de los valores anteriores al tiempo t. Por ejemplo, supongamos que quieres predecir el valor de la serie temporal en el paso de tiempo 1,200 como una función de los 30 valores anteriores. En este caso, los valores desde los pasos de tiempo 1,170 a 1,199 determinarían el valor en el paso 1,200, como se muestra en la Figura 10-2.

![Figura 10-2. Valores previos que impactan en la predicción](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.2.png)

Ahora esto comienza a parecer familiar: puedes considerar los valores de 1,170–1,199 como tus características (features) y el valor en 1,200 como tu etiqueta (label). Si puedes estructurar tu dataset para tener un número determinado de valores como características y el siguiente como la etiqueta, y haces esto para cada valor conocido en el dataset, obtendrás un conjunto bastante decente de características y etiquetas que puedes usar para entrenar un modelo.

Antes de hacer esto con el dataset de series temporales del Capítulo 9, crearemos un dataset muy simple con los mismos atributos, pero con una cantidad mucho menor de datos.

## Creación de un dataset con ventanas
Las bibliotecas tf.data contienen muchas APIs útiles para manipular datos. Puedes usarlas para crear un dataset básico que contenga los números del 0 al 9, emulando una serie temporal. Luego, transformarás este dataset en los inicios de un dataset con ventanas. Aquí está el código:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())
```

Primero, se crea el dataset usando range, que simplemente hace que el dataset contenga los valores del 0 al n−1, donde n es, en este caso, 10.

Luego, al llamar a dataset.window y pasar un parámetro de 5, se especifica dividir el dataset en ventanas de cinco elementos. Especificar shift=1 hace que cada ventana se desplace un lugar con respecto a la anterior: la primera ventana contendrá los cinco elementos comenzando en 0, la siguiente los cinco elementos comenzando en 1, y así sucesivamente. Configurar drop_remainder=True especifica que, una vez que se alcance un punto cercano al final del dataset donde las ventanas sean más pequeñas que el tamaño deseado, se descarten.

Dada la definición de ventanas, el proceso de división del dataset puede llevarse a cabo. Esto se hace con la función flat_map, que en este caso solicita un lote (batch) de cinco ventanas. Al ejecutar este código obtendrás el siguiente resultado:

```python
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
```

Pero como viste antes, queremos convertir esto en datos de entrenamiento, donde haya n valores como características y un valor subsiguiente como etiqueta. Puedes hacer esto agregando otra función lambda que divida cada ventana en todo antes del último valor (características) y luego el último valor (etiqueta). Esto crea un dataset de x y y, como se muestra aquí:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
```

Los resultados ahora están en línea con lo que esperabas. Los primeros cuatro valores en la ventana se consideran las características, y el valor subsiguiente es la etiqueta:

```python
[0 1 2 3] [4]
[1 2 3 4] [5]
[2 3 4 5] [6]
[3 4 5 6] [7]
[4 5 6 7] [8]
[5 6 7 8] [9]
```

Dado que este es un dataset, también puede admitir mezcla (shuffling) y batches mediante funciones lambda. Aquí se ha mezclado y agrupado con un tamaño de lote de 2:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
```

Los resultados muestran que el primer lote tiene dos conjuntos de x (comenzando en 2 y 3, respectivamente) con sus etiquetas, el segundo lote tiene dos conjuntos de x (comenzando en 1 y 5, respectivamente) con sus etiquetas, y así sucesivamente:

```python
x =  [[2 3 4 5]
      [3 4 5 6]]
y =  [[6]
      [7]]
x =  [[1 2 3 4]
      [5 6 7 8]]
y =  [[5]
      [9]]
x =  [[0 1 2 3]
      [4 5 6 7]]
y =  [[4]
      [8]]
```

Con esta técnica, ahora puedes convertir cualquier dataset de series temporales en un conjunto de datos de entrenamiento para una red neuronal. En la siguiente sección, explorarás cómo tomar los datos sintéticos del Capítulo 9 y crear un conjunto de entrenamiento a partir de ellos. A partir de ahí, avanzarás hacia la creación de una DNN simple entrenada con estos datos que pueda predecir valores futuros.

### Creación de una versión con ventanas del dataset de series temporales
Como recordatorio, aquí está el código usado en el capítulo anterior para crear un dataset de series temporales sintético:

```python
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5
series = baseline + trend(time, slope) 
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)
```

Esto creará una serie temporal que se verá como en la Figura 10-1. Si quieres cambiarla, puedes ajustar los valores de las constantes.

Una vez que tengas la serie, puedes convertirla en un dataset con ventanas con un código similar al de la sección anterior. Aquí está definido como una función independiente:

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```

Ten en cuenta que usa el método from_tensor_slices de tf.data.Dataset, que te permite convertir una serie en un dataset. Puedes aprender más sobre este método en la documentación de TensorFlow.

Ahora, para obtener un dataset listo para entrenamiento, puedes usar simplemente el siguiente código. Primero divides la serie en datasets de entrenamiento y validación, luego especificas detalles como el tamaño de la ventana, el tamaño del lote y el tamaño del búfer de mezcla:

```python
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
```

Lo importante a recordar ahora es que tus datos son un tf.data.Dataset, por lo que pueden pasarse fácilmente a model.fit como un único parámetro, y tf.keras se encargará del resto.

Si deseas inspeccionar cómo se ven los datos, puedes hacerlo con un código como este:

```python
dataset = windowed_dataset(series, window_size, 1, shuffle_buffer_size)
for feature, label in dataset.take(1):
    print(feature)
    print(label)
```

Aquí el batch_size se establece en 1 para que los resultados sean más legibles. Obtendrás una salida como esta, donde un solo conjunto de datos está en el lote:

```python
tf.Tensor(
 [[75.38214  66.902626 76.656364 71.96795  71.373764 76.881065 75.62607
   71.67851  79.358665 68.235466 76.79933  76.764114 72.32991  75.58744
   67.780426 78.73544  73.270195 71.66057  79.59881  70.9117  ]], 
 shape=(1, 20), dtype=float32)
tf.Tensor([67.47085], shape=(1,), dtype=float32)
```

El primer lote de números son las características. Configuramos el tamaño de la ventana en 20, por lo que es un tensor de 1×20. El segundo número es la etiqueta (67.47085 en este caso), que el modelo intentará ajustar a las características. Verás cómo funciona en la siguiente sección.

## Creación y entrenamiento de una DNN para ajustar los datos secuenciales
Ahora que tienes los datos en un tf.data.Dataset, crear un modelo de red neuronal en tf.keras se vuelve muy sencillo. Primero exploraremos una DNN simple que luce así:

```python
dataset = windowed_dataset(series, window_size, 
                           batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])
```

Es un modelo súper simple con dos capas densas. La primera acepta la forma de entrada de window_size antes de una capa de salida que contendrá el valor predicho.

El modelo se compila con una función de pérdida y un optimizador, como antes. En este caso, la función de pérdida especificada es mse, que significa mean squared error y se utiliza comúnmente en problemas de regresión (que es en lo que esto finalmente se reduce). Para el optimizador, sgd (stochastic gradient descent) es una buena opción. No entraremos en detalles sobre estos tipos de funciones en este libro, pero cualquier buen recurso sobre aprendizaje automático, como la especialización en Deep Learning de Andrew Ng en Coursera, te enseñará sobre ellas.

SGD acepta parámetros como la tasa de aprendizaje (lr) y el momento (momentum), los cuales ajustan cómo aprende el optimizador. Cada conjunto de datos es diferente, por lo que es bueno tener control sobre estos parámetros. En la siguiente sección, verás cómo puedes encontrar los valores óptimos, pero por ahora, configúralos así:

```python
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
    lr=1e-6, 
    momentum=0.9))
```

Entrenar el modelo se reduce a simplemente llamar a model.fit, pasando tu dataset y especificando el número de épocas para entrenar:

```python
model.fit(dataset, epochs=100, verbose=1)
```

A medida que entrenas, verás que la función de pérdida reporta un número que comienza alto pero disminuye constantemente. Aquí están los resultados de las primeras 10 épocas:

```python
Epoch 1/100
45/45 [==============================] - 1s 15ms/step - loss: 898.6162
Epoch 2/100
45/45 [==============================] - 0s 8ms/step - loss: 52.9352
Epoch 3/100
45/45 [==============================] - 0s 8ms/step - loss: 49.9154
Epoch 4/100
45/45 [==============================] - 0s 7ms/step - loss: 49.8471
Epoch 5/100
45/45 [==============================] - 0s 7ms/step - loss: 48.9934
Epoch 6/100
45/45 [==============================] - 0s 7ms/step - loss: 49.7624
Epoch 7/100
45/45 [==============================] - 0s 8ms/step - loss: 48.3613
Epoch 8/100
45/45 [==============================] - 0s 9ms/step - loss: 49.8874
Epoch 9/100
45/45 [==============================] - 0s 8ms/step - loss: 47.1426
Epoch 10/100
45/45 [==============================] - 0s 8ms/step - loss: 47.5133
```

## Evaluación de los resultados de la DNN
Una vez que tengas una DNN entrenada, puedes comenzar a predecir con ella. Pero recuerda que tienes un dataset con ventanas, por lo que la predicción para un punto dado se basa en los valores de un cierto número de pasos de tiempo anteriores.

En otras palabras, como tus datos están en una lista llamada series, para predecir un valor, debes pasar al modelo los valores desde el tiempo t hasta t + window_size. Luego, obtendrás el valor predicho para el siguiente paso de tiempo.

Por ejemplo, si quisieras predecir el valor en el paso de tiempo 1,020, tomarías los valores desde los pasos de tiempo 1,000 a 1,019 y los usarías para predecir el siguiente valor en la secuencia. Para obtener esos valores, utiliza el siguiente código (nota que especificas esto como series[1000:1020], no series[1000:1019]):

```python
print(series[1000:1020])
```

Luego, para obtener el valor en el paso 1,020, simplemente usa series[1020] así:

```python
print(series[1020])
```

Para obtener la predicción para ese punto de datos, pasas la serie al método model.predict. Sin embargo, para mantener la forma de entrada consistente, necesitarás un np.newaxis, así:

```python
print(model.predict(series[1000:1020][np.newaxis]))
```

O, si deseas un código más genérico, puedes usar esto:

```python
print(series[start_point:start_point+window_size])
print(series[start_point+window_size])
print(model.predict(
    series[start_point:start_point+window_size][np.newaxis]))
```

Ten en cuenta que todo esto asume un tamaño de ventana de 20 puntos de datos, que es bastante pequeño. Como resultado, tu modelo puede carecer de algo de precisión. Si deseas probar con un tamaño de ventana diferente, deberás reformatear el dataset llamando nuevamente a la función windowed_dataset y luego reentrenar el modelo.

Aquí está la salida para este dataset al tomar un punto de inicio de 1,000 y predecir el siguiente valor:

```python
[109.170746 106.86935  102.61668   99.15634  105.95478  104.503876
 107.08533  105.858284 108.00339  100.15279  109.4894   103.96404
 113.426094  99.67773  111.87749  104.26137  100.08899  101.00105
 101.893265 105.69048 ]
106.258606
[[105.36248]]
```

El primer tensor contiene la lista de valores. Luego, vemos el siguiente valor real, que es 106.258606. Finalmente, vemos el siguiente valor predicho, 105.36248. Estamos obteniendo una predicción razonable, pero ¿cómo medimos la precisión a lo largo del tiempo? Lo exploraremos en la siguiente sección.

## Explorando la predicción general
En la sección anterior, viste cómo obtener una predicción para un punto de tiempo dado al tomar el conjunto anterior de valores basado en el tamaño de la ventana (en este caso, 20) y pasarlos al modelo. Para ver los resultados generales del modelo, deberás hacer lo mismo para cada paso de tiempo.

Puedes hacerlo con un simple bucle como este:

```python
forecast = []
for time in range(len(series) - window_size):
    forecast.append(
        model.predict(series[time:time + window_size][np.newaxis]))
```

Primero, creas un nuevo array llamado forecast que almacenará los valores predichos. Luego, para cada paso de tiempo en la serie original, llamas al método predict y almacenas los resultados en el array forecast.

No puedes hacer esto para los primeros n elementos en los datos, donde n es el tamaño de la ventana, porque en ese punto no tendrás suficientes datos para hacer una predicción, ya que cada predicción requiere n valores anteriores.

Cuando este bucle termina, el array forecast tendrá los valores de las predicciones para el paso de tiempo 21 en adelante.

Si recuerdas, también dividiste el dataset en conjuntos de entrenamiento y validación en el paso de tiempo 1,000. Por lo tanto, en el siguiente paso también deberías tomar solo las predicciones desde este punto en adelante. Como tus datos de predicción ya están desplazados en 20 (o cualquier tamaño de ventana que hayas usado), puedes dividirlos y convertirlos en un array de Numpy así:

```python
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
```

Ahora tiene la misma forma que los datos de predicción, por lo que puedes graficarlos entre sí así:

```python
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```

El gráfico se verá como en la Figura 10-3.

![Figura 10-3. Graficando predicciones contra valores reales](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.3.png)

A simple vista, puedes ver que la predicción no está mal. Generalmente sigue la curva de los datos originales. Cuando hay cambios rápidos en los datos, la predicción tarda un poco en ponerse al día, pero en general no está mal.

Sin embargo, es difícil ser preciso al inspeccionar visualmente la curva. Es mejor tener una buena métrica, y en el Capítulo 9 aprendiste sobre una: el MAE. Ahora que tienes los datos de validación y los resultados, puedes medir el MAE con este código:

```python
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```

Se introdujo aleatoriedad en los datos, por lo que tus resultados pueden variar, pero cuando lo intenté obtuve un valor de 4.51 como el MAE.

Podrías argumentar que el proceso de hacer las predicciones lo más precisas posible se convierte entonces en el proceso de minimizar ese MAE. Hay algunas técnicas que puedes usar para hacerlo, incluyendo cambiar el tamaño de la ventana. Te dejaré experimentar con eso, pero en la siguiente sección realizarás una sintonización básica de hiperparámetros en el optimizador para mejorar cómo aprende tu red neuronal y verás el impacto que eso tendrá en el MAE.

## Ajustando la tasa de aprendizaje
En el ejemplo anterior, quizá recuerdes que compilaste el modelo con un optimizador que lucía así:

```python
model.compile(loss="mse",
              optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
```

En este caso, usaste una tasa de aprendizaje de 1×10 −6. Pero ese parecía un número bastante arbitrario. ¿Qué pasaría si lo cambiaras? ¿Y cómo deberías hacerlo? Llevaría muchas pruebas encontrar la mejor tasa.

Una de las herramientas que ofrece tf.keras es un callback que te ayuda a ajustar la tasa de aprendizaje con el tiempo. Aprendiste sobre los callbacks (funciones que se llaman al final de cada época) en el Capítulo 2, donde usaste uno para detener el entrenamiento cuando la precisión alcanzaba un valor deseado.

También puedes usar un callback para ajustar el parámetro de la tasa de aprendizaje, graficar el valor de ese parámetro frente a la pérdida para la época correspondiente y, a partir de ahí, determinar la mejor tasa de aprendizaje a usar.

Para hacerlo, simplemente crea un tf.keras.callbacks.LearningRateScheduler y haz que llene el parámetro lr con el valor inicial deseado. Aquí tienes un ejemplo:

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
```

En este caso, comenzarás la tasa de aprendizaje en 1×10−8, y luego, en cada época, la incrementarás por una pequeña cantidad. Para cuando haya completado 100 épocas, la tasa de aprendizaje estará cerca de 1×10−3.

Ahora puedes inicializar el optimizador con la tasa de aprendizaje de 1e−8 y especificar que deseas usar este callback dentro de la llamada a model.fit:

```python
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, 
                    callbacks=[lr_schedule], verbose=0)
```

Al usar history=model.fit, el historial de entrenamiento se almacena para ti, incluida la pérdida. Luego puedes graficar esto contra la tasa de aprendizaje por época así:

```python
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```

Esto simplemente configura el valor de lrs usando la misma fórmula que la función lambda y lo grafica contra la pérdida entre 1×10−8 y 1×10−3. La Figura 10-4 muestra el resultado.

![Figura 10-4. Graficando la pérdida frente a la tasa de aprendizaje](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.4.png)

Aunque anteriormente configuraste la tasa de aprendizaje en 1e−6, parece que 1e−5 tiene una pérdida menor. Ahora puedes volver al modelo y redefinirlo con 1e−5 como la nueva tasa de aprendizaje.

Después de entrenar el modelo, probablemente notarás que la pérdida se ha reducido un poco. En mi caso, con una tasa de aprendizaje de 1e−6, mi pérdida final fue 36.5, pero con 1e−5 se redujo a 32.9. Sin embargo, cuando realicé predicciones para todos los datos, el resultado fue el gráfico en la Figura 10-5, que, como puedes ver, parece un poco desviado.

![Figura 10-5. Gráfico con la tasa de aprendizaje ajustada](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.5.png)

Cuando medí el MAE, resultó ser 4.96, así que retrocedió un poco. Dicho esto, una vez que sepas que tienes la mejor tasa de aprendizaje, puedes comenzar a explorar otras metodologías para optimizar el rendimiento de la red. Un buen punto de partida es el tamaño de la ventana: usar datos de 20 días para predecir 1 día puede no ser suficiente, por lo que podrías probar con una ventana de 40 días. También intenta entrenar por más épocas. Con algo de experimentación, podrías obtener un MAE cercano a 4, lo cual no está mal.

## Exploración de la optimización de hiperparámetros con Keras Tuner
En la sección anterior, exploraste cómo realizar una optimización aproximada de la tasa de aprendizaje para la función de pérdida del descenso de gradiente estocástico. Fue, ciertamente, un esfuerzo muy rudimentario, modificando la tasa de aprendizaje cada pocos epochs y midiendo la pérdida. Además, estuvo algo condicionado por el hecho de que la función de pérdida ya estaba cambiando de epoch en epoch, por lo que quizá no encontraste el mejor valor exacto, sino una aproximación. Para realmente encontrar el mejor valor, tendrías que entrenar durante todos los epochs con cada valor potencial y luego comparar los resultados. Y eso es solo para un hiperparámetro: la tasa de aprendizaje. Si quisieras encontrar el mejor momento (momentum), o ajustar otros aspectos como la arquitectura del modelo (cantidad de neuronas por capa, número de capas, etc.), podrías terminar con miles de opciones para probar, y programar todo este entrenamiento sería complicado.

Afortunadamente, la herramienta Keras Tuner hace que este proceso sea relativamente sencillo. Puedes instalar Keras Tuner con un simple comando de pip:

```python
!pip install keras-tuner
```

Luego puedes usarlo para parametrizar tus hiperparámetros, especificando rangos de valores para probar. Keras Tuner entrenará múltiples modelos, uno para cada posible conjunto de parámetros, evaluará cada modelo según una métrica que determines y luego te informará sobre los mejores modelos. No entraremos aquí en todas las opciones que ofrece la herramienta, pero te mostraré cómo puedes usarla para este modelo específico.

Supongamos que queremos experimentar con solo dos cosas, la primera siendo el número de neuronas de entrada en la arquitectura del modelo. Hasta ahora, has tenido una arquitectura con 10 neuronas en la capa de entrada, seguida de una capa oculta de 10 neuronas, y la capa de salida. Pero, ¿podría la red hacerlo mejor con más neuronas? ¿Qué pasaría si experimentaras con hasta 30 neuronas en la capa de entrada, por ejemplo?

Recuerda que la capa de entrada estaba definida así:

```python
tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
```

Si quieres probar diferentes valores en lugar del valor fijo de 10, puedes configurarlo para recorrer una serie de valores enteros de esta forma:

```python
tf.keras.layers.Dense(
    units=hp.Int('units', min_value=10, max_value=30, step=2), 
    activation='relu', 
    input_shape=[window_size]
)
```

Aquí defines que la capa se probará con varios valores de entrada, comenzando en 10 y aumentando hasta 30 en pasos de 2. Ahora, en lugar de entrenar el modelo solo una vez y observar la pérdida, Keras Tuner entrenará el modelo 11 veces.

Además, cuando compilaste el modelo, configuraste el valor del hiperparámetro momentum en 0.9 de forma fija. Recuerda este código de la definición del modelo:

```python
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
```

Puedes cambiarlo para recorrer algunas opciones utilizando la función hp.Choice. Aquí tienes un ejemplo:

```python
optimizer = tf.keras.optimizers.SGD(
    hp.Choice('momentum', values=[.9, .7, .5, .3]), 
    lr=1e-5
)
```

Esto proporciona cuatro posibles opciones. Combinando esto con la arquitectura del modelo definida previamente, terminarás probando 44 combinaciones posibles. Keras Tuner puede hacer esto por ti y reportarte cuál modelo tuvo el mejor desempeño.

Para configurar todo esto, primero crea una función que construya el modelo por ti. Aquí tienes una definición actualizada del modelo:

```python
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=10, max_value=30, step=2), 
        activation='relu', 
        input_shape=[window_size]
    ))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        loss="mse", 
        optimizer=tf.keras.optimizers.SGD(
            hp.Choice('momentum', values=[.9, .7, .5, .3]), 
            lr=1e-5
        )
    )
    return model
```

Con Keras Tuner instalado, puedes crear un objeto RandomSearch que gestione todas las iteraciones para este modelo:

```python
tuner = RandomSearch(
    build_model, 
    objective='loss', 
    max_trials=150, 
    executions_per_trial=3, 
    directory='my_dir', 
    project_name='hello'
)
```

Nota que defines el modelo pasándole la función que describiste anteriormente. El parámetro hp se usa para controlar qué valores se modifican. Especificas que el objetivo es minimizar la pérdida (loss). Puedes limitar el número total de pruebas con el parámetro max_trials y especificar cuántas veces entrenar y evaluar cada modelo con executions_per_trial.

Para iniciar la búsqueda, simplemente llamas a tuner.search como lo harías con model.fit. Aquí está el código:

```python
tuner.search(dataset, epochs=100, verbose=0)
```

Al finalizar, puedes llamar a tuner.results_summary para obtener los 10 mejores intentos basados en el objetivo:

```python
tuner.results_summary()
```

Verás un resumen como este:

```python
Results summary
|-Results in my_dir/hello
|-Showing 10 best trials
|-Objective(name='loss', direction='min')
Trial summary
|-Trial ID: dcfd832e62daf4d34b729c546120fb14
|-Score: 33.18723194615371
|-Best step: 0
Hyperparameters:
|-momentum: 0.5
|-units: 28
```

De los resultados, puedes ver que la mejor puntuación de pérdida se logró con un momentum de 0.5 y 28 unidades de entrada. Puedes recuperar este modelo y otros modelos principales llamando a get_best_models, especificando cuántos quieres, por ejemplo:

```python
tuner.get_best_models(num_models=4)
```

Alternativamente, puedes crear un nuevo modelo desde cero utilizando los hiperparámetros aprendidos:

```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(28, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, verbose=1)
```

Cuando entrené usando estos hiperparámetros y realicé el pronóstico para todo el conjunto de validación como antes, obtuve un gráfico que se parecía al de la Figura 10-6.

![Figura 10-6. El gráfico de predicción con hiperparámetros optimizados](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure10.6.png)

Un cálculo del MAE con esto dio como resultado 4.47, lo cual es una ligera mejora respecto al original de 4.51 y una gran mejora frente al enfoque estadístico del capítulo anterior que dio un resultado de 5.13. Esto se hizo con la tasa de aprendizaje ajustada a 1×10−5, que puede no haber sido óptima. Usando Keras Tuner, puedes ajustar hiperparámetros como este, modificar el número de neuronas en la capa intermedia o incluso experimentar con diferentes funciones de pérdida y optimizadores. ¡Pruébalo y ve si puedes mejorar este modelo!

## Resumen
En este capítulo, tomaste el análisis estadístico de las series temporales del Capítulo 9 y aplicaste aprendizaje automático para intentar hacer un mejor trabajo de predicción. El aprendizaje automático realmente se trata de encontrar patrones, y, como se esperaba, lograste reducir el error absoluto medio casi un 10%, primero usando una red neuronal profunda para identificar patrones y luego ajustando hiperparámetros con Keras Tuner para mejorar la pérdida y aumentar la precisión. En el Capítulo 11, irás más allá de una simple red neuronal profunda (DNN) y examinarás las implicaciones de usar una red neuronal recurrente para predecir valores secuenciales.
