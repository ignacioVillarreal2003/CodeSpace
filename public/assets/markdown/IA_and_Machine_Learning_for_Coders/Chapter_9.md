# Comprensión de los datos de secuencia y series temporales
Las series temporales están en todas partes. Probablemente las hayas visto en cosas como pronósticos del clima, precios de acciones y tendencias históricas como la ley de Moore (Figura 9-1). Si no estás familiarizado con la ley de Moore, predice que la cantidad de transistores en un microchip se duplicará aproximadamente cada dos años. Durante casi 50 años, ha demostrado ser un predictor preciso del futuro de la potencia y el costo de la computación.

![Figura 9-1. La ley de Moore](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.1.png)

Los datos de series temporales son un conjunto de valores espaciados a lo largo del tiempo. Cuando se grafican, el eje x suele ser de naturaleza temporal. A menudo, hay varios valores graficados en el eje temporal, como en este ejemplo, donde la cantidad de transistores es un gráfico y el valor predicho por la ley de Moore es el otro. Esto se llama una serie temporal multivariada. Si solo hay un valor—por ejemplo, el volumen de lluvia a lo largo del tiempo—se llama una serie temporal univariada.

Con la ley de Moore, las predicciones son simples porque existe una regla fija y sencilla que nos permite predecir el futuro de manera aproximada: una regla que ha perdurado durante unos 50 años.
Pero, ¿qué pasa con una serie temporal como la de la Figura 9-2?

![Figura 9-2. Una serie temporal del mundo real](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.2.png)

Aunque esta serie temporal fue creada artificialmente (verás cómo hacerlo más adelante en este capítulo), tiene todas las características de una serie temporal compleja del mundo real, como un gráfico de acciones o precipitaciones estacionales. A pesar de la aparente aleatoriedad, las series temporales tienen algunos atributos comunes que son útiles para diseñar modelos de aprendizaje automático (ML) que puedan predecirlas, como se describe en la siguiente sección.

## Atributos comunes de las series temporales
Aunque las series temporales puedan parecer aleatorias y ruidosas, a menudo tienen atributos comunes que son predecibles. En esta sección exploraremos algunos de ellos.

### Tendencia
Las series temporales suelen moverse en una dirección específica. En el caso de la ley de Moore, es fácil ver que, con el tiempo, los valores en el eje y aumentan y hay una tendencia ascendente. También hay una tendencia ascendente en la serie temporal de la Figura 9-2. Por supuesto, esto no siempre será el caso: algunas series temporales pueden ser aproximadamente constantes a lo largo del tiempo, a pesar de los cambios estacionales, y otras tienen una tendencia descendente. Por ejemplo, esto ocurre en la versión inversa de la ley de Moore, que predice el precio por transistor.

### Estacionalidad
Muchas series temporales tienen un patrón que se repite a lo largo del tiempo, con repeticiones que ocurren a intervalos regulares llamados temporadas. Considera, por ejemplo, la temperatura en el clima. Típicamente tenemos cuatro estaciones por año, con la temperatura siendo más alta en verano. Si graficaras el clima durante varios años, verías picos que ocurren cada cuatro estaciones, dándonos el concepto de estacionalidad. Pero este fenómeno no se limita al clima. Observa, por ejemplo, la Figura 9-3, que es un gráfico del tráfico a un sitio web.

![Figura 9-3. Tráfico web](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.3.png)

Está graficado semana a semana, y se pueden observar caídas regulares. ¿Puedes adivinar a qué se deben? En este caso, el sitio proporciona información para desarrolladores de software, y como cabría esperar, recibe menos tráfico los fines de semana. Así, la serie temporal tiene una estacionalidad de cinco días altos y dos días bajos. Los datos están graficados durante varios meses, con las vacaciones de Navidad y Año Nuevo aproximadamente en el medio, por lo que también se puede observar una estacionalidad adicional. Si lo hubiera graficado durante algunos años, verías claramente otra caída al final del año.

Hay muchas formas en que la estacionalidad puede manifestarse en una serie temporal. El tráfico a un sitio web minorista, por ejemplo, podría alcanzar su punto máximo los fines de semana.

### Autocorrelación
Otro rasgo que puedes observar en las series temporales es cuando hay un comportamiento predecible después de un evento. Esto se puede ver en la Figura 9-4, donde hay picos claros, pero después de cada pico, hay una disminución determinista. Esto se llama autocorrelación.
En este caso, podemos ver un conjunto particular de comportamientos que se repiten. Las autocorrelaciones pueden estar ocultas en el patrón de una serie temporal, pero tienen una previsibilidad inherente, por lo que una serie temporal que contiene muchas de ellas puede ser predecible.

![Figura 9-4. Autocorrelación](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.4.png)

### Ruido
Como sugiere el nombre, el ruido es un conjunto de perturbaciones aparentemente aleatorias en una serie temporal. Estas perturbaciones conducen a un alto nivel de imprevisibilidad y pueden enmascarar tendencias, comportamientos estacionales y autocorrelaciones. Por ejemplo, la Figura 9-5 muestra la misma autocorrelación de la Figura 9-4, pero con un poco de ruido añadido. De repente, es mucho más difícil ver la autocorrelación y predecir valores.

![Figura 9-5. Serie autocorrelacionada con ruido añadido](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.5.png)

Dados todos estos factores, exploremos cómo puedes hacer predicciones sobre series temporales que contienen estos atributos.

## Técnicas para predecir series temporales
Antes de adentrarnos en las predicciones basadas en aprendizaje automático (ML), tema de los próximos capítulos, exploraremos algunos métodos más básicos de predicción. Estos te permitirán establecer una línea base que podrás usar para medir la precisión de tus predicciones con ML.

### Predicción básica para crear una línea base
El método más básico para predecir una serie temporal es decir que el valor predicho en el tiempo t+1 es el mismo que el valor en el tiempo t, desplazando efectivamente la serie temporal por un solo período.

Comencemos creando una serie temporal que tenga tendencia, estacionalidad y ruido:

```python
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Un patrón arbitrario que puedes cambiar si lo deseas"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repite el mismo patrón en cada período"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, .05)
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# Crear la serie
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# Actualizar con ruido
series += noise(time, noise_level, seed=42)
```

Después de graficar esto, verás algo como la Figura 9-6.

![Figura 9-6. Una serie temporal que muestra tendencia, estacionalidad y ruido](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.6.png)

Ahora que tienes los datos, puedes dividirlos como cualquier fuente de datos en un conjunto de entrenamiento, uno de validación y uno de prueba. Cuando los datos tienen cierta estacionalidad, como puedes ver en este caso, es una buena idea asegurarte de que haya temporadas completas en cada división. Por ejemplo, si quisieras dividir los datos de la Figura 9-6 en conjuntos de entrenamiento y validación, un buen punto para hacerlo podría ser en el paso de tiempo 1,000, obteniendo datos de entrenamiento hasta el paso 1,000 y datos de validación después del paso 1,000.

No necesitas hacer la división aquí porque solo estás haciendo un pronóstico básico donde cada valor t es simplemente el valor en el paso t−1. Sin embargo, para fines de ilustración en las siguientes figuras, ampliaremos los datos desde el paso de tiempo 1,000 en adelante.

Para predecir la serie a partir de un período dividido, donde el período del que quieres dividir está en la variable split_time, puedes usar un código como este:

```python
naive_forecast = series[split_time - 1:-1]
```

La Figura 9-7 muestra el conjunto de validación (desde el paso de tiempo 1,000 en adelante, que obtienes configurando split_time en 1000) con la predicción básica superpuesta.

![Figura 9-7. Pronóstico básico sobre la serie temporal](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.7.png)

Se ve bastante bien: hay una relación entre los valores y, al graficarlo a lo largo del tiempo, las predicciones parecen coincidir estrechamente con los valores originales. Pero, ¿cómo medirías la precisión?

### Medir la precisión de las predicciones
Hay varias formas de medir la precisión de una predicción, pero nos concentraremos en dos de ellas: el error cuadrático medio (MSE) y el error absoluto medio (MAE).

- Con el MSE, simplemente tomas la diferencia entre el valor predicho y el valor real en el tiempo t, la elevas al cuadrado (para eliminar negativos) y luego calculas el promedio de todas ellas.
- Con el MAE, calculas la diferencia entre el valor predicho y el valor real en el tiempo t, tomas su valor absoluto para eliminar los negativos (en lugar de elevar al cuadrado) y calculas el promedio de todas ellas.

Para el pronóstico básico que acabas de crear basado en nuestra serie temporal sintética, puedes obtener el MSE y el MAE así:

```python
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
```

Obtuve un MSE de 76.47 y un MAE de 6.89. Como con cualquier predicción, si puedes reducir el error, puedes aumentar la precisión de tus predicciones. Veremos cómo hacer eso a continuación.

### Menos ingenuo: Uso del promedio móvil para la predicción
La predicción ingenua anterior tomaba el valor en el tiempo t−1 como el valor pronosticado en el tiempo t. Usar un promedio móvil es similar, pero en lugar de tomar solo el valor de t−1, toma un grupo de valores (digamos, 30), los promedia y establece ese promedio como el valor pronosticado en el tiempo t. Aquí está el código:

```python
def moving_average_forecast(series, window_size):
    """Pronostica el promedio de los últimos valores.
    Si window_size=1, entonces esto equivale al pronóstico ingenuo"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
```

La Figura 9-8 muestra el gráfico del promedio móvil contra los datos.

![Figura 9-8. Graficando el promedio móvil](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.8.png)

Cuando graficé esta serie temporal, obtuve un MSE y MAE de 49 y 5.5, respectivamente, por lo que definitivamente mejoró un poco la predicción. Pero este enfoque no tiene en cuenta la tendencia ni la estacionalidad, así que podríamos mejorarlo aún más con un poco de análisis.

### Mejorando el análisis del promedio móvil
Dado que la estacionalidad en esta serie temporal es de 365 días, puedes suavizar la tendencia y la estacionalidad usando una técnica llamada diferenciación, que simplemente resta el valor en t−365 del valor en t. Esto aplanará el diagrama. Aquí está el código:

```python
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
```

Ahora puedes calcular un promedio móvil de estos valores y agregar de nuevo los valores pasados:

```python
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg
```

Cuando graficas esto (ver la Figura 9-9), ya puedes notar una mejora en los valores predichos: la línea de tendencia está muy cerca de los valores reales, aunque con el ruido suavizado. La estacionalidad parece estar funcionando, al igual que la tendencia.

![Figura 9-9. Promedio móvil mejorado](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure9.9.png)

Esta impresión se confirma al calcular el MSE y MAE—en este caso obtuve 40.9 y 5.13, respectivamente, mostrando una clara mejora en las predicciones.

## Resumen
Este capítulo introdujo los datos de series temporales y algunos de los atributos comunes de las series temporales. Creaste una serie temporal sintética y viste cómo puedes comenzar a hacer predicciones ingenuas sobre ella. A partir de estas predicciones, estableciste mediciones de referencia utilizando el error cuadrático medio (MSE) y el error absoluto medio (MAE). Fue un buen descanso de TensorFlow, pero en el próximo capítulo volverás a usar TensorFlow y ML para ver si puedes mejorar aún más tus predicciones.
