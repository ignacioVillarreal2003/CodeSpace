# Redes Neuronales Recurrentes para el Procesamiento de Lenguaje Natural
En el Capítulo 5, viste cómo tokenizar y secuenciar texto, convirtiendo oraciones en tensores de números que luego podrían ser alimentados a una red neuronal. Luego, ampliaste esto en el Capítulo 6 al explorar las incrustaciones (embeddings), una forma de agrupar palabras con significados similares para habilitar el cálculo de sentimientos. Esto funcionó muy bien, como viste al construir un clasificador de sarcasmo. Pero existe una limitación, y es que las oraciones no son simplemente conjuntos de palabras; a menudo, el orden en el que aparecen las palabras dicta su significado general. Los adjetivos pueden añadir o cambiar el significado de los sustantivos junto a los que aparecen. Por ejemplo, la palabra "azul" podría carecer de significado desde una perspectiva de sentimientos, al igual que "cielo", pero al combinarlas en "cielo azul", hay un sentimiento claro, usualmente positivo. Y algunos sustantivos pueden calificar a otros, como "nube de lluvia", "escritorio de escritura", "taza de café".

Para tener en cuenta secuencias como esta, se necesita un enfoque adicional, que consiste en integrar la recurrencia en la arquitectura del modelo. En este capítulo, explorarás diferentes formas de hacerlo. Veremos cómo se puede aprender la información de secuencia y cómo esta información se puede usar para crear un tipo de modelo que sea mejor para entender texto: la red neuronal recurrente (RNN).

## La Base de la Recurrencia
Para entender cómo funciona la recurrencia, primero consideremos las limitaciones de los modelos utilizados hasta ahora en el libro. En última instancia, crear un modelo se parece un poco a la Figura 7-1. Proporcionas datos y etiquetas, defines una arquitectura de modelo, y el modelo aprende las reglas que ajustan los datos a las etiquetas. Esas reglas luego están disponibles como una API que te devuelve etiquetas predichas para datos futuros.

![Figura 7-1. Vista general de la creación de un modelo](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.1.png)

Sin embargo, como puedes ver, los datos se agrupan en conjunto. No hay granularidad involucrada ni un esfuerzo por entender la secuencia en la que ocurren esos datos. Esto significa que las palabras "azul" y "cielo" no tienen un significado diferente en oraciones como "hoy estoy azul porque el cielo está gris" y "hoy estoy feliz y hay un hermoso cielo azul". Para nosotros, la diferencia en el uso de estas palabras es obvia, pero para un modelo con la arquitectura mostrada aquí, realmente no hay diferencia.

Entonces, ¿cómo solucionamos esto? Primero exploremos la naturaleza de la recurrencia y, a partir de ahí, podrás ver cómo funciona una RNN básica.

Consideremos la famosa secuencia de números de Fibonacci. En caso de que no estés familiarizado con ella, he puesto algunos de sus números en la Figura 7-2.

![Figura 7-2. Los primeros números de la secuencia de Fibonacci](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.2.png)

La idea detrás de esta secuencia es que cada número es la suma de los dos números que lo preceden. Entonces, si comenzamos con 1 y 2, el siguiente número es 1 + 2, que es 3. El siguiente es 2 + 3, que es 5, luego 3 + 5, que es 8, y así sucesivamente.

Podemos colocar esto en un gráfico computacional para obtener la Figura 7-3.

![Figura 7-3. Representación de la secuencia de Fibonacci en un gráfico computacional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.3.png)

Aquí puedes ver que alimentamos 1 y 2 en la función y obtenemos 3 como salida. Llevamos el segundo parámetro (2) al siguiente paso y lo alimentamos junto con la salida del paso anterior (3). La salida de esto es 5, y se alimenta a la función con el segundo parámetro del paso anterior (3) para producir una salida de 8. Este proceso continúa indefinidamente, con cada operación dependiendo de las anteriores. El 1 en la esquina superior izquierda "sobrevive" a través del proceso. Es un elemento del 3 que se alimenta en la segunda operación, es un elemento del 5 que se alimenta en la tercera, y así sucesivamente. Por lo tanto, parte de la esencia del 1 se conserva a lo largo de la secuencia, aunque su impacto en el valor general disminuye.

Esto es análogo a cómo se diseña una neurona recurrente. Puedes ver la representación típica de una neurona recurrente en la Figura 7-4.

![Figura 7-4. Una neurona recurrente](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.4.png)

Un valor x se alimenta a la función F en un paso de tiempo, por lo que típicamente se etiqueta como xt. Esto produce una salida y en ese paso de tiempo, típicamente etiquetada como yt. También produce un valor que se pasa al siguiente paso, indicado por la flecha de F hacia sí misma.

Esto se aclara un poco más si observas cómo las neuronas recurrentes funcionan entre sí a lo largo de pasos de tiempo, lo cual puedes ver en la Figura 7-5.

![Figura 7-5. Neuronas recurrentes en pasos de tiempo](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.5.png)


Aquí, x0 se procesa para obtener y0 y un valor que se pasa adelante. El siguiente paso obtiene ese valor y x1, produciendo y1 y otro valor que se pasa adelante. Esto es similar a lo que vimos con la secuencia de Fibonacci, y siempre encuentro esto útil como un mnemónico para recordar cómo funciona una RNN.

## Extender la Recurrencia para el Lenguaje
En la sección anterior, se mostró cómo una red neuronal recurrente (RNN) que opera a lo largo de varios pasos temporales puede ayudar a mantener el contexto en una secuencia. De hecho, las RNN se utilizarán para modelar secuencias más adelante en este libro. Sin embargo, existe una particularidad en el lenguaje que puede pasarse por alto al usar una RNN simple, como las de las Figuras 7-4 y 7-5. Tal como ocurre en el ejemplo de la secuencia de Fibonacci mencionado anteriormente, la cantidad de contexto que se conserva disminuye con el tiempo. El efecto de la salida de la neurona en el paso 1 es grande en el paso 2, menor en el paso 3, aún más pequeño en el paso 4, y así sucesivamente. Por ejemplo, si tenemos una oración como “Hoy hay un hermoso cielo azul <algo>,” la palabra “azul” tendrá un fuerte impacto en la siguiente palabra; podemos adivinar que probablemente será “cielo”. Pero, ¿qué ocurre con el contexto que proviene de mucho más atrás en una oración? Consideremos, por ejemplo, la oración: “Viví en Irlanda, así que en la escuela secundaria tuve que aprender a hablar y escribir < algo >.”

Ese < algo > es gaélico, pero la palabra que realmente nos da ese contexto es “Irlanda,” que aparece mucho más atrás en la oración. Por lo tanto, para que podamos reconocer lo que < algo > debería ser, se necesita una forma de preservar el contexto a lo largo de una distancia mayor. La memoria a corto plazo de una RNN necesita extenderse, y para ello se inventó una mejora de la arquitectura llamada memoria a largo y corto plazo (LSTM).

Aunque no entraré en detalles sobre cómo funcionan internamente las LSTM, el diagrama de alto nivel mostrado en la Figura 7-6 resume los puntos principales. Para aprender más sobre las operaciones internas, consulta el excelente artículo de Christopher Olah sobre el tema.

La arquitectura LSTM mejora la RNN básica al agregar un “estado de celda” que permite que el contexto se mantenga no solo de un paso a otro, sino a lo largo de toda la secuencia de pasos. Recordando que estas son neuronas que aprenden como lo hacen las neuronas reales, puedes ver que esto garantiza que el contexto importante se aprenda con el tiempo.

![Figura 7-6. Vista de alto nivel de la arquitectura LSTM](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.6.png)


Una parte importante de una LSTM es que puede ser bidireccional: los pasos temporales se iteran tanto hacia adelante como hacia atrás, para que se pueda aprender el contexto en ambas direcciones. Consulta la Figura 7-7 para una vista de alto nivel de esto.

![Figura 7-7. Vista de alto nivel de la arquitectura LSTM bidireccional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.7.png)

De esta manera, se realiza la evaluación en la dirección de 0 a número_de_pasos, así como de número_de_pasos a 0. En cada paso, el resultado y es una agregación del paso "hacia adelante" y el paso "hacia atrás". Puedes verlo en la Figura 7-8.

![Figura 7-8. LSTM bidireccional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.8.png)

Considera cada neurona en cada paso temporal como
F0, F1, F2, etc. Se muestra la dirección del paso temporal, por lo que el cálculo en F1 en la dirección hacia adelante es F1(−>), y en la dirección inversa es (<−)F1. Los valores de estos se agregan para dar el valor y para ese paso temporal. Además, el estado de celda también es bidireccional. Esto puede ser realmente útil para gestionar el contexto en oraciones. Considerando nuevamente la oración “Viví en Irlanda, así que en la escuela secundaria tuve que aprender a hablar y escribir < algo >,” puedes ver cómo el <algo> fue calificado como “gaélico” por la palabra de contexto “Irlanda.” Pero, ¿y si fuera al revés? “Viví en <este país>, así que en la escuela secundaria tuve que aprender a hablar y escribir gaélico.” Puedes ver que, al recorrer la oración hacia atrás, podemos aprender qué debería ser < este país >. Así, usar LSTM bidireccionales puede ser muy poderoso para entender el sentimiento en textos (y como verás en el Capítulo 8, también son realmente poderosos para generar texto).

Por supuesto, las LSTM, en particular las bidireccionales, son complejas, por lo que el entrenamiento será lento. Aquí es donde vale la pena invertir en una GPU, o al menos usar una hospedada en Google Colab si es posible.

## Crear un Clasificador de Texto con RNN
En el Capítulo 6 experimentaste con la creación de un clasificador para el conjunto de datos de sarcasmo usando embeddings. En ese caso, las palabras se transformaron en vectores antes de ser agregados y luego pasados a capas densas para su clasificación. Al usar una capa RNN como una LSTM, no haces la agregación y puedes pasar directamente la salida de la capa de embedding a la capa recurrente. En cuanto a la dimensionalidad de la capa recurrente, una regla general que a menudo verás es que debe tener el mismo tamaño que la dimensión del embedding. Esto no es obligatorio, pero puede ser un buen punto de partida.

Ten en cuenta que, aunque en el Capítulo 6 mencioné que la dimensión del embedding suele ser la raíz cuarta del tamaño del vocabulario, al usar RNN a menudo se ignora esta regla porque haría que el tamaño de la capa recurrente sea demasiado pequeño.

Por ejemplo, la arquitectura del modelo simple para el clasificador de sarcasmo que desarrollaste en el Capítulo 6 podría actualizarse así para usar una LSTM bidireccional:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

La función de pérdida y el clasificador se pueden configurar así (nota que la tasa de aprendizaje es 0.00001, o 1×10−5:

```python
adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer=adam, metrics=['accuracy'])
```

Al imprimir el resumen de la arquitectura del modelo, verás algo como esto. Nota que el tamaño del vocabulario es 20,000 y la dimensión del embedding es 64. Esto da 1,280,000 parámetros en la capa de embedding, y la capa bidireccional tendrá 128 neuronas (64 hacia adelante, 64 hacia atrás):

```python
Layer (type)                Output Shape              Param #   
=================================================================
embedding_11 (Embedding)    (None, None, 64)          1280000   
_________________________________________________________________
bidirectional_7 (Bidirection (None, 128)              66048     
_________________________________________________________________
dense_18 (Dense)            (None, 24)                3096      
_________________________________________________________________
dense_19 (Dense)            (None, 1)                 25        
=================================================================
Total params: 1,349,169
Trainable params: 1,349,169
Non-trainable params: 0
_________________________________________________________________
```

La Figura 7-9 muestra los resultados del entrenamiento con esto durante 30 épocas. 

Como puedes ver, la precisión de la red en los datos de entrenamiento sube rápidamente por encima del 90%, pero los datos de validación se estabilizan alrededor del 80%. Esto es similar a las cifras que obtuvimos anteriormente, pero al inspeccionar el gráfico de pérdida en la Figura 7-10, se observa que, aunque la pérdida para el conjunto de validación divergió después de 15 épocas, también se estabilizó en un valor mucho más bajo que los gráficos de pérdida en el Capítulo 6, a pesar de usar 20,000 palabras en lugar de 2,000.

![Figura 7-9. Precisión para LSTM en 30 épocas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.9.png)

![Figura 7-10. Pérdida con LSTM en 30 épocas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.10.png)

Esto fue solo usando una capa LSTM. En la siguiente sección verás cómo usar LSTM apiladas y explorar el impacto en la precisión al clasificar este conjunto de datos.

### Apilando LSTMs
En la sección anterior, viste cómo usar una capa LSTM después de la capa de embedding para ayudar a clasificar los contenidos del dataset Sarcasm. Pero las LSTMs se pueden apilar una sobre otra, y este enfoque se utiliza en muchos modelos NLP de última generación.

Apilar LSTMs con TensorFlow es bastante sencillo. Las agregas como capas adicionales tal como lo harías con una capa Dense, pero con la excepción de que todas las capas previas a la última deberán tener su propiedad return_sequences configurada en True. Aquí hay un ejemplo:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

La capa final también puede tener return_sequences=True, en cuyo caso devolverá secuencias de valores a las capas Dense para la clasificación en lugar de valores únicos. Esto puede ser útil al analizar la salida del modelo, como discutiremos más adelante. La arquitectura del modelo se verá así:

```python
Capa (tipo)                  Forma de salida               Parámetros
=================================================================
embedding_12 (Embedding)     (None, None, 64)              1,280,000
_________________________________________________________________
bidirectional_8 (Bidirectional) (None, None, 128)          66,048
_________________________________________________________________
bidirectional_9 (Bidirectional) (None, 128)                98,816
_________________________________________________________________
dense_20 (Dense)              (None, 24)                   3,096
_________________________________________________________________
dense_21 (Dense)              (None, 1)                    25
=================================================================
Total de parámetros: 1,447,985
Parámetros entrenables: 1,447,985
Parámetros no entrenables: 0
```

Agregar la capa extra nos dará aproximadamente 100,000 parámetros adicionales que deben ser aprendidos, un aumento de alrededor del 8%. Por lo tanto, podría ralentizar la red, pero el costo es relativamente bajo si hay un beneficio razonable.

Después de entrenar durante 30 épocas, el resultado se ve como en la Figura 7-11. Mientras que la precisión en el conjunto de validación es plana, examinar la pérdida (Figura 7-12) cuenta una historia diferente.

![Figura 7-11. Precisión para la arquitectura LSTM apilada](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.11.png)

Como se puede ver en la Figura 7-12, mientras que la precisión tanto para el entrenamiento como para la validación parecía buena, la pérdida de validación aumentó rápidamente, una clara señal de sobreajuste.

![Figura 7-12. Pérdida para la arquitectura LSTM apilada](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.12.png)

Este sobreajuste (indicado por la precisión de entrenamiento acercándose al 100% mientras la pérdida disminuye suavemente, mientras que la precisión de validación se mantiene relativamente constante y la pérdida aumenta drásticamente) es el resultado de que el modelo se especializa en exceso para el conjunto de entrenamiento. Al igual que con los ejemplos en el Capítulo 6, esto muestra que es fácil caer en una falsa sensación de seguridad si solo se observan las métricas de precisión sin examinar la pérdida.

#### Optimizando LSTMs apilados

En el Capítulo 6, viste que un método muy efectivo para reducir el sobreajuste era reducir la tasa de aprendizaje. Vale la pena explorar si eso tendrá un efecto positivo en una red neuronal recurrente también.

Por ejemplo, el siguiente código reduce la tasa de aprendizaje en un 20% de 0.00001 a 0.000008:

```python
adam = tf.keras.optimizers.Adam(learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
```

La Figura 7-13 demuestra el impacto de esto en el entrenamiento. No parece haber mucha diferencia, aunque las curvas (particularmente para el conjunto de validación) son un poco más suaves.

![Figura 7-13. Impacto de la tasa de aprendizaje reducida en la precisión con LSTMs apilados](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.13.png)

Mientras que una mirada inicial a la Figura 7-14 sugiere un impacto mínimo en la pérdida debido a la tasa de aprendizaje reducida, vale la pena observar más de cerca. A pesar de que la forma de la curva es aproximadamente similar, la tasa de aumento de la pérdida es claramente menor: después de 30 épocas está alrededor de 0.6, mientras que con la tasa de aprendizaje más alta estaba cerca de 0.8. Ajustar el hiperparámetro de la tasa de aprendizaje ciertamente parece valer la pena investigar.

![Figura 7-14. Impacto de la tasa de aprendizaje reducida en la pérdida con LSTMs apilados](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.14.png)

#### Usando dropout

Además de cambiar el parámetro de la tasa de aprendizaje, también vale la pena considerar el uso de dropout en las capas LSTM. Funciona exactamente igual que para las capas densas, como se discutió en el Capítulo 3, donde se eliminan aleatoriamente neuronas para evitar que un sesgo de proximidad impacte en el aprendizaje.

El dropout se puede implementar utilizando un parámetro en la capa LSTM. Aquí hay un ejemplo:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, dropout=0.2)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Tenga en cuenta que implementar dropout ralentizará considerablemente su entrenamiento. En mi caso, usando Colab, pasó de ~10 segundos por época a ~180 segundos.

Los resultados de precisión se pueden ver en la Figura 7-15.

![Figura 7-15. Precisión de LSTMs apilados usando dropout](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.15.png)

Como se puede ver, usar dropout no tiene mucho impacto en la precisión de la red, ¡lo cual es bueno! Siempre hay una preocupación de que perder neuronas hará que su modelo funcione peor, pero como podemos ver aquí, ese no es el caso.

También hay un impacto positivo en la pérdida, como se puede ver en la Figura 7-16.

Mientras que las curvas claramente se están separando, están más cerca de lo que estaban anteriormente, y el conjunto de validación se está aplanando en una pérdida de alrededor de 0.5. Eso es significativamente mejor que el 0.8 visto anteriormente. Como muestra este ejemplo, el dropout es otra técnica útil que puede usar para mejorar el rendimiento de las RNNs basadas en LSTM.

![Figura 7-16. Curvas de pérdida para LSTMs con dropout habilitado](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.16.png)

Vale la pena explorar estas técnicas para evitar el sobreajuste en sus datos, así como las técnicas para preprocesar sus datos que cubrimos en el Capítulo 6. Pero hay una cosa que aún no hemos probado: una forma de aprendizaje por transferencia donde puede usar incrustaciones pre aprendidas para palabras en lugar de intentar aprender las tuyas. Exploraremos eso a continuación.

## Usando Embeddings Preentrenados con RNNs
En todos los ejemplos anteriores, recopilaste el conjunto completo de palabras a usar en el conjunto de entrenamiento y luego entrenaste embeddings con ellas. Estos se agregaron inicialmente antes de ser alimentados a una red densa, y en este capítulo exploraste cómo mejorar los resultados utilizando una RNN. Al hacerlo, estuviste limitado a las palabras de tu conjunto de datos y a cómo sus embeddings podían aprenderse utilizando las etiquetas de ese conjunto de datos.

Recuerda el Capítulo 4, donde discutimos el aprendizaje por transferencia. ¿Qué pasaría si, en lugar de aprender los embeddings tú mismo, pudieras usar embeddings preentrenados, donde investigadores ya hayan hecho el trabajo duro de convertir palabras en vectores, y esos vectores estén probados? Un ejemplo de esto es el modelo GloVe (Global Vectors for Word Representation) desarrollado por Jeffrey Pennington, Richard Socher y Christopher Manning en Stanford.

En este caso, los investigadores han compartido sus vectores de palabras preentrenados para una variedad de conjuntos de datos:

- Un vocabulario de 400,000 palabras y 6 mil millones de tokens en 50, 100, 200 y 300 dimensiones, con palabras tomadas de Wikipedia y Gigaword.
- Un vocabulario de 1.9 millones de palabras y 42 mil millones de tokens en 300 dimensiones de un common crawl.
- Un vocabulario de 2.2 millones de palabras y 840 mil millones de tokens en 300 dimensiones de un common crawl.
- Un vocabulario de 1.2 millones de palabras y 27 mil millones de tokens en 25, 50, 100 y 200 dimensiones de un rastreo de 2 mil millones de tweets en Twitter.

Dado que los vectores ya están preentrenados, es sencillo reutilizarlos en tu código de TensorFlow en lugar de aprender desde cero. Primero, tendrás que descargar los datos de GloVe. Elegí usar los datos de Twitter con 27 mil millones de tokens y un vocabulario de 1.2 millones de palabras. La descarga es un archivo comprimido con versiones de 25, 50, 100 y 200 dimensiones.

Para facilitarte un poco las cosas, he alojado la versión de 25 dimensiones, y puedes descargarla en un cuaderno de Colab de esta manera:

```python
!wget --no-check-certificate \
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.twitter.27B.25d.zip \
-O /tmp/glove.zip
```

Es un archivo ZIP, por lo que puedes extraerlo así para obtener un archivo llamado glove.twitter.27b.25d.txt:

```python
# Descomprimir embeddings de GloVe
import os
import zipfile

local_zip = '/tmp/glove.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/glove')
zip_ref.close()
```

Cada entrada en el archivo es una palabra, seguida de los coeficientes dimensionales que se aprendieron para ella. La forma más sencilla de usar esto es crear un diccionario donde la clave sea la palabra y los valores sean los embeddings. Puedes configurar este diccionario así:

```python
glove_embeddings = dict()
f = open('/tmp/glove/glove.twitter.27B.25d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()
```

En este punto, podrás buscar el conjunto de coeficientes para cualquier palabra simplemente usándola como clave. Por ejemplo, para ver los embeddings de “frog”, podrías usar:

```python
glove_embeddings['frog']
```

Con este recurso a mano, puedes usar el tokenizador para obtener el índice de palabras para tu corpus como antes, pero ahora puedes crear una nueva matriz, que llamaré la matriz de embeddings. Esta utilizará los embeddings del conjunto GloVe (tomados de glove_embeddings) como sus valores. Así que, si examinas las palabras en el índice de palabras para tu conjunto de datos, como esto:

```python
{'<OOV>': 1, 'new': 2, … 'not': 5, 'just': 6, 'will': 7}
```

Entonces, la primera fila en la matriz de embeddings debería ser los coeficientes de GloVe para < OOV >, la siguiente fila serán los coeficientes para “new” y así sucesivamente.

Puedes crear esa matriz con este código:

```python
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
```

Esto simplemente crea una matriz con las dimensiones de tu vocabulario deseado y la dimensión de embeddings. Luego, para cada elemento en el índice de palabras del tokenizador, buscas los coeficientes de GloVe en glove_embeddings y agregas esos valores a la matriz.

Luego modificas la capa de embeddings para usar los embeddings preentrenados estableciendo el parámetro weights, y especificas que no deseas que la capa se entrene configurando trainable=False:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Ahora puedes entrenar como antes. Sin embargo, querrás considerar el tamaño de tu vocabulario. Una de las optimizaciones que hiciste en el capítulo anterior para evitar el sobreajuste estaba destinada a prevenir que las embeddings se sobrecargaran al aprender palabras de baja frecuencia; evitaste el sobreajuste utilizando un vocabulario más pequeño compuesto por palabras de uso frecuente. En este caso, dado que las embeddings de las palabras ya han sido aprendidas por ti con GloVe, podrías ampliar el vocabulario, pero ¿en qué medida?

Lo primero que debes explorar es cuántas de las palabras en tu corpus están realmente en el conjunto de GloVe. Este tiene 1.2 millones de palabras, pero no hay garantía de que tenga todas las de tu conjunto. Aquí tienes un código para realizar una comparación rápida y explorar cuán grande debería ser tu vocabulario:

Primero, organiza los datos. Crea una lista de Xs e Ys, donde X es el índice de la palabra, e Y=1 si la palabra está en las embeddings y 0 si no lo está. Adicionalmente, puedes crear un conjunto acumulativo, donde cuentas la proporción de palabras en cada paso. Por ejemplo, la palabra "OOV" en el índice 0 no está en GloVe, por lo que su Y acumulativo sería 0. La palabra "new", en el siguiente índice, está en GloVe, por lo que su Y acumulativo sería 0.5 (es decir, la mitad de las palabras vistas hasta ahora están en GloVe), y continúas contando de esa forma para todo el conjunto de datos:

```python
xs = []
ys = []
cumulative_x = []
cumulative_y = []
total_y = 0

for word, index in tokenizer.word_index.items():
    xs.append(index)
    cumulative_x.append(index)
    if glove_embeddings.get(word) is not None: 
        total_y = total_y + 1
        ys.append(1)
    else:
        ys.append(0)
    cumulative_y.append(total_y / index)
```

Luego, grafica los Xs contra los Ys con este código:


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 2))
ax.spines['top'].set_visible(False)

plt.margins(x=0, y=None, tight=True)
# plt.axis([13000, 14000, 0, 1])
plt.fill(ys)
```

Esto te dará un gráfico de frecuencia de palabras, que se verá algo así como en la Figura 7-17.

![Figura 7-17. Cuadro de frecuencia de palabras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.17.png)

Como puedes ver en el gráfico, la densidad cambia en algún lugar entre 10,000 y 15,000. Esto te da una referencia visual de que, aproximadamente en el token 13,000, la frecuencia de palabras que no están en las embeddings de GloVe comienza a superar a las que sí están.

Si luego graficas el cumulative_x contra el cumulative_y, puedes obtener una mejor percepción de esto. Aquí tienes el código:

```python
import matplotlib.pyplot as plt

plt.plot(cumulative_x, cumulative_y)
plt.axis([0, 25000, .915, .985])
```

Puedes ver los resultados en la Figura 7-18.

![Figura 7-18. Representación gráfica de la frecuencia del índice de palabras en relación con GloVe](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.18.png)

Ajusta los parámetros en plt.axis para acercarte y encontrar el punto de inflexión donde las palabras que no están en GloVe comienzan a superar a las que sí están. Este es un buen punto de partida para establecer el tamaño de tu vocabulario.

Usando este método, decidí usar un tamaño de vocabulario de 13,200 (en lugar de los 2,000 que se utilizaban previamente para evitar el sobreajuste) y esta arquitectura de modelo, donde el embedding_dim es 25 debido al conjunto GloVe que estoy usando:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
        weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, 
        return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
```

Entrenar esto durante 30 épocas produce excelentes resultados. La precisión se muestra en la Figura 7-19. La precisión de validación es muy cercana a la precisión de entrenamiento, lo que indica que ya no estamos sobreajustando.

![Figura 7-19. Precisión del LSTM apilado utilizando embeddings de GloVe](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.19.png)

Esto se refuerza con las curvas de pérdida, como se muestra en la Figura 7-20. La pérdida de validación ya no diverge, lo que demuestra que, aunque nuestra precisión es de solo ~73%, podemos estar seguros de que el modelo es preciso en ese grado.

![Figura 7-20. Pérdida del LSTM apilado utilizando embeddings de GloVe](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.20.png)

Entrenar el modelo por más tiempo muestra resultados muy similares e indica que, aunque el sobreajuste comienza a ocurrir alrededor de la época 80, el modelo sigue siendo muy estable. Las métricas de precisión (Figura 7-21) muestran un modelo bien entrenado. Las métricas de pérdida (Figura 7-22) muestran el inicio de la divergencia alrededor de la época 80, pero el modelo todavía se ajusta bien.

![Figura 7-21. Precisión en el LSTM apilado con GloVe durante 150 épocas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.21.png)

![Figura 7-22. Pérdida en el LSTM apilado con GloVe durante 150 épocas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure7.22.png)

Esto nos indica que este modelo es un buen candidato para detenerse tempranamente, donde puedes simplemente entrenarlo durante 75–80 épocas para obtener los resultados óptimos.

Lo probé con titulares de The Onion, fuente de titulares sarcásticos en el conjunto de datos de sarcasmo, frente a otras oraciones, como se muestra aquí:

```python
test_sentences = [
"It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, Explaining Why They Had To Give Him Haircut",
"It's a beautiful sunny day",
"I lived in Ireland, so in high school they made me learn to speak and write in Gaelic",
"Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally Household Sizes"
]
```

Los resultados para estos titulares son los siguientes: recuerda que los valores cercanos al 50% (0.5) se consideran neutrales, cercanos a 0 no sarcásticos y cercanos a 1 sarcásticos:

```python
[[0.8170955 ]
 [0.08711044]
 [0.61809343]
 [0.8015281 ]]
```

Las primeras y cuartas oraciones, tomadas de The Onion, mostraron una probabilidad de sarcasmo superior al 80%. La afirmación sobre el clima fue claramente no sarcástica (9%) y la oración sobre asistir a la escuela secundaria en Irlanda fue considerada potencialmente sarcástica, pero no con alta confianza (62%).

## Resumen
Este capítulo te introdujo a las redes neuronales recurrentes, que utilizan lógica orientada a secuencias en su diseño y pueden ayudarte a entender el sentimiento en oraciones basándose no solo en las palabras que contienen, sino también en el orden en que aparecen.

Viste cómo funciona una RNN básica, así como cómo un LSTM puede construir sobre esto para permitir que el contexto se preserve a largo plazo. Usaste esto para mejorar el modelo de análisis de sentimiento en el que has estado trabajando. Luego analizaste problemas de sobreajuste con las RNN y técnicas para mejorarlos, incluyendo el uso de aprendizaje por transferencia con embeddings preentrenados. En el Capítulo 8 usarás lo que has aprendido para explorar cómo predecir palabras, y a partir de ahí podrás crear un modelo que genere texto, ¡escribiendo poesía para ti!
