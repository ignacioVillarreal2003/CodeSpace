# Haciendo que el sentimiento sea programable usando incrustaciones
En el capítulo 5, viste cómo tomar palabras y codificarlas en tokens. Luego viste cómo codificar oraciones llenas de palabras en secuencias llenas de tokens, rellenándolas o truncándolas según sea necesario para obtener un conjunto de datos bien estructurado que se pueda usar para entrenar una red neuronal. En ninguno de esos pasos hubo algún tipo de modelado del significado de una palabra. Si bien es cierto que no existe una codificación numérica absoluta que pueda encapsular el significado, sí existen codificaciones relativas. En este capítulo aprenderás sobre ellas, y en particular sobre el concepto de incrustaciones, donde se crean vectores en un espacio de alta dimensionalidad para representar palabras. Las direcciones de estos vectores se pueden aprender con el tiempo en función del uso de las palabras en el corpus. Luego, cuando se te da una oración, puedes investigar las direcciones de los vectores de palabras, sumarlas, y a partir de la dirección general de la suma, establecer el sentimiento de la oración como un producto de sus palabras.

En este capítulo exploraremos cómo funciona esto. Usando el conjunto de datos de sarcasmo del capítulo 5, construirás incrustaciones para ayudar a un modelo a detectar sarcasmo en una oración. También verás algunas herramientas de visualización interesantes que te ayudarán a entender cómo las palabras en un corpus se asignan a vectores, para que puedas ver qué palabras determinan la clasificación general.

## Estableciendo significado a partir de palabras
Antes de entrar en los vectores de mayor dimensión para las incrustaciones, intentemos visualizar cómo se puede derivar significado a partir de valores numéricos con algunos ejemplos simples. Considera esto: usando el conjunto de datos de sarcasmo del capítulo 5, ¿qué pasaría si codificaras todas las palabras que forman titulares sarcásticos con números positivos y aquellas que forman titulares realistas con números negativos?

### Un ejemplo simple: positivos y negativos
Toma, por ejemplo, este titular sarcástico del conjunto de datos:

```python
christian bale given neutered male statuette named oscar
```

Suponiendo que todas las palabras en nuestro vocabulario comienzan con un valor de 0, podríamos sumar 1 a los valores de cada una de las palabras en esta oración, y terminaríamos con esto:

```python
{ "christian": 1, "bale": 1, "given": 1, "neutered": 1, "male": 1, "statuette": 1, "named": 1, "oscar": 1 }
```

> Nota que esto no es lo mismo que la tokenización de palabras que realizaste en el capítulo anterior. Podrías considerar reemplazar cada palabra (por ejemplo, “christian”) con el token que la representa y que está codificado en el corpus, pero dejaré las palabras por ahora para hacerlo más fácil de leer.

Luego, en el siguiente paso, considera un titular ordinario, no sarcástico, como este:

```python
gareth bale scores wonder goal against germany
```

Dado que este es un sentimiento diferente, podríamos en su lugar restar 1 del valor actual de cada palabra, de modo que nuestro conjunto de valores se vería así:

```python
{ "christian": 1, "bale": 0, "given": 1, "neutered": 1, "male": 1, "statuette": 1, "named": 1, "oscar": 1, "gareth": -1, "scores": -1, "wonder": -1, "goal": -1, "against": -1, "germany": -1 }
```

Nota que el “bale” sarcástico (de “christian bale”) ha sido compensado por el “bale” no sarcástico (de “gareth bale”), por lo que su puntuación termina siendo 0. Repite este proceso miles de veces y obtendrás una lista enorme de palabras de tu corpus puntuadas en función de su uso.

Ahora imagina que queremos establecer el sentimiento de esta oración:

```python
neutered male named against germany, wins statuette!
```

Usando nuestro conjunto de valores existente, podríamos observar las puntuaciones de cada palabra y sumarlas. Obtendríamos una puntuación de 2, lo que indica (porque es un número positivo) que esta es una oración sarcástica.

> Para que lo sepas, “bale” se usa cinco veces en el conjunto de datos de sarcasmo, dos veces en un titular normal y tres veces en uno sarcástico, por lo que en un modelo como este la palabra “bale” tendría una puntuación de –1 en todo el conjunto de datos.

### Yendo un Poco Más Profundo: Vectores
Con suerte, el ejemplo anterior te ha ayudado a comprender el modelo mental de establecer alguna forma de significado relativo para una palabra, a través de su asociación con otras palabras en la misma “dirección”. En nuestro caso, aunque la computadora no entiende los significados de las palabras individuales, puede mover palabras etiquetadas de un titular sarcástico conocido en una dirección (sumando 1) y palabras etiquetadas de un titular normal conocido en otra dirección (restando 1). Esto nos da una comprensión básica del significado de las palabras, pero pierde algo de matices.

¿Qué pasaría si aumentáramos la dimensionalidad de la dirección para intentar capturar más información? Por ejemplo, supongamos que consideramos personajes de la novela Orgullo y Prejuicio de Jane Austen, en las dimensiones de género y nobleza. Podríamos graficar la primera en el eje x y la segunda en el eje y, con la longitud del vector representando la riqueza de cada personaje (Figura 6-1).

![Figura 6-1. Personajes de Orgullo y Prejuicio como vectores](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.1.png)

Al inspeccionar el gráfico, puedes derivar bastante información sobre cada personaje. Tres de ellos son hombres. Mr. Darcy es extremadamente rico, pero su nobleza no es clara (se le llama "Mister", a diferencia del menos rico pero aparentemente más noble Sir William Lucas). El otro "Mister", Mr. Bennet, claramente no es noble y tiene dificultades financieras. Elizabeth Bennet, su hija, es similar a él, pero femenina. Lady Catherine, el otro personaje femenino de nuestro ejemplo, es noble y muy rica. El romance entre Mr. Darcy y Elizabeth causa tensiones: prejuicio proveniente del lado noble de los vectores hacia el menos noble.

Como muestra este ejemplo, al considerar múltiples dimensiones podemos comenzar a ver un significado real en las palabras (aquí, nombres de personajes). De nuevo, no estamos hablando de definiciones concretas, sino más bien de un significado relativo basado en los ejes y la relación entre el vector de una palabra y los otros vectores.

Esto nos lleva al concepto de embedding, que es simplemente una representación vectorial de una palabra aprendida mientras se entrena una red neuronal. Exploraremos esto a continuación.

## Embeddings en TensorFlow
Como has visto con Dense y Conv2D, tf.keras implementa los embeddings usando una capa. Esto crea una tabla de búsqueda que mapea de un entero a una tabla de embeddings, cuyo contenido son los coeficientes del vector que representa la palabra identificada por ese entero. Así, en el ejemplo de Orgullo y Prejuicio de la sección anterior, las coordenadas x e y nos darían los embeddings para un personaje particular del libro. Por supuesto, en un problema real de PLN, usaremos muchas más dimensiones que dos.

Por lo tanto, la dirección de un vector en el espacio vectorial podría considerarse como una codificación del "significado" de una palabra, y palabras con vectores similares, es decir, que apuntan en aproximadamente la misma dirección, podrían considerarse relacionadas con esa palabra.

La capa de embedding se inicializará aleatoriamente, es decir, las coordenadas de los vectores serán completamente aleatorias al principio y se aprenderán durante el entrenamiento usando retropropagación (backpropagation). Cuando el entrenamiento esté completo, los embeddings codificarán aproximadamente similitudes entre palabras, permitiéndonos identificar palabras algo similares basándonos en la dirección de los vectores para esas palabras.

Todo esto es bastante abstracto, así que creo que la mejor manera de entender cómo usar embeddings es arremangarse y probarlos. Comencemos con un detector de sarcasmo usando el conjunto de datos Sarcasm.

### Construyendo un Detector de Sarcasmo Usando Embeddings
En el Capítulo 5 cargaste e hiciste algo de preprocesamiento en un conjunto de datos JSON llamado News Headlines Dataset for Sarcasm Detection (Sarcasm, en resumen). Al final, tenías listas de datos de entrenamiento y prueba y etiquetas. Estas se pueden convertir al formato Numpy, usado por TensorFlow para entrenar, con código como este:

```python
import numpy as np  
training_padded = np.array(training_padded)  
training_labels = np.array(training_labels)  
testing_padded = np.array(testing_padded)  
testing_labels = np.array(testing_labels)  
```

Estas fueron creadas usando un tokenizer con un tamaño máximo de vocabulario especificado y un token fuera de vocabulario:

```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
```

Para inicializar la capa de embedding, necesitarás el tamaño del vocabulario y un número especificado de dimensiones para el embedding:

```python
tf.keras.layers.Embedding(vocab_size, embedding_dim),
```

Esto inicializará un arreglo con puntos de embedding_dim para cada palabra. Así, por ejemplo, si embedding_dim es 16, cada palabra en el vocabulario será asignada a un vector de 16 dimensiones.

Con el tiempo, las dimensiones se aprenderán a través de retropropagación mientras la red aprende asociando los datos de entrenamiento con sus etiquetas.

Un paso importante a continuación es alimentar la salida de la capa de embedding en una capa dense. La manera más fácil de hacerlo, similar a cómo lo harías al usar una red neuronal convolucional, es usar pooling. En este caso, las dimensiones de los embeddings se promedian para producir un vector de salida de longitud fija.

Como ejemplo, considera esta arquitectura de modelo:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

Aquí se define una capa de embedding, y se le da el tamaño del vocabulario (10000) y una dimensión de embedding de 16. Veamos el número de parámetros entrenables en la red usando model.summary:

```python
Model: "sequential_2"  
_________________________________________________________________  
Layer (type)                 Output Shape              Param #     
=================================================================  
embedding_2 (Embedding)      (None, None, 16)          160000      
_________________________________________________________________  
global_average_pooling1d_2 ( (None, 16)                0           
_________________________________________________________________  
dense_4 (Dense)              (None, 24)               408         
_________________________________________________________________  
dense_5 (Dense)              (None, 1)                25          
=================================================================  
Total params: 160,433  
Trainable params: 160,433  
Non-trainable params: 0  
_________________________________________________________________
```

Como el embedding tiene un vocabulario de 10,000 palabras y cada palabra será un vector en 16 dimensiones, el número total de parámetros entrenables será 160,000. La capa de average pooling tiene 0 parámetros entrenables, ya que simplemente promedia los parámetros en la capa de embedding antes de ella para obtener un único vector de 16 valores.

Esto se alimenta luego a la capa dense de 24 neuronas. Recuerda que una neurona densa calcula efectivamente usando pesos y sesgos, por lo que necesitará aprender (24 × 16) + 16 = 408 parámetros.

La salida de esta capa se pasa luego a la capa final de una sola neurona, donde habrá (1 × 24) + 1 = 25 parámetros por aprender.

Si entrenamos este modelo, obtendremos una precisión bastante buena del 99+% después de 30 épocas, pero nuestra precisión de validación será solo del 81% (Figura 6-2).

![Figura 6-2. Precisión de entrenamiento frente a precisión de validación](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.2.png)

Esto podría parecer una curva razonable dado que los datos de validación probablemente contienen muchas palabras que no están presentes en los datos de entrenamiento. Sin embargo, si examinas las curvas de pérdida para el entrenamiento frente a la validación durante las 30 épocas, verás un problema.

Aunque esperarías que la precisión de entrenamiento sea mayor que la precisión de validación, un claro indicador de sobreajuste es que, mientras que la precisión de validación disminuye un poco con el tiempo (en la Figura 6-2), su pérdida aumenta drásticamente, como se muestra en la Figura 6-3.

![Figura 6-3. Pérdida de entrenamiento frente a pérdida de validación](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.3.png)

El sobreajuste como este es común con modelos de PLN debido a la naturaleza algo impredecible del lenguaje. En las siguientes secciones, veremos cómo reducir este efecto usando una serie de técnicas.

### Reduciendo el sobreajuste en modelos de lenguaje
El sobreajuste ocurre cuando la red se especializa demasiado en los datos de entrenamiento, y una parte de esto implica que se vuelve muy buena para detectar patrones en datos "ruidosos" del conjunto de entrenamiento que no existen en ningún otro lugar. Dado que este ruido particular no está presente en el conjunto de validación, cuanto mejor se vuelve la red para ajustarlo, peor será la pérdida en el conjunto de validación. Esto puede resultar en la escalada de la pérdida que viste en la Figura 6-3. En esta sección, exploraremos varias formas de generalizar el modelo y reducir el sobreajuste.

#### Ajustando la tasa de aprendizaje
Quizás el mayor factor que puede llevar al sobreajuste es si la tasa de aprendizaje de tu optimizador es demasiado alta. Esto significa que la red aprende demasiado rápido. Para este ejemplo, el código para compilar el modelo era el siguiente:

```python
model.compile(loss='binary_crossentropy',  
              optimizer='adam', metrics=['accuracy'])  
```

El optimizador simplemente se declara como adam, lo que invoca al optimizador Adam con parámetros predeterminados. Este optimizador, sin embargo, admite múltiples parámetros, incluida la tasa de aprendizaje. Puedes cambiar el código a esto:

```python
adam = tf.keras.optimizers.Adam(learning_rate=0.0001,  
                                beta_1=0.9, beta_2=0.999, amsgrad=False)  
model.compile(loss='binary_crossentropy',  
              optimizer=adam, metrics=['accuracy'])  
```

donde el valor predeterminado para la tasa de aprendizaje, que típicamente es 0.001, se ha reducido en un 90%, a 0.0001. Los valores de beta_1 y beta_2 permanecen en sus valores predeterminados, al igual que amsgrad.

beta_1 y beta_2 deben estar entre 0 y 1, y generalmente ambos están cerca de 1. amsgrad es una implementación alternativa del optimizador Adam, introducida en el artículo “On the Convergence of Adam and Beyond” de Sashank Reddi, Satyen Kale y Sanjiv Kumar.

Esta tasa de aprendizaje mucho más baja tiene un impacto profundo en la red. La Figura 6-4 muestra la precisión de la red durante 100 épocas. La tasa de aprendizaje más baja se puede observar en las primeras 10 épocas aproximadamente, donde parece que la red no está aprendiendo, antes de que "despegue" y comience a aprender rápidamente.

![Figura 6-4. Precisión con una tasa de aprendizaje más baja](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.4.png)

Al explorar la pérdida (como se ilustra en la Figura 6-5), podemos ver que incluso cuando la precisión no aumentaba en las primeras épocas, la pérdida disminuía, por lo que podrías estar seguro de que la red eventualmente comenzaría a aprender si la observabas época por época.

![Figura 6-5. Pérdida con una tasa de aprendizaje más baja](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.5.png)

Y aunque la pérdida comienza a mostrar la misma curva de sobreajuste que viste en la Figura 6-3, observa que ocurre mucho más tarde y a un ritmo mucho menor. Para la época 30, la pérdida está en alrededor de 0.45, mientras que con la tasa de aprendizaje más alta en la Figura 6-3 era más del doble de esa cantidad. Y aunque la red tarda más en alcanzar una buena tasa de precisión, lo hace con menos pérdida, por lo que puedes tener más confianza en los resultados.

Con estos hiperparámetros, la pérdida en el conjunto de validación comenzó a aumentar alrededor de la época 60, momento en el cual el conjunto de entrenamiento tenía una precisión del 90% y el conjunto de validación alrededor del 81%, lo que demuestra que tenemos una red bastante efectiva.

Por supuesto, es fácil simplemente ajustar el optimizador y luego declarar victoria, pero hay una serie de otros métodos que puedes usar para mejorar tu modelo, los cuales verás en las próximas secciones. Para estos, volví a usar el optimizador Adam predeterminado para que los efectos de ajustar la tasa de aprendizaje no oculten los beneficios ofrecidos por estas otras técnicas.

#### Explorando el tamaño del vocabulario
El conjunto de datos de Sarcasmo trabaja con palabras, por lo que si exploras las palabras en el conjunto de datos, y en particular su frecuencia, podrías obtener una pista que ayude a resolver el problema de sobreajuste. El tokenizador te permite hacer esto con su propiedad word_counts. Si lo imprimieras, verías algo como esto, un OrderedDict que contiene tuplas de palabra y su cantidad:

```python
wc = tokenizer.word_counts  
print(wc)  
OrderedDict([('former', 75), ('versace', 1), ('store', 35), ('clerk', 8), 
('sues', 12), ('secret', 68), ('black', 203), ('code', 16),...  
```

El orden de las palabras está determinado por su orden de aparición en el conjunto de datos. Si miras el primer titular en el conjunto de entrenamiento, es uno sarcástico sobre un ex empleado de una tienda Versace. Se han eliminado las palabras vacías; de lo contrario, verías un alto volumen de palabras como "a" y "the".

Dado que es un OrderedDict, puedes ordenarlo en orden descendente por volumen de palabras:

```python
from collections import OrderedDict  
newlist = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))  
print(newlist)  
OrderedDict([('new', 1143), ('trump', 966), ('man', 940), ('not', 555), ('just',  
430), ('will', 427), ('one', 406), ('year', 386), ...  
```

Si quieres graficarlo, puedes iterar a través de cada elemento en la lista y hacer que el valor x sea el ordinal de donde estás (1 para el primer elemento, 2 para el segundo, etc.). El valor y será entonces newlist[item]. Esto se puede graficar con matplotlib. Aquí está el código:

```python
xs = []  
ys = []  
curr_x = 1  
for item in newlist:  
    xs.append(curr_x)  
    curr_x = curr_x + 1  
    ys.append(newlist[item])  
plt.plot(xs, ys)  
plt.show()  
```

El resultado se muestra en la Figura 6-6.

![Figura 6-6. Explorando la frecuencia de palabras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.6.png)

Esta curva en forma de "stick de hockey" nos muestra que muy pocas palabras se usan muchas veces, mientras que la mayoría se usan muy pocas veces. Pero cada palabra tiene el mismo peso porque todas tienen una "entrada" en el embedding. Dado que tenemos un conjunto de entrenamiento relativamente grande en comparación con el conjunto de validación, terminamos en una situación donde hay muchas palabras presentes en el conjunto de entrenamiento que no están presentes en el de validación.

Puedes acercarte a los datos cambiando el eje del gráfico justo antes de llamar a plt.show. Por ejemplo, para observar el volumen de palabras de 300 a 10,000 en el eje x con la escala de 0 a 100 en el eje y, puedes usar este código:

```python
plt.plot(xs, ys)  
plt.axis([300, 10000, 0, 100])  
plt.show()  
```

El resultado está en la Figura 6-7.

![Figura 6-7. Frecuencia de palabras 300–10,000](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.7.png)

Aunque hay más de 20,000 palabras en el corpus, el código está configurado para entrenar solo con 10,000. Pero si miramos las palabras en las posiciones 2,000–10,000, que representan más del 80% de nuestro vocabulario, vemos que cada una se usa menos de 20 veces en todo el corpus. Esto podría explicar el sobreajuste. Ahora considera qué sucede si cambias el tamaño del vocabulario a dos mil y vuelves a entrenar. La Figura 6-8 muestra las métricas de precisión. Ahora la precisión del conjunto de entrenamiento es ~82% y la precisión del conjunto de validación es aproximadamente 76%. Están más cerca entre sí y no divergen, lo cual es una buena señal de que hemos eliminado gran parte del sobreajuste.

![Figura 6-8. Precisión con un vocabulario de dos mil palabras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.8.png)

Esto se refuerza un poco con el gráfico de pérdida en la Figura 6-9. La pérdida en el conjunto de validación está aumentando, pero mucho más lentamente que antes, por lo que reducir el tamaño del vocabulario para evitar que el conjunto de entrenamiento se sobreajuste con palabras de baja frecuencia que posiblemente solo estaban presentes en el conjunto de entrenamiento parece haber funcionado.

![Figura 6-9. Pérdida con un vocabulario de dos mil palabras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.9.png)

Vale la pena experimentar con diferentes tamaños de vocabulario, pero recuerda que también puedes tener un vocabulario demasiado pequeño y sobreajustarlo. Necesitarás encontrar un equilibrio. En este caso, mi elección de tomar palabras que aparecen 20 veces o más fue puramente arbitraria.

#### Explorando las dimensiones de embedding
En este ejemplo, se eligió arbitrariamente una dimensión de embedding de 16. En este caso, las palabras se codifican como vectores en un espacio de 16 dimensiones, con sus direcciones indicando su significado general. Pero ¿es 16 un buen número? Con solo dos mil palabras en nuestro vocabulario, podría ser excesivo, llevando a un alto grado de dispersión en las direcciones.

La mejor práctica para el tamaño de embedding es que sea la cuarta raíz del tamaño del vocabulario. La cuarta raíz de 2,000 es 6.687, así que exploremos qué sucede si cambiamos la dimensión de embedding a 7 y volvemos a entrenar el modelo por 100 épocas.

Puedes ver el resultado en la precisión en la Figura 6-9. La precisión del conjunto de entrenamiento se estabilizó en aproximadamente 83% y la del conjunto de validación en aproximadamente 77%. A pesar de algunas fluctuaciones, las líneas son bastante planas, mostrando que el modelo ha convergido. Esto no es muy diferente de los resultados en la Figura 6-6, pero reducir la dimensionalidad del embedding permite que el modelo entrene un 30% más rápido.

![Figura 6-10. Precisión en entrenamiento versus validación para siete dimensiones](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.10.png)

La Figura 6-11 muestra la pérdida en entrenamiento y validación. Aunque inicialmente parecía que la pérdida estaba aumentando en aproximadamente la época 20, pronto se estabilizó. ¡Otra buena señal!

![Figura 6-11. Pérdida en entrenamiento versus validación para siete dimensiones](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.11.png)

Ahora que se ha reducido la dimensionalidad, podemos ajustar un poco más la arquitectura del modelo.

#### Explorando la arquitectura del modelo
Después de las optimizaciones de las secciones anteriores, la arquitectura del modelo ahora luce así:

```python
model = tf.keras.Sequential([  
    tf.keras.layers.Embedding(2000, 7),  
    tf.keras.layers.GlobalAveragePooling1D(),  
    tf.keras.layers.Dense(24, activation='relu'),  
    tf.keras.layers.Dense(1, activation='sigmoid')  
])  
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
```

Algo que salta a la vista es la dimensionalidad: la capa GlobalAveragePooling1D ahora emite solo siete dimensiones, pero se están alimentando a una capa densa de 24 neuronas, lo cual es excesivo. Exploremos qué sucede cuando esto se reduce a solo ocho neuronas y se entrena durante cien épocas.

Puedes ver la precisión de entrenamiento versus validación en la Figura 6-12. En comparación con la Figura 6-7, donde se usaron 24 neuronas, el resultado general es bastante similar, pero las fluctuaciones se han suavizado (visible en las líneas menos irregulares). También es algo más rápido de entrenar.

![Figura 6-12. Resultados de precisión con arquitectura densa reducida](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.12.png)

De manera similar, las curvas de pérdida en la Figura 6-13 muestran resultados similares, pero con menos irregularidades.

![Figura 6-13. Resultados de pérdida con arquitectura densa reducida](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.13.png)

#### Uso de dropout
Una técnica común para reducir el sobreajuste es agregar dropout a una red neuronal densa. Exploramos esta técnica en redes neuronales convolucionales en el Capítulo 3. Es tentador implementarla directamente para observar sus efectos en el sobreajuste, pero en este caso quise esperar hasta que se abordaran el tamaño del vocabulario, el tamaño del embedding y la complejidad de la arquitectura. Estos cambios suelen tener un impacto mucho mayor que el uso de dropout, y ya hemos visto resultados positivos.

Ahora que nuestra arquitectura ha sido simplificada para tener solo ocho neuronas en la capa densa intermedia, el efecto del dropout puede ser minimizado, pero exploremos de todas formas. Aquí está el código actualizado para la arquitectura del modelo, agregando un dropout de 0.25 (lo que equivale a dos de nuestras ocho neuronas):

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

La Figura 6-14 muestra los resultados de la precisión al entrenar durante cien épocas.

Esta vez, vemos que la precisión en el conjunto de entrenamiento está subiendo por encima de su umbral anterior, mientras que la precisión en el conjunto de validación disminuye lentamente. Esto indica que estamos entrando nuevamente en un territorio de sobreajuste. Esto se confirma explorando las curvas de pérdida en la Figura 6-15.

![Figura 6-14. Precisión con dropout añadido](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.14.png)

![Figura 6-15. Pérdida con dropout añadido](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.15.png)

Aquí se puede observar que el modelo está volviendo a su patrón anterior de aumento en la pérdida de validación con el tiempo. No es tan grave como antes, pero va en la dirección incorrecta.

En este caso, con muy pocas neuronas, probablemente no fue buena idea introducir dropout. Aun así, es útil tener esta herramienta en tu arsenal, así que asegúrate de tenerla en cuenta para arquitecturas más sofisticadas que esta.

#### Uso de regularización
La regularización es una técnica que ayuda a prevenir el sobreajuste al reducir la polarización de los pesos. Si los pesos en algunas neuronas son demasiado altos, la regularización los penaliza. En términos generales, existen dos tipos de regularización: L1 y L2.

- Regularización L1: también llamada lasso (operador de reducción y selección absoluta mínima). Ayuda a ignorar los pesos que son cero o cercanos a cero al calcular un resultado en una capa.

- Regularización L2: conocida como regresión ridge porque separa valores tomando sus cuadrados. Esto tiende a amplificar las diferencias entre valores distintos de cero y los que son cero o cercanos a cero, creando un efecto de cresta.
Ambos enfoques pueden combinarse en lo que a veces se llama regularización elástica.

Para problemas de NLP como el que estamos considerando, se utiliza más comúnmente L2. Puede añadirse como un atributo en la capa Dense usando la propiedad kernel_regularizers y toma un valor de punto flotante como factor de regularización. Este es otro hiperparámetro que puedes experimentar para mejorar tu modelo.

Aquí tienes un ejemplo:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu', 
    kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

El impacto de añadir regularización en un modelo simple como este no es particularmente grande, pero suaviza un poco nuestra pérdida de entrenamiento y validación. Quizás sea excesivo para este escenario, pero, al igual que el dropout, es buena idea entender cómo usar la regularización para evitar que tu modelo se especialice demasiado.

#### Otras consideraciones de optimización
Si bien las modificaciones que hemos hecho han mejorado mucho el modelo con menos sobreajuste, hay otros hiperparámetros que puedes experimentar. Por ejemplo, elegimos un máximo de longitud de oración de cien palabras, pero esto fue totalmente arbitrario y probablemente no sea óptimo. Es una buena idea explorar el corpus y determinar una longitud de oración más adecuada. Aquí hay un fragmento de código que analiza las oraciones y grafica las longitudes de cada una, ordenadas de menor a mayor:

```python
xs=[]
ys=[]
current_item=1
for item in sentences:
    xs.append(current_item)
    current_item=current_item+1
    ys.append(len(item))
newys = sorted(ys)

import matplotlib.pyplot as plt
plt.plot(xs,newys)
plt.show()
```

Los resultados de esto se muestran en la Figura 6-16.

![Figura 6-16. Exploración de la longitud de las oraciones](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.16.png)

Menos de 200 oraciones en el corpus total de más de 26,000 tienen una longitud de 100 palabras o más. Por lo tanto, al elegir este valor como la longitud máxima, estamos introduciendo mucho padding innecesario, lo que afecta el rendimiento del modelo. Reducirlo a 85 palabras aún conservaría 26,000 oraciones (más del 99%) sin padding alguno.

### Usando el modelo para clasificar una oración
Ahora que has creado el modelo, lo has entrenado y optimizado para eliminar muchos de los problemas que causaban el sobreajuste, el siguiente paso es ejecutar el modelo e inspeccionar sus resultados. Para hacerlo, crea un arreglo de nuevas oraciones. Por ejemplo:

```python
sentences = ["granny starting to fear spiders in the garden might be real",  
"game of thrones season finale showing this sunday night",  
"TensorFlow book will be a best seller"]
```

Luego, estas oraciones pueden ser codificadas utilizando el mismo tokenizador que se usó al crear el vocabulario para el entrenamiento. Es importante usarlo porque tiene los tokens de las palabras con las que la red fue entrenada.

```python
sequences = tokenizer.texts_to_sequences(sentences)  
print(sequences)
```

La salida del print será las secuencias para las oraciones anteriores:

```python
[[1, 816, 1, 691, 1, 1, 1, 1, 300, 1, 90],  
[111, 1, 1044, 173, 1, 1, 1, 1463, 181],  
[1, 234, 7, 1, 1, 46, 1]]
```

Hay muchos tokens "1" (“< OOV >”), porque palabras funcionales como “in” y “the” han sido eliminadas del diccionario, y palabras como “granny” y “spiders” no aparecen en el diccionario.

Antes de que las secuencias puedan ser pasadas al modelo, necesitarán tener la forma que el modelo espera, es decir, la longitud deseada. Puedes hacer esto con pad_sequences de la misma manera que lo hiciste al entrenar el modelo:

```python
padded = pad_sequences(sequences, maxlen=max_length,  
padding=padding_type, truncating=trunc_type)  
print(padded)
```

Esto generará las oraciones como secuencias de longitud 100, por lo que la salida para la primera secuencia será:

```python
[   1  816    1  691    1    1    1    1  300    1   90    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0    0    0    0    0    0    0    0    0    0    0    0    0  
    0    0]
```

¡Fue una oración muy corta!

Ahora que las oraciones han sido tokenizadas y rellenadas para ajustarse a las expectativas del modelo respecto a las dimensiones de entrada, es momento de pasarlas al modelo y obtener predicciones. Esto es tan sencillo como:

```python
print(model.predict(padded))
```

Los resultados serán devueltos como una lista e impresos, con valores altos indicando probabilidad de sarcasmo. Aquí están los resultados para nuestras oraciones de ejemplo:

```python
[[0.7194135 ]  
 [0.02041999]  
 [0.13156283]]
```

El puntaje alto para la primera oración (“granny starting to fear spiders in the garden might be real”), a pesar de tener muchas palabras funcionales y estar rellena con muchos ceros, indica que hay un alto nivel de sarcasmo aquí. Las otras dos oraciones obtuvieron puntajes mucho más bajos, indicando una menor probabilidad de sarcasmo.

## Visualizando los embeddings
Para visualizar los embeddings puedes usar una herramienta llamada Embedding Projector. Viene precargada con muchos conjuntos de datos existentes, pero en esta sección verás cómo tomar los datos del modelo que acabas de entrenar y visualizarlos usando esta herramienta.

Primero, necesitarás una función para invertir el índice de palabras. Actualmente tiene la palabra como token y la clave como valor, pero esto debe ser invertido para tener los valores de palabras a graficar en el proyector. Aquí está el código para hacerlo:

```python
reverse_word_index = dict([(value, key)  
for (key, value) in word_index.items()])
```

También necesitarás extraer los pesos de los vectores en los embeddings:

```python
e = model.layers[0]  
weights = e.get_weights()[0]  
print(weights.shape)
```

La salida de esto será (2000, 7) si seguiste las optimizaciones en este capítulo. Usamos un vocabulario de 2000 palabras y 7 dimensiones para el embedding. Si quieres explorar una palabra y los detalles de su vector, puedes hacerlo con código como este:

```python
print(reverse_word_index[2])  
print(weights[2])
```

Esto producirá la siguiente salida:

```python
new  
[ 0.8091359   0.54640186 -0.9058702  -0.94764805 -0.8809764  -0.70225513  
  0.86525863]
```

Entonces, la palabra “new” está representada por un vector con esos siete coeficientes en sus ejes.

El Embedding Projector utiliza dos archivos de valores separados por tabulaciones (TSV), uno para las dimensiones del vector y otro para los metadatos. Este código los generará por ti:

```python
import io  

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')  
out_m = io.open('meta.tsv', 'w', encoding='utf-8')  
for word_num in range(1, vocab_size):  
    word = reverse_word_index[word_num]  
    embeddings = weights[word_num]  
    out_m.write(word + "\n")  
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")  

out_v.close()  
out_m.close()
```

Si estás usando Google Colab, puedes descargar los archivos TSV con el siguiente código o desde el panel de archivos:

```python
try:  
    from google.colab import files  
except ImportError:  
    pass  
else:  
    files.download('vecs.tsv')  
    files.download('meta.tsv')
```

Una vez que los tengas, puedes presionar el botón Load en el proyector para visualizar los embeddings, como se muestra en la Figura 6-17.

Usa los archivos TSV de vectores y meta donde se recomienda en el diálogo resultante, y luego haz clic en Sphereize Data en el proyector. Esto hará que las palabras se agrupen en una esfera y te dará una clara visualización de la naturaleza binaria de este clasificador. Solo ha sido entrenado en oraciones sarcásticas y no sarcásticas, por lo que las palabras tienden a agruparse hacia una etiqueta u otra (Figura 6-18).

![Figura 6-17. Uso del proyector de incrustaciones](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.17.png)

![Figura 6-18. Visualización de las incrustaciones de sarcasmo](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.18.png)

Las capturas de pantalla no le hacen justicia; ¡deberías probarlo tú mismo! Puedes rotar la esfera central y explorar las palabras en cada “polo” para ver el impacto que tienen en la clasificación general. También puedes seleccionar palabras y mostrar palabras relacionadas en el panel derecho. Experimenta y diviértete.

## Usar Embeddings Preentrenados de TensorFlow Hub
Una alternativa a entrenar tus propios embeddings es usar aquellos que ya han sido preentrenados y empaquetados en capas de Keras para ti. Hay muchos de estos en TensorFlow Hub que puedes explorar. Algo a tener en cuenta es que también pueden contener la lógica de tokenización por ti, por lo que no necesitas manejar la tokenización, secuenciación y padding como lo has estado haciendo hasta ahora.

TensorFlow Hub está preinstalado en Google Colab, por lo que el código en este capítulo funcionará tal cual. Si quieres instalarlo como dependencia en tu máquina, necesitarás seguir las instrucciones para instalar la última versión.

Por ejemplo, con los datos de Sarcasmo, en lugar de toda la lógica para la tokenización, gestión del vocabulario, secuenciación, padding, etc., podrías simplemente hacer algo como esto una vez tengas tu conjunto completo de oraciones y etiquetas. Primero, divídelos en conjuntos de entrenamiento y prueba:

```python
training_size = 24000  
training_sentences = sentences[0:training_size]  
testing_sentences = sentences[training_size:]  
training_labels = labels[0:training_size]  
testing_labels = labels[training_size:]  
```

Una vez que tengas esto, puedes descargar una capa preentrenada desde TensorFlow Hub de esta manera:

```python
import tensorflow_hub as hub 

hub_layer = hub.KerasLayer(  
    "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",  
    output_shape=[20], input_shape=[],  
    dtype=tf.string, trainable=False  
)  
```

Esto toma los embeddings del conjunto de datos Swivel, entrenado en 130 GB de Google News. Usar esta capa codificará tus oraciones, las tokenizará, usará las palabras de ellas con los embeddings aprendidos como parte de Swivel y luego codificará tus oraciones en un único embedding. Vale la pena recordar esa última parte. La técnica que hemos estado usando hasta ahora es simplemente usar las codificaciones de palabras y clasificar el contenido basado en todas ellas. Cuando usas una capa como esta, obtienes toda la oración agregada en una nueva codificación.

Luego, puedes crear una arquitectura de modelo utilizando esta capa en lugar de una de embeddings. Aquí hay un modelo simple que la utiliza:

```python
model = tf.keras.Sequential([  
    hub_layer,  
    tf.keras.layers.Dense(16, activation='relu'),  
    tf.keras.layers.Dense(1, activation='sigmoid')  
])  

adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,  
                                beta_2=0.999, amsgrad=False)  
                                
model.compile(loss='binary_crossentropy', optimizer=adam,  
              metrics=['accuracy'])  
```

Este modelo alcanzará rápidamente su precisión máxima durante el entrenamiento y no tendrá tanto sobreajuste como vimos anteriormente. La precisión durante 50 épocas muestra que el entrenamiento y la validación están muy alineados entre sí (Figura 6-19).

![Figura 6-19. Métricas de precisión utilizando embeddings Swivel](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.19.png)

Los valores de pérdida también están alineados, mostrando que el ajuste es muy adecuado (Figura 6-20).

![Figura 6-20. Métricas de pérdida utilizando embeddings Swivel](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure6.20.png)

Sin embargo, vale la pena señalar que la precisión general (alrededor del 67%) es bastante baja, considerando que lanzar una moneda tendría un 50% de probabilidad de acertar. Esto se debe a la codificación de todos los embeddings basados en palabras en uno basado en oraciones. En el caso de los titulares sarcásticos, parece que las palabras individuales pueden tener un gran efecto en la clasificación (ver Figura 6-18). Por lo tanto, aunque usar embeddings preentrenados puede acelerar mucho el entrenamiento y reducir el sobreajuste, también debes entender para qué son útiles y que no siempre pueden ser lo mejor para tu escenario.

## Resumen
En este capítulo, construiste tu primer modelo para comprender el sentimiento en texto. Lo hizo tomando el texto tokenizado del Capítulo 5 y mapeándolo a vectores. Luego, utilizando retropropagación, aprendió la “dirección” adecuada para cada vector basado en la etiqueta de la oración que lo contenía. Finalmente, pudo usar todos los vectores de una colección de palabras para construir una idea del sentimiento dentro de la oración. También exploraste formas de optimizar tu modelo para evitar el sobreajuste y viste una visualización interesante de los vectores finales que representan tus palabras.

Aunque esta fue una buena forma de clasificar oraciones, simplemente trató cada oración como un conjunto de palabras. No hubo una secuencia inherente involucrada, y dado que el orden de aparición de las palabras es muy importante para determinar el significado real de una oración, es una buena idea ver si podemos mejorar nuestros modelos considerando la secuencia. Exploraremos eso en el próximo capítulo con la introducción de un nuevo tipo de capa: una capa recurrente, que es la base de las redes neuronales recurrentes. También verás otro embedding preentrenado llamado GloVe, que te permitirá usar embeddings basados en palabras en un escenario de aprendizaje por transferencia.