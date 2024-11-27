# Introducción a la Visión por Computadora
El capítulo anterior introdujo los conceptos básicos de cómo funciona el aprendizaje automático. Viste cómo comenzar a programar utilizando redes neuronales para asociar datos con etiquetas, y a partir de ahí, cómo inferir las reglas que pueden ser utilizadas para distinguir elementos. Un paso lógico siguiente es aplicar estos conceptos a la visión por computadora, donde tendremos un modelo que aprenderá a reconocer contenido en imágenes para que pueda “ver” lo que hay en ellas. En este capítulo trabajarás con un conjunto de datos popular de artículos de ropa y construirás un modelo que pueda diferenciarlos, logrando así “ver” la diferencia entre distintos tipos de ropa.

## Reconociendo Artículos de Ropa
Para nuestro primer ejemplo, consideremos lo que se necesita para reconocer artículos de ropa en una imagen. Considera, por ejemplo, los artículos en la Figura 2-1.

![Figura 2-1. Ejemplos de ropa](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure2.1.png)

Aquí hay varios artículos de ropa diferentes, y puedes reconocerlos. Entiendes qué es una camisa, un abrigo o un vestido. Pero, ¿cómo explicarías esto a alguien que nunca ha visto ropa? ¿Y qué hay de un zapato? Hay dos zapatos en esta imagen, pero ¿cómo describirías eso a alguien? Este es otro caso en el que la programación basada en reglas, de la que hablamos en el Capítulo 1, puede fallar. A veces, simplemente no es viable describir algo con reglas.

Por supuesto, la visión por computadora no es una excepción. Pero considera cómo aprendiste a reconocer todos estos artículos: viendo muchos ejemplos diferentes y adquiriendo experiencia sobre cómo se usan. ¿Podemos hacer lo mismo con una computadora? La respuesta es sí, pero con limitaciones. Veamos un primer ejemplo de cómo enseñar a una computadora a reconocer artículos de ropa, utilizando un conjunto de datos bien conocido llamado Fashion MNIST.

## Los Datos: Fashion MNIST
Uno de los conjuntos de datos fundamentales para aprender y evaluar algoritmos es la base de datos Modified National Institute of Standards and Technology (MNIST), creada por Yann LeCun, Corinna Cortes y Christopher Burges. Este conjunto de datos está compuesto por imágenes de 70,000 dígitos escritos a mano del 0 al 9. Las imágenes tienen un tamaño de 28 × 28 y son en escala de grises.

Fashion MNIST está diseñado para ser un reemplazo directo de MNIST, con el mismo número de registros, las mismas dimensiones de imagen y el mismo número de clases. Pero, en lugar de imágenes de los dígitos del 0 al 9, Fashion MNIST contiene imágenes de 10 tipos diferentes de ropa. Puedes ver un ejemplo del contenido del conjunto de datos en la Figura 2-2. Aquí, se dedican tres líneas a cada tipo de artículo de ropa.

![Figura 2-2. Explorando el conjunto de datos Fashion MNIST](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure2.2.png)

Incluye una buena variedad de ropa, como camisas, pantalones, vestidos y muchos tipos de zapatos. Como puedes notar, es monocromático, por lo que cada imagen consiste en un cierto número de píxeles con valores entre 0 y 255. Esto hace que el conjunto de datos sea más sencillo de manejar.

Puedes ver un primer plano de una imagen específica del conjunto de datos en la Figura 2-3.

![Figura 2-3. Primer plano de una imagen en el conjunto de datos Fashion MNIST](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure2.3.png)

Como cualquier imagen, es una cuadrícula rectangular de píxeles. En este caso, el tamaño de la cuadrícula es 28 × 28, y cada píxel es simplemente un valor entre 0 y 255, como se mencionó anteriormente. Ahora veamos cómo puedes usar estos valores de píxeles con las funciones que vimos previamente.

## Neuronas para Visión
En el Capítulo 1, viste un escenario muy simple donde a una máquina se le daba un conjunto de valores X y Y, y aprendía que la relación entre estos era Y = 2X – 1. Esto se logró usando una red neuronal muy sencilla con una capa y una neurona.

Si lo representaras visualmente, podría verse como la Figura 2-4.

Cada una de nuestras imágenes es un conjunto de 784 valores (28 × 28) entre 0 y 255. Estos pueden ser nuestro X. Sabemos que tenemos 10 tipos diferentes de imágenes en nuestro conjunto de datos, así que consideremos que estos son nuestro Y. Ahora queremos aprender cómo se ve la función donde Y es una función de X.

![Figura 2-4. Una sola neurona aprendiendo una relación lineal](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure2.4.png)

Dado que tenemos 784 valores X por imagen, y nuestro Y estará entre 0 y 9, está bastante claro que no podemos usar Y = mX + c como hicimos antes.

Pero lo que podemos hacer es tener varias neuronas trabajando juntas. Cada una de estas aprenderá parámetros, y cuando tenemos una función combinada de todos estos parámetros trabajando juntos, podemos ver si podemos ajustar ese patrón a nuestra respuesta deseada (Figura 2-5).

![Figura 2-5. Extendiendo nuestro patrón para un ejemplo más complejo](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure2.5.png)

Los cuadros en la parte superior de este diagrama pueden considerarse los píxeles de la imagen, o nuestros valores X. Cuando entrenamos la red neuronal, cargamos estos valores en una capa de neuronas. La Figura 2-5 muestra que solo se cargan en la primera neurona, pero en realidad los valores se cargan en todas ellas. Considera que el peso y sesgo (m y c) de cada neurona están inicializados aleatoriamente. Luego, al sumar los valores de la salida de cada neurona, obtenemos un valor. Esto se hará para cada neurona en la capa de salida, de modo que la neurona 0 contendrá el valor de la probabilidad de que los píxeles correspondan a la etiqueta 0, la neurona 1 a la etiqueta 1, y así sucesivamente.

Con el tiempo, queremos que ese valor coincida con la salida deseada—que para esta imagen podemos ver que es el número 9, la etiqueta para la bota mostrada en la Figura 2-3. En otras palabras, esta neurona debería tener el valor más grande de todas las neuronas de salida.

Dado que hay 10 etiquetas, una inicialización aleatoria debería obtener la respuesta correcta alrededor del 10% de las veces. A partir de ahí, la función de pérdida y el optimizador pueden hacer su trabajo, iteración tras iteración, para ajustar los parámetros internos de cada neurona y mejorar ese 10%.

Y así, con el tiempo, la computadora aprenderá a "ver" qué hace que un zapato sea un zapato o un vestido sea un vestido.

## Diseñando la Red Neuronal
Ahora exploremos cómo se ve esto en código. Primero, veamos el diseño de la red neuronal mostrado en la Figura 2-5:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

Si recuerdas, en el Capítulo 1 usamos un modelo Sequential para especificar que teníamos varias capas. En ese caso, solo había una capa, pero aquí tenemos múltiples capas.

La primera, Flatten, no es una capa de neuronas, sino una especificación de la capa de entrada. Nuestros datos de entrada son imágenes de 28 × 28, pero queremos tratarlas como una serie de valores numéricos, como los cuadros grises en la parte superior de la Figura 2-5. Flatten toma ese valor "cuadrado" (un arreglo 2D) y lo convierte en una línea (un arreglo 1D).

La siguiente, Dense, es una capa de neuronas, y estamos especificando que queremos 128 de ellas. Esta es la capa intermedia mostrada en la Figura 2-5. A menudo escucharás que tales capas se describen como capas ocultas. Las capas que están entre las entradas y las salidas no son visibles para quien llama al modelo, por lo que se usa el término "ocultas" para describirlas. Estamos pidiendo que las 128 neuronas tengan sus parámetros internos inicializados aleatoriamente.

Una pregunta común en este punto es: “¿Por qué 128?”. Esto es completamente arbitrario: no hay una regla fija para determinar cuántas neuronas usar. Al diseñar las capas, debes elegir un número adecuado de valores que permitan a tu modelo aprender. Más neuronas significan que el modelo se ejecutará más lentamente, ya que tiene que aprender más parámetros. También, más neuronas podrían llevar a una red que sea excelente reconociendo los datos de entrenamiento, pero no tan buena reconociendo datos nuevos (esto se conoce como sobreajuste, y lo discutiremos más adelante en este capítulo). Por otro lado, menos neuronas podrían hacer que el modelo no tenga suficientes parámetros para aprender.

Encontrar los valores correctos toma experimentación con el tiempo. Este proceso generalmente se llama ajuste de hiperparámetros. En aprendizaje automático, un hiperparámetro es un valor usado para controlar el entrenamiento, a diferencia de los valores internos de las neuronas que se entrenan/aprenden, los cuales se denominan parámetros.

También notarás que se especifica una función de activación en esa capa. La función de activación es un código que se ejecutará en cada neurona de la capa. TensorFlow admite varias de ellas, pero una muy común en capas intermedias es relu, que significa "unidad lineal rectificada". Es una función simple que solo devuelve un valor si es mayor que 0. En este caso, no queremos valores negativos que se pasen a la siguiente capa y potencialmente afecten la función de suma, por lo que en lugar de escribir mucho código con condicionales, podemos simplemente activar la capa con relu.

Finalmente, hay otra capa Dense, que es la capa de salida. Esta tiene 10 neuronas, porque tenemos 10 clases. Cada una de estas neuronas terminará con una probabilidad de que los píxeles de entrada coincidan con esa clase, así que nuestro trabajo es determinar cuál tiene el valor más alto. Podríamos recorrerlas en un bucle para elegir ese valor, pero la función de activación softmax hace eso por nosotros.

Entonces, cuando entrenamos nuestra red neuronal, el objetivo es que podamos alimentar un arreglo de píxeles de 28 × 28 y las neuronas en la capa intermedia tendrán pesos y sesgos (valores m y c) que, al combinarse, harán coincidir esos píxeles con uno de los 10 valores de salida.

## Código Completo
Ahora que hemos explorado la arquitectura de la red neuronal, veamos el código completo para entrenar una con los datos de Fashion MNIST:

```python
import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
```

Ahora expliquemos esto paso a paso. Primero, tenemos un atajo práctico para acceder a los datos:

```python
data = tf.keras.datasets.fashion_mnist
```

Keras tiene una serie de conjuntos de datos integrados que puedes acceder con una sola línea de código como esta. En este caso, no tienes que gestionar la descarga de las 70,000 imágenes, dividirlas en conjuntos de entrenamiento y prueba, etc. Todo esto se realiza con una línea de código. Esta metodología se ha mejorado con una API llamada TensorFlow Datasets, pero para los propósitos de estos primeros capítulos, y para reducir la cantidad de conceptos nuevos que necesitas aprender, simplemente usaremos tf.keras.datasets.

Podemos llamar al método load_data para devolver nuestros conjuntos de entrenamiento y prueba de esta manera:

```python
(training_images, training_labels), (test_images, test_labels) = data.load_data()
```

Fashion MNIST está diseñado para tener 60,000 imágenes de entrenamiento y 10,000 imágenes de prueba. Entonces, la devolución de data.load_data te dará un arreglo de 60,000 arreglos de píxeles de 28 × 28 llamado training_images y un arreglo de 60,000 valores (0–9) llamado training_labels. De manera similar, el arreglo test_images contendrá 10,000 arreglos de píxeles de 28 × 28, y el arreglo test_labels contendrá 10,000 valores entre 0 y 9.

Nuestro trabajo será ajustar las imágenes de entrenamiento a las etiquetas de entrenamiento de una manera similar a como ajustamos Y a X en el Capítulo 1. Reservaremos las imágenes y etiquetas de prueba para que la red no las vea mientras entrena. Estas se pueden usar para probar la eficacia de la red con datos no vistos previamente.

Las siguientes líneas de código pueden parecer un poco inusuales:

```python
training_images  = training_images / 255.0
test_images = test_images / 255.0
```

Python te permite realizar una operación en todo el arreglo con esta notación. Recuerda que todos los píxeles en nuestras imágenes son en escala de grises, con valores entre 0 y 255. Dividir entre 255 asegura que cada píxel sea representado por un número entre 0 y 1. Este proceso se llama normalización de la imagen.

Las matemáticas detrás de por qué los datos normalizados son mejores para entrenar redes neuronales están fuera del alcance de este libro, pero ten en cuenta que normalizar mejorará el rendimiento. A menudo, tu red no aprenderá y tendrá errores masivos cuando trate con datos no normalizados. El ejemplo Y = 2X – 1 del Capítulo 1 no requería que los datos fueran normalizados porque era muy simple, pero para divertirte, intenta entrenarlo con diferentes valores de X e Y donde X sea mucho más grande y verás que rápidamente falla.

Luego, definimos la red neuronal que compone nuestro modelo, como discutimos anteriormente:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

Cuando compilamos nuestro modelo, especificamos la función de pérdida y el optimizador como antes:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

La función de pérdida en este caso se llama entropía cruzada categórica dispersa, y es una de las funciones de pérdida integradas en TensorFlow. Nuevamente, elegir qué función de pérdida usar es un arte en sí mismo, y con el tiempo aprenderás cuáles son las mejores para usar en diferentes escenarios. Una gran diferencia entre este modelo y el que creamos en el Capítulo 1 es que, en lugar de intentar predecir un solo número, aquí estamos eligiendo una categoría. Nuestro artículo de ropa pertenecerá a una de las 10 categorías de ropa, y, por lo tanto, usar una función de pérdida categórica es el camino a seguir. La entropía cruzada categórica dispersa es una buena opción.

Lo mismo aplica para elegir un optimizador. El optimizador Adam es una evolución del optimizador de gradiente descendente estocástico (SGD) que usamos en el Capítulo 1, y se ha demostrado que es más rápido y eficiente. Dado que estamos manejando 60,000 imágenes de entrenamiento, cualquier mejora de rendimiento que podamos obtener será útil, por lo que se eligió este.

Es posible que notes que también hay una nueva línea que especifica las métricas que queremos reportar. Aquí, queremos informar sobre la precisión de la red mientras estamos entrenando. El ejemplo simple en el Capítulo 1 solo informaba sobre la pérdida, y nosotros interpretábamos que la red estaba aprendiendo al observar cómo se reducía la pérdida. En este caso, es más útil ver cómo la red está aprendiendo al observar la precisión, que devolverá con qué frecuencia coincidió correctamente los píxeles de entrada con la etiqueta de salida.

A continuación, entrenaremos la red ajustando las imágenes de entrenamiento a las etiquetas de entrenamiento durante cinco épocas:

```python
model.fit(training_images, training_labels, epochs=5)
```

Finalmente, podemos hacer algo nuevo: evaluar el modelo usando una sola línea de código. Tenemos un conjunto de 10,000 imágenes y etiquetas para pruebas, y podemos pasarlas al modelo entrenado para que prediga lo que piensa que es cada imagen, compare eso con su etiqueta real y sume los resultados:

```python
model.evaluate(test_images, test_labels)
```

## Entrenando la Red Neuronal
Ejecuta el código y verás cómo la red entrena época por época. Después de ejecutar el entrenamiento, verás algo al final que se parece a esto:

```python
58016/60000 [=====>.] - ETA: 0s - loss: 0.2941 - accuracy: 0.8907  
59552/60000 [=====>.] - ETA: 0s - loss: 0.2943 - accuracy: 0.8906  
60000/60000 [] - 2s 34us/sample - loss: 0.2940 - accuracy: 0.8906 
```

Nota que ahora se está reportando la precisión. Entonces, en este caso, usando los datos de entrenamiento, nuestro modelo terminó con una precisión de aproximadamente 89% después de solo cinco épocas.

Pero, ¿qué pasa con los datos de prueba? Los resultados de model.evaluate en nuestros datos de prueba se verán algo así:

```python
10000/1 [====] - 0s 30us/sample - loss: 0.2521 - accuracy: 0.8736  
```

En este caso, la precisión del modelo fue de 87.36%, lo cual no está mal considerando que solo lo entrenamos durante cinco épocas.

Probablemente te estés preguntando por qué la precisión es menor para los datos de prueba que para los datos de entrenamiento. Esto se observa muy comúnmente, y cuando lo piensas, tiene sentido: la red neuronal realmente solo sabe cómo emparejar las entradas en las que ha sido entrenada con las salidas correspondientes a esos valores. Nuestra esperanza es que, con suficientes datos, sea capaz de generalizar a partir de los ejemplos que ha visto, "aprendiendo" cómo se ve un zapato o un vestido. Pero siempre habrá ejemplos de elementos que no ha visto y que son lo suficientemente diferentes como para confundirla.

Por ejemplo, si creciste viendo únicamente zapatillas deportivas, y eso es lo que te parece un zapato, cuando veas por primera vez un tacón alto podrías estar un poco confundido. Desde tu experiencia, probablemente sea un zapato, pero no lo sabes con certeza. Este es un concepto similar.

## Explorando la Salida del Modelo
Ahora que el modelo ha sido entrenado y tenemos una buena idea de su precisión utilizando el conjunto de prueba, exploremos un poco:

```python
classifications = model.predict(test_images)  
print(classifications[0])  
print(test_labels[0])  
```

Obtendremos un conjunto de clasificaciones al pasar las imágenes de prueba a model.predict. Luego, veamos qué obtenemos si imprimimos la primera de las clasificaciones y la comparamos con la etiqueta de prueba:

```python
[1.9177722e-05 1.9856788e-07 6.3756357e-07 7.1702580e-08 5.5287035e-07  
1.2249852e-02 6.0708484e-05 7.3229447e-02 8.3050705e-05 9.1435629e-01]  
9  
```

Notarás que la clasificación nos devuelve un arreglo de valores. Estos son los valores de las 10 neuronas de salida. La etiqueta es la etiqueta real para el artículo de ropa, en este caso, 9. Mira el arreglo: verás que algunos de los valores son muy pequeños, y el último (índice del arreglo 9) es el más grande con diferencia. Estas son las probabilidades de que la imagen coincida con la etiqueta en ese índice en particular. Entonces, lo que está reportando la red neuronal es que hay un 91.4% de probabilidad de que el artículo de ropa en el índice 0 sea la etiqueta 9. Sabemos que es la etiqueta 9, así que acertó.

Prueba con algunos valores diferentes y mira si encuentras algún caso en el que el modelo se equivoque.

## Entrenando por Más Tiempo—Descubriendo el Sobreajuste
En este caso, entrenamos solo por cinco épocas. Es decir, pasamos por todo el bucle de entrenamiento en el que las neuronas se inicializan aleatoriamente, se comparan con sus etiquetas, se mide el desempeño mediante la función de pérdida, y luego se actualizan con el optimizador cinco veces. Y los resultados que obtuvimos fueron bastante buenos: 89% de precisión en el conjunto de entrenamiento y 87% en el conjunto de prueba.

Entonces, ¿qué pasa si entrenamos por más tiempo?

Intenta actualizarlo para entrenar durante 50 épocas en lugar de 5. En mi caso, obtuve estas cifras de precisión en el conjunto de entrenamiento:

```python
58112/60000 [==>.] - ETA: 0s - loss: 0.0983 - accuracy: 0.9627  
59520/60000 [==>.] - ETA: 0s - loss: 0.0987 - accuracy: 0.9627  
60000/60000 [====] - 2s 35us/sample - loss: 0.0986 - accuracy: 0.9627  
```

Esto es particularmente emocionante porque estamos logrando mucho más: 96.27% de precisión. Para el conjunto de prueba, alcanzamos 88.6%:

```python
[====] - 0s 30us/sample - loss: 0.3870 - accuracy: 0.8860  
```

Entonces, logramos una gran mejora en el conjunto de entrenamiento y una más pequeña en el conjunto de prueba.

Esto podría sugerir que entrenar nuestra red por mucho más tiempo llevaría a resultados mucho mejores, pero ese no es siempre el caso. La red está funcionando mucho mejor con los datos de entrenamiento, pero no necesariamente es un mejor modelo. De hecho, la divergencia en los números de precisión muestra que se ha vuelto demasiado especializada en los datos de entrenamiento, un proceso llamado comúnmente sobreajuste (overfitting). A medida que construyas más redes neuronales, este es un aspecto que deberás tener en cuenta, y a medida que avances en este libro aprenderás varias técnicas para evitarlo.

## Deteniendo el Entrenamiento
En cada uno de los casos anteriores, hemos codificado de forma rígida el número de épocas para las que entrenamos. Si bien esto funciona, podríamos querer entrenar hasta alcanzar la precisión deseada en lugar de estar probando constantemente diferentes números de épocas, entrenando y reentrenando hasta llegar al valor deseado. Por ejemplo, si queremos entrenar hasta que el modelo alcance un 95% de precisión en el conjunto de entrenamiento, sin saber cuántas épocas tomará, ¿cómo podríamos lograrlo?

El enfoque más sencillo es usar un callback durante el entrenamiento. Veamos el código actualizado que utiliza callbacks:

```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
```

Veamos qué hemos cambiado aquí. Primero, creamos una nueva clase llamada myCallback. Esta toma tf.keras.callbacks.Callback como parámetro. Dentro de ella, definimos la función on_epoch_end, que nos dará detalles sobre los registros (logs) de esta época. En estos registros hay un valor de precisión (accuracy), por lo que todo lo que tenemos que hacer es verificar si es mayor que 0.95 (o 95%); si lo es, podemos detener el entrenamiento diciendo self.model.stop_training = True.

Una vez que hemos especificado esto, creamos un objeto callbacks que será una instancia de la función myCallback.

Ahora mira la instrucción model.fit. Verás que la actualicé para entrenar durante 50 épocas y luego añadí un parámetro callbacks. A este le paso el objeto callbacks. Durante el entrenamiento, al final de cada época, se llamará a la función de callback. Así que al final de cada época se verificará, y después de aproximadamente 34 épocas verás que tu entrenamiento terminará porque se alcanzó el 95% de precisión (tu número puede ser ligeramente diferente debido a la inicialización aleatoria inicial, pero probablemente será bastante cercano a 34):

```python
56896/60000 [====>..] - ETA: 0s - loss: 0.1309 - accuracy: 0.9500  
58144/60000 [====>.] - ETA: 0s - loss: 0.1308 - accuracy: 0.9502  
59424/60000 [====>.] - ETA: 0s - loss: 0.1308 - accuracy: 0.9502  
Reached 95% accuracy so cancelling training!  
```

## Resumen
En el capítulo 1 aprendiste cómo el aprendizaje automático se basa en ajustar características a etiquetas mediante un sofisticado reconocimiento de patrones con una red neuronal. En este capítulo llevaste eso al siguiente nivel, yendo más allá de una sola neurona y aprendiste cómo crear tu primera red neuronal para visión por computadora (bastante básica).

Era algo limitada debido a los datos. Todas las imágenes eran de 28 × 28 en escala de grises, con el artículo de ropa centrado en el marco. Es un buen comienzo, pero es un escenario muy controlado. Para mejorar en visión, podríamos necesitar que la computadora aprenda características de una imagen en lugar de simplemente los píxeles en bruto.

Podemos hacer esto con un proceso llamado convoluciones. Aprenderás cómo definir redes neuronales convolucionales para comprender el contenido de las imágenes en el próximo capítulo.
