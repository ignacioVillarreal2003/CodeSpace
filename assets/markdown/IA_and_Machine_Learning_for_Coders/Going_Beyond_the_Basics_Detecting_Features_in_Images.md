# Más Allá de lo Básico: Detectando Características en Imágenes
En el capítulo 2 aprendiste cómo iniciarte en la visión por computadora creando una simple red neuronal que correspondía los píxeles de entrada del conjunto de datos Fashion MNIST con 10 etiquetas, cada una representando un tipo (o clase) de ropa. Y aunque creaste una red que era bastante buena detectando tipos de ropa, había una clara desventaja.
Tu red neuronal fue entrenada con imágenes pequeñas en blanco y negro que contenían únicamente un único artículo de ropa, y ese artículo estaba centrado dentro de la imagen.

Para llevar el modelo al siguiente nivel, necesitas ser capaz de detectar características en las imágenes. Por ejemplo, en lugar de mirar únicamente los píxeles brutos de la imagen, ¿qué pasaría si tuviéramos una forma de filtrar las imágenes para reducirlas a sus elementos constitutivos? Hacer coincidir esos elementos, en lugar de los píxeles brutos, nos ayudaría a detectar los contenidos de las imágenes de forma más efectiva.

Considera el conjunto de datos Fashion MNIST que utilizamos en el último capítulo. Al detectar un zapato, la red neuronal pudo haberse activado por muchos píxeles oscuros agrupados en la parte inferior de la imagen, que percibía como la suela del zapato. Pero cuando el zapato ya no está centrado y llenando el marco, esta lógica deja de ser válida.

Un método para detectar características proviene de la fotografía y las metodologías de procesamiento de imágenes con las que podrías estar familiarizado. Si alguna vez has usado una herramienta como Photoshop o GIMP para enfocar una imagen, estás usando un filtro matemático que trabaja sobre los píxeles de la imagen. Otro término para estos filtros es convolución, y al usarlos en una red neuronal creas una red neuronal convolucional (CNN).

En este capítulo aprenderás cómo usar convoluciones para detectar características en una imagen. Luego profundizarás en la clasificación de imágenes basada en las características internas. Exploraremos la ampliación (augmentation) de imágenes para obtener más características y el aprendizaje por transferencia (transfer learning) para aprovechar características preexistentes aprendidas por otros. Finalmente, veremos brevemente cómo optimizar tus modelos utilizando dropouts.

## Convoluciones
Una convolución es simplemente un filtro de pesos que se utiliza para multiplicar un píxel con sus vecinos y obtener un nuevo valor para el píxel. Por ejemplo, considera la imagen de un botín del conjunto Fashion MNIST y los valores de píxeles que se muestran en la Figura 3-1.

![Figura 3-1. Botín con convolución](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.1.png)

Si miramos el píxel en el centro de la selección, podemos ver que tiene el valor 192 (recuerda que Fashion MNIST utiliza imágenes monocromáticas con valores de píxel entre 0 y 255). El píxel arriba y a la izquierda tiene el valor 0, el que está inmediatamente arriba tiene el valor 64, etc.

Si luego definimos un filtro en la misma cuadrícula de 3 × 3, como se muestra debajo de los valores originales, podemos transformar ese píxel calculando un nuevo valor para él. Esto se hace multiplicando el valor actual de cada píxel en la cuadrícula por el valor en la misma posición en la cuadrícula del filtro y sumando el total. Este total será el nuevo valor para el píxel actual. Luego repetimos esto para todos los píxeles de la imagen.

Entonces, en este caso, mientras el valor actual del píxel en el centro de la selección es 192, el nuevo valor después de aplicar el filtro será:

```python
new_val = (-1 * 0) + (0 * 64) + (-2 * 128) +  
          (0.5 * 48) + (4.5 * 192) + (-1.5 * 144) +  
          (1.5 * 142) + (2 * 226) + (-3 * 168)
```

Esto da como resultado 577, que será el nuevo valor para ese píxel. Repetir este proceso en todos los píxeles de la imagen nos dará una imagen filtrada.

Consideremos el impacto de aplicar un filtro en una imagen más complicada: la imagen ascent que viene incluida en SciPy para pruebas sencillas. Esta es una imagen en escala de grises de 512 × 512 que muestra a dos personas subiendo una escalera.

Usar un filtro con valores negativos en la izquierda, positivos en la derecha y ceros en el medio terminará eliminando la mayor parte de la información de la imagen, excepto por las líneas verticales, como puedes ver en la Figura 3-2.

![Figura 3-2. Usando un filtro para obtener líneas verticales](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.2.png)

De manera similar, un pequeño cambio en el filtro puede enfatizar las líneas horizontales, como se muestra en la Figura 3-3.

![Figura 3-3. Usando un filtro para obtener líneas horizontales](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.3.png)

Estos ejemplos también muestran que la cantidad de información en la imagen se reduce, por lo que podemos aprender un conjunto de filtros que reduzcan la imagen a características, y esas características pueden coincidir con las etiquetas como antes. Anteriormente, aprendimos parámetros que se usaban en las neuronas para hacer coincidir entradas con salidas. De manera similar, los mejores filtros para hacer coincidir entradas con salidas pueden aprenderse con el tiempo.

Cuando se combinan con pooling, podemos reducir la cantidad de información en la imagen mientras mantenemos las características. Exploraremos eso a continuación.

## Pooling
El pooling es el proceso de eliminar píxeles en tu imagen mientras mantienes la semántica del contenido dentro de la imagen. Se explica mejor visualmente. La Figura 3-4 muestra el concepto de max pooling.

![Figura 3-4. Demostrando max pooling](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.4.png)

En este caso, considera el recuadro a la izquierda como los píxeles en una imagen monocromática. Luego los agrupamos en matrices de 2 × 2, por lo que en este caso los 16 píxeles se agrupan en cuatro matrices de 2 × 2. Estas se llaman pools.

Luego seleccionamos el valor máximo de cada uno de los grupos y los volvemos a ensamblar en una nueva imagen. Así, los píxeles a la izquierda se reducen en un 75% (de 16 a 4), con el valor máximo de cada pool formando la nueva imagen.

La Figura 3-5 muestra la versión de ascent de la Figura 3-2, con las líneas verticales mejoradas, después de que se ha aplicado max pooling.

![Figura 3-5. Ascent después del filtro vertical y max pooling](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.5.png)

Observa cómo las características filtradas no solo se han mantenido, sino que se han mejorado aún más. Además, el tamaño de la imagen ha cambiado de 512 × 512 a 256 × 256, un cuarto del tamaño original.

> Existen otros enfoques para pooling, como min pooling, que toma el valor mínimo del píxel del pool, y average pooling, que toma el valor promedio general.

## Implementando Redes Neuronales Convolucionales
En el Capítulo 2 creaste una red neuronal que reconocía imágenes de moda. Para conveniencia, aquí está el código completo:

```python
import tensorflow as tf  

data = tf.keras.datasets.fashion_mnist  
(training_images, training_labels), (test_images, test_labels) = data.load_data()  

training_images = training_images / 255.0  
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
```

Para convertir esto en una red neuronal convolucional, simplemente usamos capas convolucionales en la definición del modelo. También agregaremos capas de pooling.

Para implementar una capa convolucional, usarás el tipo tf.keras.layers.Conv2D. Este acepta como parámetros el número de convoluciones a usar en la capa, el tamaño de las convoluciones, la función de activación, etc.

Por ejemplo, aquí tienes una capa convolucional usada como capa de entrada para una red neuronal:

```python
tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
```

En este caso, queremos que la capa aprenda 64 convoluciones. Estas se inicializarán aleatoriamente y, con el tiempo, aprenderán los valores del filtro que mejor se ajusten para asociar los valores de entrada con sus etiquetas. El (3, 3) indica el tamaño del filtro. Anteriormente mostré filtros 3 × 3, y eso es lo que estamos especificando aquí. Este es el tamaño de filtro más común; puedes cambiarlo como desees, pero normalmente verás un número impar de ejes como 5 × 5 o 7 × 7 debido a cómo los filtros eliminan píxeles de los bordes de la imagen, como verás más adelante.

Los parámetros de activación y input_shape son los mismos que antes. Como estamos usando Fashion MNIST en este ejemplo, la forma sigue siendo 28 × 28. Sin embargo, ten en cuenta que, dado que las capas Conv2D están diseñadas para imágenes multicolores, estamos especificando la tercera dimensión como 1, por lo que nuestra forma de entrada es 28 × 28 × 1. Las imágenes a color generalmente tienen un 3 como tercer parámetro, ya que se almacenan como valores de R, G y B.

Aquí te muestro cómo usar una capa de pooling en la red neuronal. Normalmente lo harías inmediatamente después de la capa convolucional:

```python
tf.keras.layers.MaxPooling2D(2, 2),
```

En el ejemplo de la Figura 3-4, dividimos la imagen en pools de 2 × 2 y seleccionamos el valor máximo en cada uno. Esta operación podría haber sido parametrizada para definir el tamaño del pool. Esos son los parámetros que puedes ver aquí: el (2, 2) indica que nuestros pools son de 2 × 2.

Ahora veamos el código completo para Fashion MNIST con una CNN:

```python
import tensorflow as tf  

data = tf.keras.datasets.fashion_mnist  
(training_images, training_labels), (test_images, test_labels) = data.load_data() 
 
training_images = training_images.reshape(60000, 28, 28, 1)  
training_images = training_images / 255.0  
test_images = test_images.reshape(10000, 28, 28, 1)  
test_images = test_images / 255.0  

model = tf.keras.models.Sequential([  
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
    tf.keras.layers.MaxPooling2D(2, 2),  
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  
    tf.keras.layers.MaxPooling2D(2, 2),  
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  
])  

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  
              
model.fit(training_images, training_labels, epochs=50)  
model.evaluate(test_images, test_labels)  

classifications = model.predict(test_images)  
print(classifications[0])  
print(test_labels[0])
```

Hay algunas cosas a tener en cuenta aquí. Recuerda que cuando mencioné anteriormente que la forma de entrada para las imágenes debía coincidir con lo que una capa Conv2D esperaría, y la actualizamos para que sea una imagen de 28 × 28 × 1. Los datos también tuvieron que ser reestructurados en consecuencia. 28 × 28 es el número de píxeles en la imagen, y 1 es el número de canales de color. Generalmente encontrarás que esto es 1 para una imagen en escala de grises o 3 para una imagen a color, donde hay tres canales (rojo, verde y azul), con el número indicando la intensidad de ese color. 

Entonces, antes de normalizar las imágenes, también reestructuramos cada arreglo para que tenga esa dimensión extra. El siguiente código cambia nuestro conjunto de datos de entrenamiento de 60,000 imágenes, cada una de 28 × 28 (y por lo tanto un arreglo de 60,000 × 28 × 28), a 60,000 imágenes, cada una de 28 × 28 × 1:

```python
training_images = training_images.reshape(60000, 28, 28, 1)
```

Luego hacemos lo mismo con el conjunto de datos de prueba.

También observa que en la red neuronal profunda (DNN) original pasábamos la entrada a través de una capa Flatten antes de alimentarla a la primera capa Dense. Esto lo hemos perdido en la capa de entrada aquí; en su lugar, solo especificamos la forma de entrada. Ten en cuenta que antes de la capa Dense, después de las convoluciones y el pooling, los datos se aplanarán.

Entrenando esta red con los mismos datos durante las mismas 50 épocas que la red mostrada en el Capítulo 2, podemos ver un gran aumento en la precisión. Mientras que el ejemplo anterior alcanzó un 89% de precisión en el conjunto de prueba en 50 épocas, este alcanzará un 99% en alrededor de la mitad de ese tiempo: 24 o 25 épocas. Así que podemos ver que agregar convoluciones a la red neuronal definitivamente aumenta su capacidad para clasificar imágenes. Ahora echemos un vistazo al recorrido que hace una imagen a través de la red para entender un poco más sobre por qué esto funciona.

## Explorando la red convolucional
Puedes inspeccionar tu modelo usando el comando model.summary. Cuando lo ejecutas en la red convolucional de Fashion MNIST en la que hemos estado trabajando, verás algo como esto:

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape       Param #  
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64) 640    
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64) 0     
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64) 36928   
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)   0     
_________________________________________________________________
flatten (Flatten)            (None, 1600)       0     
_________________________________________________________________
dense (Dense)                (None, 128)        204928  
_________________________________________________________________
dense_1 (Dense)              (None, 10)         1290   
=================================================================
Total params: 243,786  
Trainable params: 243,786  
Non-trainable params: 0  
```

Primero echemos un vistazo a la columna Output Shape para entender lo que sucede aquí. Nuestra primera capa manejará imágenes de 28 × 28 píxeles y aplicará 64 filtros sobre ellas. Pero como nuestro filtro es de 3 × 3, se perderá un borde de 1 píxel alrededor de la imagen, reduciendo nuestra información general a 26 × 26 píxeles. Considera la Figura 3-6. Si tomamos cada cuadro como un píxel en la imagen, el primer filtro posible comenzará en la segunda fila y la segunda columna. Lo mismo ocurre en el lado derecho y en la parte inferior del diagrama.

![Figura 3-6. Pérdida de píxeles al ejecutar un filtro](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.6.png)

Por lo tanto, una imagen de forma A × B píxeles, al pasar por un filtro de 3 × 3, se convertirá en una de (A–2) × (B–2) píxeles. De manera similar, un filtro de 5 × 5 la haría de (A–4) × (B–4), y así sucesivamente. Dado que estamos usando una imagen de 28 × 28 píxeles y un filtro de 3 × 3, nuestra salida ahora será de 26 × 26 píxeles.

Después, la capa de agrupamiento (pooling) es de 2 × 2, por lo que el tamaño de la imagen se reducirá a la mitad en cada eje, quedando en (13 × 13). La siguiente capa convolucional reducirá esto aún más a 11 × 11, y el siguiente agrupamiento, redondeando hacia abajo, dejará la imagen en 5 × 5.

De este modo, cuando la imagen haya pasado por dos capas convolucionales, el resultado serán muchas imágenes de 5 × 5 píxeles. ¿Cuántas? Lo podemos ver en la columna Param # (parámetros).

Cada convolución es un filtro de 3 × 3 más un sesgo (bias). Recuerda que, en nuestras capas densas, cada capa era Y = mX + c, donde m era nuestro parámetro (también llamado peso) y c era nuestro sesgo. Esto es muy similar, excepto que, como el filtro es de 3 × 3, hay 9 parámetros por aprender. Dado que definimos 64 convoluciones, tendremos 640 parámetros en total (cada convolución tiene 9 parámetros más un sesgo, lo que da un total de 10, y hay 64 convoluciones).

Las capas de MaxPooling no aprenden nada; solo reducen la imagen, por lo que no tienen parámetros aprendidos. Por eso, el reporte muestra 0.

La siguiente capa convolucional tiene 64 filtros, pero cada uno se multiplica por los 64 filtros previos, cada uno con 9 parámetros. También tenemos un sesgo en cada uno de los nuevos 64 filtros, por lo que el número total de parámetros será:
(64 × (64 × 9)) + 64 = 36,928

Si esto es confuso, intenta cambiar el número de convoluciones en la primera capa a, por ejemplo, 10. Verás que el número de parámetros en la segunda capa se convierte en 5,824, que es:
(64 × (10 × 9)) + 64.

Para cuando llegamos a la segunda convolución, nuestras imágenes son de 5 × 5 píxeles y tenemos 64 de ellas. Si multiplicamos esto, ahora tenemos 1,600 valores, que alimentaremos en una capa densa con 128 neuronas. Cada neurona tiene un peso y un sesgo, y tenemos 128 de ellas. El número de parámetros que la red aprenderá será:
((5 × 5 × 64) × 128) + 128 = 204,928.

Nuestra última capa densa, con 10 neuronas, toma la salida de las 128 anteriores, por lo que el número de parámetros aprendidos será:
(128 × 10) + 10 = 1,290.

El número total de parámetros será la suma de todos estos: 243,786.

Entrenar esta red requiere aprender el mejor conjunto de estos 243,786 parámetros para relacionar las imágenes de entrada con sus etiquetas. Es un proceso más lento debido a la cantidad de parámetros, pero, como podemos ver en los resultados, también construye un modelo más preciso.

Por supuesto, con este conjunto de datos todavía tenemos la limitación de que las imágenes son de 28 × 28 píxeles, monocromáticas y centradas. A continuación, exploraremos el uso de convoluciones con un conjunto de datos más complejo, compuesto por imágenes a color de caballos y humanos, e intentaremos determinar si una imagen contiene uno u otro. En este caso, el sujeto no siempre estará centrado como en Fashion MNIST, por lo que tendremos que depender de las convoluciones para detectar características distintivas.

Construcción de una CNN para Distinguir entre Caballos y Humanos

En esta sección exploraremos un escenario más complejo que el clasificador de Fashion MNIST. Ampliaremos lo aprendido sobre convoluciones y redes neuronales convolucionales para intentar clasificar el contenido de imágenes donde la ubicación de una característica no siempre está en el mismo lugar. Para ello, he creado el conjunto de datos Horses or Humans.

## Construcción de una CNN para Distinguir entre Caballos y Humanos
En esta sección exploraremos un escenario más complejo que el clasificador de Fashion MNIST. Ampliaremos lo aprendido sobre convoluciones y redes neuronales convolucionales para intentar clasificar el contenido de imágenes donde la ubicación de una característica no siempre está en el mismo lugar. Para ello, he creado el conjunto de datos Horses or Humans.

### El Conjunto de Datos Horses or Humans
Este conjunto de datos contiene más de mil imágenes de 300 × 300 píxeles, aproximadamente la mitad de caballos y la otra mitad de humanos, presentados en diferentes poses. A continuación se muestran algunos ejemplos en la Figura 3-7.

![Figura 3-7. Caballos y Humanos](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.7.png)

Como se puede observar, los sujetos tienen diferentes orientaciones y poses, y la composición de las imágenes varía. Por ejemplo, los dos caballos tienen sus cabezas orientadas de forma distinta: uno está más alejado, mostrando al animal completo, mientras que el otro está más cerca, mostrando solo la cabeza y parte del cuerpo. Similarmente, los humanos tienen diferentes tonos de piel, iluminación y poses. El hombre tiene las manos en las caderas, mientras que la mujer las tiene extendidas. Además, las imágenes incluyen fondos como árboles y playas, lo que implica que el clasificador debe determinar qué partes de la imagen son características importantes para distinguir entre un caballo y un humano, sin ser influenciado por el fondo.

Mientras que los ejemplos anteriores, como predecir Y = 2X − 1 o clasificar pequeñas imágenes monocromáticas de ropa, podrían haber sido posibles con codificación tradicional, está claro que este problema es mucho más difícil. Aquí, entramos en el territorio donde el aprendizaje automático es esencial.

Un dato interesante es que estas imágenes son generadas por computadora. La teoría es que las características detectadas en una imagen generada por computadora de un caballo deberían aplicarse a una imagen real. Más adelante en este capítulo veremos qué tan bien funciona esto.

## Keras ImageDataGenerator
El conjunto de datos Fashion MNIST que hemos utilizado hasta ahora incluye etiquetas asociadas a cada imagen. Sin embargo, muchos conjuntos de datos basados en imágenes no tienen etiquetas, y Horses or Humans no es una excepción. En lugar de etiquetas, las imágenes están organizadas en subdirectorios según su tipo. Con Keras en TensorFlow, una herramienta llamada ImageDataGenerator puede usar esta estructura para asignar automáticamente etiquetas a las imágenes.

Para utilizar ImageDataGenerator, simplemente necesitas asegurarte de que la estructura del directorio tenga un conjunto de subdirectorios con nombres, donde cada subdirectorio sea una etiqueta. Por ejemplo, el conjunto de datos Horses or Humans está disponible como un conjunto de archivos ZIP: uno con los datos de entrenamiento (más de 1,000 imágenes) y otro con los datos de validación (256 imágenes). Al descargarlos y descomprimirlos en un directorio local para entrenamiento y validación, la estructura del archivo debe verse como la Figura 3-8.

Aquí está el código para descargar los datos de entrenamiento y extraerlos en los subdirectorios apropiados, como se muestra en la figura:

```python
import urllib.request
import zipfile

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'

urllib.request.urlretrieve(url, file_name)
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()
```

![Figura 3-8. Asegurando que las imágenes estén en subdirectorios con nombres](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.8.png)

Este código descarga el archivo ZIP de los datos de entrenamiento y lo descomprime en un directorio llamado horse-or-human/training. Este será el directorio principal que contendrá subdirectorios para cada tipo de imagen.

> En su defecto en la siguiente dirección están los datos para descargar manualmente https://laurencemoroney.com/datasets.html

Para utilizar ImageDataGenerator, ahora simplemente usamos el siguiente código:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# Todas las imágenes serán reescaladas por 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'horse-or-human/training/',
    target_size=(300, 300),
    class_mode='binary'
)
```

Primero, creamos una instancia de un ImageDataGenerator llamado train_datagen. Luego, especificamos que este generará imágenes para el proceso de entrenamiento fluyendo desde un directorio. El directorio es training_dir, como se especificó anteriormente. También indicamos algunos hiperparámetros sobre los datos, como el tamaño objetivo (en este caso, las imágenes son de 300 × 300) y el modo de clase. Este último suele ser binary si solo hay dos tipos de imágenes (como en este caso) o categorical si hay más de dos.

## Arquitectura CNN para Caballos o Humanos
Existen varias diferencias importantes entre este conjunto de datos y el de Fashion MNIST que debes considerar al diseñar una arquitectura para clasificar las imágenes. Primero, las imágenes son mucho más grandes (300 × 300 píxeles), por lo que se pueden necesitar más capas. Segundo, las imágenes están en color completo, no en escala de grises, por lo que cada imagen tendrá tres canales en lugar de uno. Tercero, solo hay dos tipos de imágenes, por lo que tenemos un clasificador binario que se puede implementar utilizando solo una neurona de salida, donde los valores se aproximan a 0 para una clase y 1 para la otra. Ten en cuenta estas consideraciones al explorar esta arquitectura:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Hay varios aspectos a notar aquí. Primero, observa la capa inicial. Estamos definiendo 16 filtros, cada uno de 3 × 3, pero la forma de entrada de la imagen es (300, 300, 3). Recuerda que esto se debe a que nuestra imagen de entrada es de 300 × 300 y está en color, por lo que tiene tres canales en lugar de uno, como el conjunto de datos en escala de grises Fashion MNIST que usamos antes.

En el extremo opuesto, nota que solo hay una neurona en la capa de salida. Esto se debe a que estamos usando un clasificador binario, y podemos obtener una clasificación binaria con solo una neurona si la activamos con una función sigmoid. La función sigmoid tiene como objetivo dirigir un conjunto de valores hacia 0 y el otro hacia 1, lo cual es perfecto para una clasificación binaria.

Luego, observa cómo apilamos varias capas de convolución adicionales. Hacemos esto porque nuestra fuente de imágenes es bastante grande, y queremos, con el tiempo, obtener imágenes más pequeñas, cada una con características resaltadas. Si observamos los resultados de model.summary, veremos esto en acción:

```python
=================================================================
Layer (type)                Output Shape              Param #
=================================================================
conv2d (Conv2D)             (None, 298, 298, 16)     448
_________________________________________________________________
max_pooling2d (MaxPooling2D)(None, 149, 149, 16)     0
_________________________________________________________________
conv2d_1 (Conv2D)           (None, 147, 147, 32)     4640
_________________________________________________________________
max_pooling2d_1 (MaxPooling2(None, 73, 73, 32)       0
_________________________________________________________________
conv2d_2 (Conv2D)           (None, 71, 71, 64)       18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2(None, 35, 35, 64)       0
_________________________________________________________________
conv2d_3 (Conv2D)           (None, 33, 33, 64)       36928
_________________________________________________________________
max_pooling2d_3 (MaxPooling2(None, 16, 16, 64)       0
_________________________________________________________________
conv2d_4 (Conv2D)           (None, 14, 14, 64)       36928
_________________________________________________________________
max_pooling2d_4 (MaxPooling2(None, 7, 7, 64)         0
_________________________________________________________________
flatten (Flatten)           (None, 3136)             0
_________________________________________________________________
dense (Dense)               (None, 512)              1606144
_________________________________________________________________
dense_1 (Dense)             (None, 1)                513
=================================================================
Total params: 1,704,097
Trainable params: 1,704,097
Non-trainable params: 0
=================================================================
```

Nota cómo, al pasar los datos por todas las capas de convolución y pooling, terminan siendo elementos de 7 × 7. La teoría es que estos serán mapas de características activadas que son relativamente simples y contienen solo 49 píxeles. Estos mapas de características pueden luego pasarse a la red neuronal densa para asociarlos con las etiquetas apropiadas.

Esto, por supuesto, nos lleva a tener muchos más parámetros que la red anterior, por lo que será más lenta para entrenar. Con esta arquitectura, vamos a aprender 1.7 millones de parámetros.

Para entrenar la red, debemos compilarla con una función de pérdida y un optimizador. En este caso, la función de pérdida puede ser binary crossentropy, porque solo hay dos clases, y como sugiere el nombre, esta es una función de pérdida diseñada para ese escenario. Podemos intentar un nuevo optimizador, Root Mean Square Propagation (RMSprop), que toma un parámetro de tasa de aprendizaje (learning rate, lr) que nos permite ajustarla. Aquí está el código:

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)
```

Entrenamos utilizando fit_generator y pasándole el training_generator que creamos anteriormente:

```python
history = model.fit(
    train_generator,
    epochs=15
)
```

Este ejemplo funcionará en Colab, pero si deseas ejecutarlo en tu propia máquina, asegúrate de que las bibliotecas Pillow estén instaladas usando pip install pillow.

Nota que con TensorFlow Keras puedes usar model.fit para ajustar tus datos de entrenamiento a tus etiquetas. Al usar un generador, las versiones anteriores requerían que usaras model.fit_generator en su lugar. Las versiones más recientes de TensorFlow te permitirán usar cualquiera de las dos.

En solo 15 épocas, esta arquitectura nos da una precisión muy impresionante de más del 95 % en el conjunto de entrenamiento. Por supuesto, esto es solo con los datos de entrenamiento y no es un indicativo del rendimiento en datos que la red no ha visto previamente.

A continuación, veremos cómo agregar el conjunto de validación usando un generador y medir su rendimiento para darnos una buena indicación de cómo podría funcionar este modelo en la vida real.

## Añadiendo Validación al Dataset Horses or Humans
Para agregar validación, necesitarás un conjunto de datos de validación que sea independiente del de entrenamiento. En algunos casos recibirás un conjunto de datos maestro que tendrás que dividir tú mismo, pero en el caso de Horses or Humans, hay un conjunto de validación separado que puedes descargar.

Puede que te preguntes por qué estamos hablando de un conjunto de datos de validación aquí, en lugar de un conjunto de prueba, y si son lo mismo.

> Para modelos simples como los desarrollados en los capítulos anteriores, a menudo es suficiente dividir el conjunto de datos en dos partes, una para entrenamiento y otra para prueba. Pero para modelos más complejos como el que estamos construyendo aquí, querrás crear conjuntos de validación y prueba separados. 
¿Cuál es la diferencia? 
Datos de entrenamiento: son los datos que se usan para enseñar a la red cómo los datos y las etiquetas encajan entre sí. 
Datos de validación: se usan para ver cómo le va a la red con datos no vistos previamente mientras la entrenas—es decir, no se usan para ajustar datos a etiquetas, sino para inspeccionar qué tan bien está yendo el ajuste. 
Datos de prueba: se usan después del entrenamiento para ver cómo le va a la red con datos que nunca antes ha visto.Algunos conjuntos de datos vienen con una división en tres partes, y en otros casos querrás separar el conjunto de prueba en dos partes para validación y prueba. Aquí, descargarás algunas imágenes adicionales para probar el modelo.

Puedes usar un código muy similar al usado para las imágenes de entrenamiento para descargar el conjunto de validación y descomprimirlo en un directorio diferente:

```python
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"  

validation_file_name = "validation-horse-or-human.zip"  
validation_dir = 'horse-or-human/validation/'  
urllib.request.urlretrieve(validation_url, validation_file_name)  

zip_ref = zipfile.ZipFile(validation_file_name, 'r')  
zip_ref.extractall(validation_dir)  
zip_ref.close()  
```

> En su defecto en la siguiente dirección están los datos para descargar manualmente https://laurencemoroney.com/datasets.html

Una vez que tengas los datos de validación, puedes configurar otro ImageDataGenerator para manejar estas imágenes:

```python
validation_datagen = ImageDataGenerator(rescale=1/255)  

validation_generator = validation_datagen.flow_from_directory(  
    validation_dir,  
    target_size=(300, 300),  
    class_mode='binary'  
)  
```

Para que TensorFlow realice la validación por ti, simplemente actualiza tu método model.fit_generator para indicar que quieres usar los datos de validación para probar el modelo época por época. Lo haces usando el parámetro validation_data y pasándole el generador de validación que acabas de construir:

```python
history = model.fit_generator(  
    train_generator,  
    epochs=15,  
    validation_data=validation_generator  
)  
```

Después de entrenar durante 15 épocas, deberías ver que tu modelo tiene una precisión de más del 99 % en el conjunto de entrenamiento, pero solo alrededor del 88 % en el conjunto de validación. Esto indica que el modelo está sobreajustando, como vimos en el capítulo anterior.

Aun así, el rendimiento no es malo considerando lo pocas imágenes con las que fue entrenado y qué tan diversas eran esas imágenes. Estás comenzando a toparte con un límite causado por la falta de datos, pero hay algunas técnicas que puedes usar para mejorar el rendimiento de tu modelo. Las exploraremos más adelante en este capítulo, pero antes de eso veamos cómo usar este modelo.

## Probando Imágenes de Horses or Humans
Construir un modelo está muy bien, pero por supuesto, querrás probarlo. Una de las principales frustraciones que tuve cuando comencé mi viaje en IA fue que podía encontrar mucho código que mostraba cómo construir modelos y gráficos de cómo estaban funcionando esos modelos, pero muy pocas veces había código para probar el modelo por mi cuenta. Intentaré evitar eso en este libro.

Probar el modelo es quizás más fácil usando Colab. He proporcionado un notebook de Horses or Humans en GitHub que puedes abrir directamente en Colab.

Una vez que hayas entrenado el modelo, verás una sección llamada “Ejecutando el Modelo.” Antes de ejecutarlo, encuentra algunas imágenes de caballos o humanos en línea y descárgalas a tu computadora. Pixabay.com es un buen sitio para buscar imágenes libres de derechos. Es una buena idea preparar tus imágenes de prueba primero, porque el nodo puede agotarse mientras buscas.

La Figura 3-9 muestra algunas imágenes de caballos y humanos que descargué de Pixabay para probar el modelo.

![Figura 3-9. Imágenes de prueba](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.9.png)

Cuando las subí, como puedes ver en la Figura 3-10, el modelo clasificó correctamente la primera imagen como un humano y la tercera imagen como un caballo, pero la imagen del medio, a pesar de ser obviamente un humano, fue clasificada incorrectamente como un caballo.

![Figura 3-10. Ejecutando el modelo](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.10.png)

También puedes cargar múltiples imágenes simultáneamente y hacer que el modelo haga predicciones para todas ellas. Puede que notes que tiende a sobreajustarse hacia los caballos. Si el humano no está completamente posado—es decir, si no puedes ver su cuerpo completo—puede inclinarse hacia los caballos. Eso fue lo que ocurrió en este caso. El primer humano está completamente posado y la imagen se parece a muchas de las poses del conjunto de datos, por lo que pudo clasificarla correctamente. El segundo humano estaba frente a la cámara, pero solo aparece la mitad superior de su cuerpo en la imagen. No había datos de entrenamiento que se vieran así, por lo que el modelo no pudo identificarla correctamente.

Exploremos ahora el código para ver qué está haciendo. Quizás la parte más importante es este fragmento:

```python
img = image.load_img(path, target_size=(300, 300))  
x = image.img_to_array(img)  
x = np.expand_dims(x, axis=0)  
```

Aquí, estamos cargando la imagen desde la ruta que Colab escribió. Ten en cuenta que especificamos que el tamaño objetivo sea 300 × 300. Las imágenes cargadas pueden tener cualquier forma, pero si las vamos a alimentar al modelo, deben ser de 300 × 300, porque ese es el tamaño que el modelo fue entrenado para reconocer. Entonces, la primera línea de código carga la imagen y la redimensiona a 300 × 300.

La siguiente línea de código convierte la imagen en un array 2D. Sin embargo, el modelo espera un array 3D, como se indica en input_shape en la arquitectura del modelo. Afortunadamente, Numpy proporciona un método expand_dims que maneja esto y nos permite agregar fácilmente una nueva dimensión al array.

Ahora que tenemos nuestra imagen en un array 3D, solo queremos asegurarnos de que esté apilada verticalmente para que tenga la misma forma que los datos de entrenamiento:

```python
image_tensor = np.vstack([x])  
```

Con nuestra imagen en el formato correcto, es fácil hacer la clasificación:

```python
classes = model.predict(image_tensor)  
```

El modelo devuelve un array que contiene las clasificaciones. Como en este caso solo hay una clasificación, es efectivamente un array que contiene otro array. Puedes ver esto en la Figura 3-10, donde para el primer modelo (humano) se ve algo como [[1.]].

Así que ahora es simplemente cuestión de inspeccionar el valor del primer elemento en ese array. Si es mayor que 0.5, estamos viendo un humano:

```python
if classes[0] > 0.5:  
    print(fn + " es un humano")  
else:  
    print(fn + " es un caballo")  
```

Hay algunos puntos importantes a considerar aquí. Primero, aunque la red fue entrenada con imágenes sintéticas generadas por computadora, funciona bastante bien al identificar caballos o humanos en fotografías reales. Esto es una ventaja potencial, ya que puede que no necesites miles de fotografías para entrenar un modelo, y puedes hacerlo de manera relativamente económica con imágenes generadas por computadora.

Pero este conjunto de datos también demuestra un problema fundamental que enfrentarás. Tu conjunto de entrenamiento no puede esperar representar cada posible escenario que tu modelo podría enfrentar en el mundo real, y por lo tanto, el modelo siempre tendrá algún nivel de sobreespecialización hacia el conjunto de entrenamiento. Un ejemplo claro y simple de esto se mostró aquí, donde el humano en el centro de la Figura 3-9 fue mal categorizado. El conjunto de entrenamiento no incluía un humano en esa pose, y por lo tanto, el modelo no "aprendió" que un humano podría lucir así. Como resultado, había una gran probabilidad de que viera la figura como un caballo, y en este caso, así fue.

¿Cuál es la solución? La solución obvia es agregar más datos de entrenamiento, con humanos en esa pose particular y otras que no estaban representadas inicialmente. Sin embargo, eso no siempre es posible. Por suerte, hay un truco interesante en TensorFlow que puedes usar para extender virtualmente tu conjunto de datos: se llama aumento de imágenes (image augmentation), y lo exploraremos a continuación.

## Aumento de Imágenes

En la sección anterior, construiste un modelo clasificador de caballos o humanos que fue entrenado con un conjunto de datos relativamente pequeño. Como resultado, pronto comenzaste a enfrentar problemas al clasificar algunas imágenes no vistas previamente, como la mala clasificación de una mujer con un caballo, ya que el conjunto de entrenamiento no incluía imágenes de personas en esa pose.

Una forma de lidiar con tales problemas es mediante el aumento de imágenes. La idea detrás de esta técnica es que, mientras TensorFlow carga tus datos, puede crear datos adicionales nuevos modificando lo que tiene usando una serie de transformaciones. Por ejemplo, mira la Figura 3-11. Aunque no hay nada en el conjunto de datos que se parezca a la mujer de la derecha, la imagen de la izquierda es algo similar.

![Figura 3-11. Similitudes en el conjunto de datos](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.11.png)

Entonces, si pudieras, por ejemplo, hacer un zoom en la imagen de la izquierda mientras entrenas, como se muestra en la Figura 3-12, aumentarías las posibilidades de que el modelo pueda clasificar correctamente la imagen de la derecha como una persona.

![Figura 3-12. Haciendo zoom en los datos del conjunto de entrenamiento](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.12.png)

De manera similar, puedes ampliar el conjunto de entrenamiento con una variedad de otras transformaciones, incluyendo:

- Rotación
- Desplazamiento horizontal
- Desplazamiento vertical
- Cizalladura
- Zoom
- Volteo

Dado que has estado utilizando el ImageDataGenerator para cargar las imágenes, ya has visto que realiza una transformación, como cuando normalizó las imágenes así:

```python
train_datagen = ImageDataGenerator(rescale=1/255)
```

Las otras transformaciones también están fácilmente disponibles dentro del ImageDataGenerator, así que, por ejemplo, podrías hacer algo como esto:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

Aquí, además de reescalar la imagen para normalizarla, también estás haciendo lo siguiente:

- Rotando cada imagen aleatoriamente hasta 40 grados hacia la izquierda o hacia la derecha
- Traduciendo la imagen hasta un 20% vertical o horizontalmente
- Cizallando la imagen hasta un 20%
- Haciendo zoom en la imagen hasta un 20%
- Volteando aleatoriamente la imagen horizontal o verticalmente
- Rellenando cualquier píxel faltante después de un movimiento o cizalladura con los vecinos más cercanos

Cuando vuelvas a entrenar con estos parámetros, una de las primeras cosas que notarás es que el entrenamiento toma más tiempo debido a todo el procesamiento de imágenes. Además, la precisión de tu modelo puede no ser tan alta como antes, porque anteriormente estaba sobreajustado a un conjunto de datos mayormente uniforme.

En mi caso, cuando entrené con estos aumentos, mi precisión bajó del 99% al 85% después de 15 épocas, con una validación ligeramente superior al 89%. (Esto indica que el modelo está un poco subajustado, por lo que los parámetros podrían ajustarse un poco).

¿Qué pasa con la imagen de la Figura 3-9 que clasificó mal antes? Esta vez, la clasifica correctamente. Gracias a los aumentos de imágenes, ahora el conjunto de entrenamiento tiene suficiente cobertura para que el modelo entienda que esta imagen en particular también es un ser humano (ver Figura 3-13). Este es solo un punto de datos, y puede no ser representativo de los resultados para datos reales, pero es un pequeño paso en la dirección correcta.

![Figura 3-13. La mujer con zoom ahora se clasifica correctamente](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.13.png)

Como puedes ver, incluso con un conjunto de datos relativamente pequeño como Caballos o Humanos, puedes comenzar a construir un clasificador bastante decente. Con conjuntos de datos más grandes podrías llevar esto más lejos. Otra técnica para mejorar el modelo es usar características que ya se han aprendido en otro lugar. Muchos investigadores con enormes recursos (millones de imágenes) y grandes modelos entrenados en miles de clases han compartido sus modelos, y utilizando un concepto llamado aprendizaje por transferencia, puedes usar las características que esos modelos aprendieron y aplicarlas a tus datos. ¡Exploraremos eso a continuación!

## Aprendizaje por Transferencia
Como ya hemos visto en este capítulo, el uso de convoluciones para extraer características puede ser una herramienta poderosa para identificar el contenido de una imagen. Los mapas de características resultantes pueden luego ser alimentados a las capas densas de una red neuronal para asignarlos a las etiquetas y darnos una forma más precisa de determinar el contenido de una imagen.

Usando este enfoque, con una red neuronal simple y rápida de entrenar y algunas técnicas de aumento de imágenes, construimos un modelo que tenía un 80-90% de precisión al distinguir entre un caballo y un humano cuando fue entrenado con un conjunto de datos muy pequeño.

Pero podemos mejorar aún más nuestro modelo utilizando un método llamado aprendizaje por transferencia. La idea detrás del aprendizaje por transferencia es simple: en lugar de aprender un conjunto de filtros desde cero para nuestro conjunto de datos, ¿por qué no usar un conjunto de filtros que fueron aprendidos en un conjunto de datos mucho más grande, con muchas más características de las que podemos "permitirnos" construir desde cero? Podemos colocarlos en nuestra red y luego entrenar un modelo con nuestros datos utilizando los filtros preaprendidos.

Por ejemplo, nuestro conjunto de datos de Caballos o Humanos tiene solo dos clases. Podemos usar un modelo existente que fue preentrenado para mil clases, pero en algún momento tendremos que descartar parte de la red preexistente y agregar las capas que nos permitirán tener un clasificador para dos clases.

La Figura 3-14 muestra cómo podría ser una arquitectura de CNN para una tarea de clasificación como la nuestra. Tenemos una serie de capas convolucionales que conducen a una capa densa, que a su vez conduce a una capa de salida.

![Figura 3-14. Una arquitectura de red neuronal convolucional](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.14.png)

Hemos visto que podemos construir un clasificador bastante bueno usando esta arquitectura. Pero con el aprendizaje por transferencia, ¿qué pasa si podemos tomar las capas preaprendidas de otro modelo, congelarlas o bloquearlas para que no sean entrenables, y luego ponerlas encima de nuestro modelo, como en la Figura 3-15?

![Figura 3-15. Tomando capas de otra arquitectura mediante aprendizaje por transferencia](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.15.png)

Cuando consideramos que, una vez que han sido entrenadas, todas estas capas son solo un conjunto de números que indican los valores de los filtros, los pesos y los sesgos, junto con una arquitectura conocida (número de filtros por capa, tamaño del filtro, etc.), la idea de reutilizarlas es bastante directa.

Veamos cómo se vería esto en el código. Ya hay varios modelos preentrenados disponibles de diversas fuentes. Usaremos la versión 3 del popular modelo Inception de Google, que está entrenado con más de un millón de imágenes de una base de datos llamada ImageNet. Tiene docenas de capas y puede clasificar imágenes en mil categorías. Un modelo guardado está disponible con los pesos preentrenados. Para usar esto, simplemente descargamos los pesos, creamos una instancia de la arquitectura Inception V3 y luego cargamos los pesos en esta arquitectura de esta forma:

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights="imagenet")
```

Ahora tenemos un modelo Inception completo que está preentrenado. Si deseas inspeccionar su arquitectura, puedes hacerlo con:

```python
pre_trained_model.summary()
```

¡Advertencia! Es enorme. Aún así, échale un vistazo para ver las capas y sus nombres. Me gusta usar la llamada "mixed7" porque su salida es pequeña y agradable: imágenes de 7 × 7, pero siéntete libre de experimentar con otras.

A continuación, congelaremos toda la red (excepto las ultimas capas) para evitar que se entrene nuevamente y luego asignaremos una variable para señalar la salida de "mixed7" como el lugar donde queremos recortar la red. Podemos hacer eso con este código:

```python
for layer in pre_trained_model.layers:
    layer.trainable = False

for layer in pre_trained_model.layers[-10:]:
    layer.trainable = True

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
```

Nota que imprimimos la forma de la salida de la última capa, y verás que obtenemos imágenes de 7 × 7 en este punto. Esto indica que, después de que las imágenes hayan pasado por "mixed7", las imágenes de salida de los filtros tienen un tamaño de 7 × 7, por lo que son bastante fáciles de manejar. Nuevamente, no tienes que elegir esa capa específica; puedes experimentar con otras.

Ahora veamos cómo agregar nuestras capas densas debajo de esto:

```python
# Aplanar la capa de salida a 1 dimensión
x = Flatten()(last_output)
# Agregar una capa completamente conectada con 1,024 unidades ocultas y activación relu
x = Dense(1024, activation='relu')(x)
# Agregar una capa de sigmoid para clasificación
x = Dense(1, activation='sigmoid')(x)
```

Es tan simple como crear un conjunto de capas aplanadas a partir de la última salida, porque vamos a alimentar los resultados a una capa densa. Luego, agregamos una capa densa de 1,024 neuronas y una capa densa con 1 neurona para nuestra salida.

Ahora podemos definir nuestro modelo simplemente diciendo que es la entrada de nuestro modelo preentrenado seguida de la "x" que acabamos de definir. Luego lo compilamos de la manera usual:

```python
model = Model(inputs=pre_trained_model.input, outputs=x)
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

Entrenar el modelo con esta arquitectura durante 40 épocas dio una precisión del 99%+, con una precisión de validación del 96%+ (ver Figura 3-16).

![Figura 3-16. Entrenando el clasificador de caballos o humanos con aprendizaje por transferencia](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.16.png)

Los resultados aquí son mucho mejores que con nuestro modelo anterior, pero puedes seguir ajustando y mejorándolo. También puedes explorar cómo funcionará el modelo con un conjunto de datos mucho más grande, como el famoso "Dogs vs. Cats" de Kaggle. Este es un conjunto de datos extremadamente variado que consiste en 25,000 imágenes de gatos y perros, a menudo con los sujetos algo ocultos; por ejemplo, si son sostenidos por un humano.

Usando el mismo algoritmo y diseño de modelo que antes, puedes entrenar un clasificador de "Dogs vs. Cats" en Colab, utilizando una GPU, a aproximadamente 3 minutos por época. Para 20 épocas, esto equivale a aproximadamente 1 hora de entrenamiento.

Cuando se probó con imágenes muy complejas como las de la Figura 3-17, este clasificador las clasificó todas correctamente. Elegí una imagen de un perro con orejas parecidas a las de un gato, y una con su espalda vuelta. Ambas imágenes de gatos no eran típicas.

![Figura 3-17. Perros y gatos inusuales que fueron clasificados correctamente](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.17.png)

El gato en la esquina inferior derecha, con los ojos cerrados, las orejas hacia abajo y la lengua afuera mientras se lavaba la pata, dio los resultados de la Figura 3-18 cuando se cargó en el modelo. Puedes ver que dio un valor muy bajo (4.98 × 10–24), lo que muestra que la red estaba casi segura de que era un gato.


![Figura 3-18. Clasificando al gato que se lava la pata](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.18.png)

Puedes encontrar el código completo para los clasificadores de Caballos o Humanos y Dogs vs. Cats en el repositorio de GitHub de este libro.

## Clasificación Multiclase
En todos los ejemplos anteriores, has estado construyendo clasificadores binarios, aquellos que eligen entre dos opciones (caballos o humanos, gatos o perros). Cuando construimos clasificadores multiclase, los modelos son casi los mismos, pero existen algunas diferencias importantes.

En lugar de una sola neurona activada por sigmoide, o dos neuronas activadas de forma binaria, ahora tu capa de salida necesitará n neuronas, donde n es el número de clases que deseas clasificar. También tendrás que cambiar tu función de pérdida a una apropiada para múltiples categorías. Por ejemplo, mientras que para los clasificadores binarios que has construido hasta ahora en este capítulo, la función de pérdida era la entropía cruzada binaria, si deseas extender el modelo para múltiples clases, deberías usar en su lugar la entropía cruzada categórica. Si estás utilizando el ImageDataGenerator para proporcionar tus imágenes, el etiquetado se realiza automáticamente, por lo que las múltiples categorías funcionarán igual que las binarias; el ImageDataGenerator simplemente etiquetará según la cantidad de subdirectorios.

Considera, por ejemplo, el juego Piedra, Papel o Tijeras. Si quisieras entrenar un conjunto de datos para reconocer los diferentes gestos de las manos, necesitarías manejar tres categorías. Afortunadamente, hay un conjunto de datos simple que puedes usar para esto.

Hay dos descargas: un conjunto de entrenamiento de muchas manos diversas, con diferentes tamaños, formas, colores y detalles como esmalte de uñas; y un conjunto de prueba de manos igualmente diversas, ninguna de las cuales está en el conjunto de entrenamiento.

Puedes ver algunos ejemplos en la Figura 3-19.

![Figura 3-19. Ejemplos de gestos de Piedra/Papel/Tijeras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.19.png)

Usar el conjunto de datos es simple. Descárgalo y descomprímelo; los subdirectorios ya están organizados en el archivo ZIP; luego utilízalo para inicializar un ImageDataGenerator:

```python
!wget --no-check-certificate \
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /tmp/rps.zip
local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()
TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
  rescale = 1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)
```

Sin embargo, ten en cuenta que cuando configures el generador de datos a partir de esto, debes especificar que el modo de clase es categórico para que el ImageDataGenerator use más de dos subdirectorios:

```python
train_generator = training_datagen.flow_from_directory(
  TRAINING_DIR,
  target_size=(150, 150),
  class_mode='categorical'
)
```

Cuando definas tu modelo, mientras te aseguras de que las capas de entrada y salida coincidan, debes asegurarte de que la entrada coincida con la forma de los datos (en este caso 150 × 150) y que la salida coincida con el número de clases (ahora tres):

```python
model = tf.keras.models.Sequential([
  # Nota: la forma de entrada es el tamaño deseado de la imagen: 
  # 150x150 con 3 bytes de color
  # Esta es la primera convolución
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  # La segunda convolución
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # La tercera convolución
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # La cuarta convolución
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # Aplanar los resultados para alimentar a una DNN
  tf.keras.layers.Flatten(),
  # Capa oculta con 512 neuronas
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])
```

Finalmente, al compilar tu modelo, asegúrate de que utilice una función de pérdida categórica, como la entropía cruzada categórica. La entropía cruzada binaria no funcionará con más de dos clases:

```python
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

El entrenamiento es luego el mismo que antes:

```python
history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)
```

Tu código para probar las predicciones también necesitará cambiar un poco. Ahora hay tres neuronas de salida, y estas saldrán con un valor cercano a 1 para la clase predicha y cercano a 0 para las otras clases. Ten en cuenta que la función de activación utilizada es softmax, lo que garantizará que las tres predicciones sumen 1. Por ejemplo, si el modelo ve algo sobre lo que no está seguro, puede que dé una salida de .4, .4, .2, pero si ve algo sobre lo que está bastante seguro, podría dar .98, .01, .01.
También ten en cuenta que cuando uses el ImageDataGenerator, las clases se cargan en orden alfabético; por lo tanto, aunque podrías esperar que las neuronas de salida estén en el orden del nombre del juego, en realidad el orden será Papel, Piedra, Tijeras.

El código para probar predicciones en un cuaderno Colab será algo así. Es muy similar a lo que viste antes:

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

  # prediciendo imágenes
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)
```

Ten en cuenta que no analiza la salida, solo imprime las clases. La Figura 3-20 muestra cómo se ve al usarlo.

![Figura 3-20. Probando el clasificador de Piedra/Papel/Tijeras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.20.png)

Puedes ver desde los nombres de archivo qué imágenes eran. Paper1.png terminó siendo [1, 0, 0], lo que significa que la primera neurona se activó y las demás no. De manera similar, Rock1.png terminó siendo [0, 1, 0], activando la segunda neurona, y Scissors2.png fue [0, 0, 1]. ¡Recuerda que las neuronas están en orden alfabético por etiqueta!

Algunas imágenes que puedes usar para probar el conjunto de datos están disponibles para su descarga. Alternativamente, por supuesto, puedes probar con las tuyas propias. Ten en cuenta que las imágenes de entrenamiento están hechas contra un fondo blanco, por lo que puede haber algo de confusión si hay muchos detalles en el fondo de las fotos que tomes.

## Regularización por Dropout
Anteriormente en este capítulo discutimos el sobreajuste, donde una red neuronal puede volverse demasiado especializada en un tipo particular de datos de entrada y tener un desempeño deficiente con otros tipos. Una técnica para ayudar a superar esto es el uso de la regularización por dropout.

Cuando se entrena una red neuronal, cada neurona individual tiene un efecto sobre las neuronas en las capas siguientes. Con el tiempo, especialmente en redes más grandes, algunas neuronas pueden volverse demasiado especializadas, lo que afecta hacia abajo, potencialmente causando que toda la red se vuelva demasiado especializada y conduzca al sobreajuste.

Además, las neuronas vecinas pueden terminar con pesos y sesgos similares, lo que, si no se monitorea, puede hacer que el modelo en general se vuelva demasiado especializado en las características activadas por esas neuronas.

Por ejemplo, considera la red neuronal en la Figura 3-21, donde hay capas de 2, 6, 6 y 2 neuronas. Las neuronas en las capas intermedias podrían terminar con pesos y sesgos muy similares.

![Figura 3-21. Una red neuronal simple](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.21.png)

Mientras entrenas, si eliminas un número aleatorio de neuronas y las ignoras, su contribución a las neuronas de la siguiente capa se bloquea temporalmente (Figura 3-22).

![Figura 3-22. Una red neuronal con dropout](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure3.22.png)

Esto reduce las probabilidades de que las neuronas se vuelvan demasiado especializadas. La red seguirá aprendiendo el mismo número de parámetros, pero debería ser mejor para generalizar, es decir, debería ser más resistente a diferentes entradas.

> El concepto de dropouts fue propuesto por Nitish Srivastava et al. en su artículo de 2014 “Dropout: A Simple Way to Prevent Neural Networks from Overfitting”.

Para implementar dropouts en TensorFlow, puedes usar una capa simple de Keras como esta:

```python
tf.keras.layers.Dropout(0.2)
```

Esto eliminará, aleatoriamente, el porcentaje especificado de neuronas (en este caso, 20%) en la capa especificada. Ten en cuenta que puede ser necesario experimentar para encontrar el porcentaje adecuado para tu red.

Para un ejemplo simple que demuestra esto, considera el clasificador de Fashion MNIST del Capítulo 2. Cambiaré la definición de la red para tener muchas más capas, como esta:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

Entrenar esta red durante 20 épocas dio alrededor de un 94% de precisión en el conjunto de entrenamiento y aproximadamente 88.5% en el conjunto de validación. Esto es una señal de posible sobreajuste.

Introducir dropouts después de cada capa densa se ve así:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

Cuando se entrenó esta red durante el mismo período con los mismos datos, la precisión en el conjunto de entrenamiento bajó a aproximadamente 89.5%. La precisión en el conjunto de validación permaneció casi igual, en 88.3%. Estos valores están mucho más cerca entre sí; por lo tanto, la introducción de dropouts no solo demostró que el sobreajuste estaba ocurriendo, sino que también mostró que el uso de dropouts puede ayudar a eliminar esa ambigüedad asegurando que la red no se especialice demasiado en los datos de entrenamiento.

Recuerda que, al diseñar tus redes neuronales, obtener excelentes resultados en tu conjunto de entrenamiento no siempre es algo bueno. Esto podría ser una señal de sobreajuste. Introducir dropouts puede ayudarte a eliminar este problema, para que puedas optimizar tu red en otras áreas sin esa falsa sensación de seguridad.

## Resumen
Este capítulo te introdujo a una forma más avanzada de lograr visión por computadora utilizando redes neuronales convolucionales. Viste cómo usar convoluciones para aplicar filtros que pueden extraer características de las imágenes y diseñaste tus primeras redes neuronales para tratar con escenarios de visión más complejos que los que encontraste con los conjuntos de datos MNIST y Fashion MNIST. También exploraste técnicas para mejorar la precisión de tu red y evitar el sobreajuste, como el uso de aumento de imágenes y dropouts.

Antes de explorar más escenarios, en el Capítulo 4 recibirás una introducción a TensorFlow Datasets, una tecnología que facilita el acceso a datos para entrenar y probar tus redes. En este capítulo, descargabas archivos ZIP y extraías imágenes, pero eso no siempre será posible. Con TensorFlow Datasets podrás acceder a muchos conjuntos de datos con una API estándar.
