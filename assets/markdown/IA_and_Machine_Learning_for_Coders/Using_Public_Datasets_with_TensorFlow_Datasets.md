# Usar conjuntos de datos públicos con TensorFlow Datasets
En los primeros capítulos de este libro, entrenaste modelos utilizando una variedad de datos, desde el conjunto de datos Fashion MNIST que está convenientemente incluido con Keras hasta los conjuntos de datos basados en imágenes Horses or Humans y Dogs vs. Cats, que estaban disponibles como archivos ZIP que tuviste que descargar y preprocesar. Probablemente ya te hayas dado cuenta de que hay muchas formas diferentes de obtener los datos con los que entrenar un modelo.

Sin embargo, muchos conjuntos de datos públicos requieren que aprendas muchas habilidades específicas de dominio antes de empezar a considerar la arquitectura de tu modelo. El objetivo detrás de TensorFlow Datasets (TFDS) es exponer conjuntos de datos de una manera fácil de consumir, donde todos los pasos de preprocesamiento para adquirir los datos y hacerlos compatibles con las API de TensorFlow ya estén hechos para ti.

Ya has visto un poco de esta idea con cómo Keras manejó Fashion MNIST en los Capítulos 1 y 2. Como recordatorio, todo lo que tenías que hacer para obtener los datos era esto:

```python
data = tf.keras.datasets.fashion_mnist  
(training_images, training_labels), (test_images, test_labels) = data.load_data()  
```

TFDS amplía esta idea, pero expande enormemente no solo el número de conjuntos de datos disponibles, sino también la diversidad de tipos de conjuntos de datos. La lista de conjuntos de datos disponibles está creciendo constantemente, en categorías como:

- Audio: Datos de habla y música
- Imagen: Desde conjuntos de datos simples para aprendizaje como Horses or Humans hasta conjuntos de datos avanzados para investigación, como la detección de retinopatía diabética
- Detección de objetos: COCO, Open Images, y más
- Datos estructurados: Sobrevivientes del Titanic, reseñas de Amazon, y más
- Resumen: Noticias de CNN y Daily Mail, artículos científicos, wikiHow, y más
- Texto: Reseñas de IMDb, preguntas de lenguaje natural, y más
- Traducción: Varios conjuntos de datos de entrenamiento de traducción
- Video: Moving MNIST, Starcraft, y más

> TensorFlow Datasets es una instalación separada de TensorFlow, ¡así que asegúrate de instalarlo antes de probar cualquier ejemplo! Si estás usando Google Colab, ya está preinstalado.

Este capítulo te introducirá a TFDS y cómo puedes usarlo para simplificar enormemente el proceso de entrenamiento. Exploraremos la estructura subyacente de TFRecord y cómo puede proporcionar una base común independientemente del tipo de datos subyacente. También aprenderás sobre el patrón Extract-Transform-Load (ETL) utilizando TFDS, que se puede usar para entrenar modelos con grandes cantidades de datos de manera eficiente.

## Comenzando con TFDS
Veamos algunos ejemplos simples de cómo usar TFDS para ilustrar cómo nos proporciona una interfaz estándar para nuestros datos, independientemente del tipo de datos.

Si necesitas instalarlo, puedes hacerlo con un comando pip:

```python
pip install tensorflow-datasets
```

Una vez instalado, puedes usarlo para acceder a un conjunto de datos con tfds.load, pasando el nombre del conjunto de datos deseado. Por ejemplo, si quieres usar Fashion MNIST, puedes usar un código como este:

```python
import tensorflow as tf  
import tensorflow_datasets as tfds  
mnist_data = tfds.load("fashion_mnist")  
for item in mnist_data:  
    print(item)  
```

Asegúrate de inspeccionar el tipo de datos que obtienes como resultado del comando tfds.load: la salida al imprimir los elementos serán las diferentes divisiones que están disponibles de forma nativa en los datos. En este caso, es un diccionario que contiene dos cadenas, test y train. Estas son las divisiones disponibles.

Si deseas cargar estas divisiones en un conjunto de datos que contenga los datos reales, simplemente puedes especificar la división que deseas en el comando tfds.load, como esto:

```python
mnist_train = tfds.load(name="fashion_mnist", split="train")  
assert isinstance(mnist_train, tf.data.Dataset)  
print(type(mnist_train))  
```

En este caso, verás que la salida es un DatasetAdapter, que puedes iterar para inspeccionar los datos. Una característica interesante de este adaptador es que puedes simplemente llamar a take(1) para obtener el primer registro. Hagámoslo para inspeccionar cómo lucen los datos:

```python
for item in mnist_train.take(1):  
    print(type(item))  
    print(item.keys())  
```

La salida del primer print mostrará que el tipo de item en cada registro es un diccionario. Cuando imprimimos las claves de ese diccionario, veremos que en este conjunto de imágenes los tipos son image y label. Entonces, si queremos inspeccionar un valor en el conjunto de datos, podemos hacer algo como esto:

```python
for item in mnist_train.take(1):  
    print(type(item))  
    print(item.keys())  
    print(item['image'])  
    print(item['label'])  
```

Verás que la salida para la imagen es un arreglo de 28 × 28 valores (en un tf.Tensor) de 0 a 255 que representan la intensidad de los píxeles. La etiqueta se mostrará como tf.Tensor(2, shape=(), dtype=int64), indicando que esta imagen pertenece a la clase 2 en el conjunto de datos.

También puedes obtener información sobre el conjunto de datos utilizando el parámetro with_info al cargar el conjunto de datos, así:

```python
mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")  
print(info)  
```

Imprimir info te dará detalles sobre el contenido del conjunto de datos. Por ejemplo, para Fashion MNIST, verás una salida como esta:

```python
tfds.core.DatasetInfo(  
    name='fashion_mnist', version=3.0.0,  
    description='Fashion-MNIST is a dataset of Zalando\'s article images  
    consisting of a training set of 60,000 examples and a test set of 10,000  
    examples. Each example is a 28x28 grayscale image, associated with a  
    label from 10 classes.',  
    homepage='https://github.com/zalandoresearch/fashion-mnist',  
    features=FeaturesDict({  
        'image': Image(shape=(28, 28, 1), dtype=tf.uint8),  
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),  
    }),  
    total_num_examples=70000,  
    splits={  
        'test': 10000,  
        'train': 60000,  
    },  
    supervised_keys=('image', 'label'),  
    citation="""@article{DBLP:journals/corr/abs-1708-07747,  
          author    = {Han Xiao and  
                       Kashif Rasul and  
                       Roland Vollgraf},  
          title     = {Fashion-MNIST: a Novel Image Dataset for Benchmarking  
                       Machine Learning Algorithms},  
          journal   = {CoRR},  
          volume    = {abs/1708.07747},  
          year      = {2017},  
          url       = {http://arxiv.org/abs/1708.07747},  
          archivePrefix = {arXiv},  
          eprint    = {1708.07747},  
          timestamp = {Mon, 13 Aug 2018 16:47:27 +0200},  
          biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1708-07747},  
          bibsource = {dblp computer science bibliography, https://dblp.org}  
    }""",  
    redistribution_info=,  
)  
```

Dentro de esto, puedes ver detalles como las divisiones (como se demostró anteriormente) y las características dentro del conjunto de datos, así como información adicional como la cita, descripción y versión del conjunto de datos.

## Usando TFDS con Modelos Keras
En el Capítulo 2, viste cómo crear un modelo de visión por computadora simple utilizando TensorFlow y Keras, con los conjuntos de datos integrados en Keras (incluido Fashion MNIST), utilizando un código simple como este:

```python
mnist = tf.keras.datasets.fashion_mnist  
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()  
```

Al usar TFDS, el código es muy similar, pero con algunos cambios menores. Los conjuntos de datos de Keras nos dieron tipos ndarray que funcionaban de forma nativa en model.fit, pero con TFDS necesitamos hacer un pequeño trabajo de conversión:

```python
(training_images, training_labels),  
(test_images, test_labels) =  tfds.as_numpy(tfds.load('fashion_mnist',  
                                                        split = ['train', 'test'],  
                                                        batch_size=-1,  
                                                        as_supervised=True))  
```

En este caso, usamos tfds.load, pasándole fashion_mnist como el conjunto de datos deseado. Sabemos que tiene divisiones de entrenamiento y prueba, por lo que pasar estas en un arreglo nos devolverá un arreglo de adaptadores de conjuntos de datos con las imágenes y etiquetas en ellos. Usar tfds.as_numpy en la llamada a tfds.load hace que se devuelvan como arreglos Numpy. Especificar batch_size=-1 nos da todos los datos, y as_supervised=True asegura que obtengamos tuplas de (entrada, etiqueta) devueltas.

Una vez hecho esto, tenemos prácticamente el mismo formato de datos que estaba disponible en los conjuntos de datos de Keras, con una modificación: la forma en TFDS es (28, 28, 1), mientras que en los conjuntos de datos de Keras era (28, 28).

Esto significa que el código necesita cambiar un poco para especificar que la forma de los datos de entrada es (28, 28, 1) en lugar de (28, 28):

```python
import tensorflow as tf  
import tensorflow_datasets as tfds  

(training_images, training_labels), (test_images, test_labels) =  
tfds.as_numpy(tfds.load('fashion_mnist', split = ['train', 'test'],  
batch_size=-1, as_supervised=True))  

training_images = training_images / 255.0  
test_images = test_images / 255.0  

model = tf.keras.models.Sequential([  
    tf.keras.layers.Flatten(input_shape=(28,28,1)),  
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  
])  

model.compile(optimizer='adam',  
loss='sparse_categorical_crossentropy',  
metrics=['accuracy'])  

model.fit(training_images, training_labels, epochs=5)  
```

Para un ejemplo más complejo, puedes ver el conjunto de datos Horses or Humans utilizado en el Capítulo 3. Este también está disponible en TFDS. Aquí está el código completo para entrenar un modelo con él:

```python
import tensorflow as tf  
import tensorflow_datasets as tfds  

data = tfds.load('horses_or_humans', split='train', as_supervised=True)  
train_batches = data.shuffle(100).batch(10)  

model = tf.keras.models.Sequential([  
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',  
    input_shape=(300, 300, 3)),  
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

model.compile(optimizer='Adam', loss='binary_crossentropy',  
metrics=['accuracy'])  

history = model.fit(train_batches, epochs=10)  
```

Como puedes ver, es bastante sencillo: simplemente llama a tfds.load, pasándole la división que deseas (en este caso train), y úsala en el modelo. Los datos se agrupan y se mezclan para hacer el entrenamiento más efectivo.

El conjunto de datos Horses or Humans se divide en conjuntos de entrenamiento y prueba, por lo que si deseas validar tu modelo mientras entrenas, puedes hacerlo cargando un conjunto de validación separado desde TFDS como este:

```python
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)  
```

Necesitarás agruparlo, al igual que hiciste con el conjunto de entrenamiento. Por ejemplo:

```python
validation_batches = val_data.batch(32)
```

Luego, al entrenar, especificas los datos de validación como estos lotes. También tienes que establecer explícitamente el número de pasos de validación a usar por época, o TensorFlow lanzará un error. Si no estás seguro, simplemente configúralo en 1 como este:

```python
history = model.fit(train_batches, epochs=10,  
validation_data=validation_batches, validation_steps=1)  
```

### Cargando Versiones Específicas
Todos los conjuntos de datos almacenados en TFDS usan un sistema de numeración MAJOR.MINOR.PATCH. Las garantías de este sistema son las siguientes. Si PATCH se actualiza, los datos devueltos por una llamada son idénticos, pero la organización subyacente puede haber cambiado. Cualquier cambio debería ser invisible para los desarrolladores. Si MINOR se actualiza, los datos aún no cambian, con la excepción de que puede haber características adicionales en cada registro (cambios no disruptivos). Además, para cualquier segmento particular (ver "Using Custom Splits" en la página 74) los datos serán los mismos, por lo que los registros no se reordenan. Si MAJOR se actualiza, entonces puede haber cambios en el formato de los registros y su ubicación, de modo que segmentos particulares pueden devolver valores diferentes.

Cuando inspeccionas conjuntos de datos, verás cuando hay diferentes versiones disponibles. Por ejemplo, este es el caso para el conjunto de datos cnn_dailymail. Si no deseas el predeterminado, que al momento de escribir era 3.0.0, y en su lugar deseas una versión anterior, como 1.0.0, puedes simplemente cargarlo así:

```python
data, info = tfds.load("cnn_dailymail:1.0.0", with_info=True)
```

Ten en cuenta que si estás usando Colab, siempre es una buena idea verificar la versión de TFDS que utiliza. Al momento de escribir, Colab estaba preconfigurado para TFDS 2.0, pero hay algunos errores al cargar conjuntos de datos (incluido el cnn_dailymail) que se han corregido en TFDS 2.1 y versiones posteriores, así que asegúrate de usar una de esas versiones, o al menos instálalas en Colab, en lugar de confiar en el predeterminado integrado.

## Usando funciones de mapeo para la aumentación
En el Capítulo 3, viste las útiles herramientas de aumentación disponibles al usar un ImageDataGenerator para proporcionar datos de entrenamiento para tu modelo. Podrías preguntarte cómo lograr lo mismo al usar TFDS, ya que no estás procesando las imágenes desde un subdirectorio como antes. La mejor manera de lograr esto, o cualquier otra forma de transformación, es usar una función de mapeo en el adaptador de datos. Veamos cómo hacerlo.

Anteriormente, con nuestros datos de Horses or Humans, simplemente cargamos los datos desde TFDS y creamos lotes como este:

```python
data = tfds.load('horses_or_humans', split='train', as_supervised=True)  
train_batches = data.shuffle(100).batch(10)  
```

Para realizar transformaciones y asignarlas al conjunto de datos, puedes crear una función de mapeo. Esto es simplemente código estándar de Python. Por ejemplo, supongamos que creas una función llamada augmentimages y haces que realice alguna aumentación de imágenes, como esta:

```python
def augmentimages(image, label):  
    image = tf.cast(image, tf.float32)  
    image = (image / 255)  
    image = tf.image.random_flip_left_right(image)  
    return image, label  
```

Luego, puedes mapear esto a los datos para crear un nuevo conjunto de datos llamado train:

```python
train = data.map(augmentimages)  
```

Luego, al crear los lotes, hazlo desde train en lugar de desde data, como esto:

```python
train_batches = train.shuffle(100).batch(32)  
```

Puedes ver en la función augmentimages que hay un volteo aleatorio hacia la izquierda o derecha de la imagen, hecho usando tf.image.random_flip_left_right(image). Hay muchas funciones en la biblioteca tf.image que puedes usar para la aumentación; consulta la documentación para más detalles.

### Usando TensorFlow Addons
La biblioteca TensorFlow Addons contiene aún más funciones que puedes usar. Algunas de las funciones de aumentación en ImageDataGenerator (como rotate) solo se pueden encontrar allí, por lo que es una buena idea revisarla.

Usar TensorFlow Addons es bastante sencillo: simplemente instala la biblioteca con:

```python
pip install tensorflow-addons  
```

Una vez hecho esto, puedes integrar los addons en tu función de mapeo. Aquí hay un ejemplo donde se usa el addon rotate en la función de mapeo anterior:

```python
import tensorflow_addons as tfa  

def augmentimages(image, label):  
    image = tf.cast(image, tf.float32)  
    image = (image / 255)  
    image = tf.image.random_flip_left_right(image)  
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')  
    return image, label  
```

## Usando divisiones personalizadas
Hasta este punto, todos los datos que has estado usando para construir modelos han sido predivididos en conjuntos de entrenamiento y prueba. Por ejemplo, con Fashion MNIST tenías 60,000 y 10,000 registros, respectivamente. Pero, ¿y si no quieres usar esas divisiones? ¿Y si deseas dividir los datos tú mismo según tus necesidades? Ese es uno de los aspectos más poderosos de TFDS: viene completo con una API que te da un control granular y detallado sobre cómo dividir tus datos.

De hecho, ya lo has visto al cargar datos como este:

```python
data = tfds.load('cats_vs_dogs', split='train', as_supervised=True)  
```

Nota que el parámetro split es una cadena, y en este caso estás pidiendo la división train, que resulta ser el conjunto de datos completo. Si estás familiarizado con la notación de slices de Python, también puedes usarla. Esta notación se puede resumir como la definición de los fragmentos deseados dentro de corchetes, como este: [<inicio>:<fin>:<paso>]. Es una sintaxis bastante sofisticada que te da gran flexibilidad.

Por ejemplo, si deseas que los primeros 10,000 registros de train sean tus datos de entrenamiento, puedes omitir <inicio> y simplemente llamar a train[:10000] (un recordatorio útil es leer los dos puntos iniciales como “los primeros,” por lo que esto se leería como “entrena los primeros 10,000 registros”):

```python
data = tfds.load('cats_vs_dogs', split='train[:10000]', as_supervised=True)  
```

También puedes usar % para especificar la división. Por ejemplo, si deseas que el primer 20% de los registros se use para el entrenamiento, puedes usar :20% como este:

```python
data = tfds.load('cats_vs_dogs', split='train[:20%]', as_supervised=True)  
```

Incluso podrías combinar divisiones. Es decir, si deseas que tus datos de entrenamiento sean una combinación de los primeros y los últimos mil registros, podrías hacer lo siguiente (donde -1000: significa “los últimos 1,000 registros” y :1000 significa “los primeros 1,000 registros”):

```python
data = tfds.load('cats_vs_dogs', split='train[-1000:]+train[:1000]', as_supervised=True)  
```

El conjunto de datos Dogs vs. Cats no tiene divisiones fijas de entrenamiento, prueba y validación, pero con TFDS, crearlas es sencillo. Supongamos que deseas una división de 80%, 10%, 10%. Podrías crear los tres conjuntos como este:

```python
train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)  
validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)  
test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)  
```

Una vez que los tienes, puedes usarlos como cualquier división nombrada.

Un detalle es que debido a que los conjuntos de datos que se devuelven no se pueden interrogar para obtener su longitud, a menudo es difícil verificar que hayas dividido correctamente el conjunto original. Para ver cuántos registros tienes en una división, debes iterar a través de todo el conjunto y contarlos uno por uno. Aquí está el código para hacerlo en el conjunto de entrenamiento que acabas de crear:

```python
train_length = [i for i, _ in enumerate(train_data)][-1] + 1  
print(train_length)  
```

Este proceso puede ser lento, ¡así que asegúrate de usarlo solo cuando estés depurando!

## Entendiendo TFRecord

Cuando usas TFDS, tus datos se descargan y se almacenan en caché en el disco para que no necesites descargarlos cada vez que los uses. TFDS utiliza el formato TFRecord para el almacenamiento en caché. Si observas detenidamente mientras los datos se descargan, notarás esto. Por ejemplo, la Figura 4-1 muestra cómo se descarga, mezcla y escribe el conjunto de datos cnn_dailymail en un archivo TFRecord.

![Figura 4-1. Descarga del conjunto de datos cnn_dailymail como un archivo TFRecord.](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure4.1.png)

Este es el formato preferido en TensorFlow para almacenar y recuperar grandes cantidades de datos. Es una estructura de archivo muy simple, leída secuencialmente para un mejor rendimiento. En el disco, el archivo es bastante directo, con cada registro que consiste en un entero que indica la longitud del registro, un código de redundancia cíclica (CRC) de eso, un arreglo de bytes de los datos y un CRC de ese arreglo de bytes. Los registros se concatenan en el archivo y luego se dividen en fragmentos en caso de conjuntos de datos grandes.

Por ejemplo, la Figura 4-2 muestra cómo el conjunto de entrenamiento de cnn_dailymail se divide en 16 archivos después de la descarga.

Para observar un ejemplo más simple, descarga el conjunto de datos MNIST e imprime su información:

```python
data, info = tfds.load("mnist", with_info=True)  
print(info)  
```

Dentro de la información, verás que sus características están almacenadas así:

```python
features=FeaturesDict({  
    'image': Image(shape=(28, 28, 1), dtype=tf.uint8),  
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),  
}),
```

Similar al ejemplo de CNN/DailyMail, el archivo se descarga en /root/tensorflow_datasets/mnist/<version>/files.

Puedes cargar los registros sin procesar como un TFRecordDataset de esta manera:

```python
filename="/root/tensorflow_datasets/mnist/3.0.0/mnist-test.tfrecord-00000-of-00001"  
raw_dataset = tf.data.TFRecordDataset(filename)  
for raw_record in raw_dataset.take(1):  
    print(repr(raw_record))  
```

Ten en cuenta que la ubicación de tu archivo puede ser diferente dependiendo de tu sistema operativo.

![Figura 4-2. Inspección de los TFRecords para cnn_dailymail.](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure4.2.png)

Esto imprimirá el contenido sin procesar del registro, así:

```python
<tf.Tensor: shape=(), dtype=string, numpy=b"\n\x85\x03\n\xf2\x02\n\x05image\x12\xe8\x02\n\xe5\x02\n\xe2\x02\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x1c\x00\x00\x00\x1c\x08\x00\x00\x00\x00Wf\x80H\x00\x00\x01)IDAT(\x91\xc5\xd2\xbdK\xc3P\x14\x05\xf0S(v\x13)\x04,.\x82\xc5Aq\xac\xedb\x1d\xdc\n.\x12\x87n\x0e\x82\x93\x7f@Q\xb2\x08\xba\tbQ0.\xe2\xe2\xd4\xb1\xa2h\x9c\x82\xba\x8a(\nq\xf0\x83Fh\x95\n6\x88\xe7R\x87\x88\xf9\xa8Y\xf5\x0e\x8f\xc7\xfd\xdd\x0b\x87\xc7\x03\xfe\xbeb\x9d\xadT\x927Q\xe3\xe9\x07:\xab\xbf\xf4\xf3\xcf\xf6\x8a\xd9\x14\xd29\xea\xb0\x1eKH\xde\xab\xea%\xaba\x1b=\xa4P/\xf5\x02\xd7\\\x07\x00\xc4=,L\xc0,>\x01@2\xf6\x12\xde\x9c\xde[t/\xb3\x0e\x87\xa2\xe2\xc2\xe0A<\xca\xb26\xd5(\x1b\xa9\xd3\xe8\x0e\xf5\x86\x17\xceE\xdarV\xae\xb7_\xf3AR\r!I\xf7(\x06m\xaaE\xbb\xb6\xac\r*\x9b$e<\xb8\xd7\xa2\x0e\x00\xd0l\x92\xb2\xd5\x15\xcc\xae'\x00\xf4m\x08O'+\xc2y\x9f\x8d\xc9\x15\x80\xfe\x99[q\x962@CN|i\xf7\xa9!=\xd7\xab\x19\x00\xc8\xd6\xb8\xeb\xa1\xf0\xd8l\xca\xfb]\xee\xfb]*\x9fV\xe1\x07\xb7\xc9\x8b55\xe7M\xef\xb0\x04\xc0\xfd&\x89\x01<\xbe\xf9\x03*\x8a\xf5\x81\x7f\xaa/2y\x87ks\xec\x1e\xc1\x00\x00\x00\x00IEND\xaeB`\x82\n\x0e\n\x05label\x12\x05\x1a\x03\n\x01\x02">  
```

Es una cadena larga que contiene los detalles del registro, junto con sumas de verificación, etc. Pero si ya conocemos las características, podemos crear una descripción de características y usar esto para analizar los datos. Aquí está el código:

```python
# Crear una descripción de las características  
feature_description = {  
    'image': tf.io.FixedLenFeature([], dtype=tf.string),  
    'label': tf.io.FixedLenFeature([], dtype=tf.int64),  
}  
def _parse_function(example_proto):  
    # Analizar el proto de entrada `tf.Example` usando el diccionario anterior  
    return tf.io.parse_single_example(example_proto, feature_description)  

parsed_dataset = raw_dataset.map(_parse_function)  
for parsed_record in parsed_dataset.take(1):  
    print((parsed_record))  
```

La salida de esto es un poco más amigable. Primero, puedes ver que la imagen es un Tensor y que contiene un PNG. PNG es un formato de imagen comprimido con un encabezado definido por IHDR y los datos de la imagen entre IDAT y IEND. Si miras de cerca, puedes verlos en la secuencia de bytes. También está la etiqueta, almacenada como un entero y que contiene el valor 2:

```python
{'image': <tf.Tensor: shape=(), dtype=string, numpy=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x1c\x00\x00\x00\x1c\x08\x00\x00\x00\x00Wf\x80H\x00\x00\x01)IDAT(\x91\xc5\xd2\xbdK\xc3P\x14\x05\xf0S(v\x13)\x04,.\x82\xc5Aq\xac\xedb\x1d\xdc\n.\x12\x87n\x0e\x82\x93\x7f@Q\xb2\x08\xba\tbQ0.\xe2\xe2\xd4\xb1\xa2h\x9c\x82\xba\x8a(\nq\xf0\x83Fh\x95\n6\x88\xe7R\x87\x88\xf9\xa8Y\xf5\x0e\x8f\xc7\xfd\xdd\x0b\x87\xc7\x03\xfe\xbeb\x9d\xadT\x927Q\xe3\xe9\x07:\xab\xbf\xf4\xf3\xcf\xf6\x8a\xd9\x14\xd29\xea\xb0\x1eKH\xde\xab\xea%\xaba\x1b=\xa4P/\xf5\x02\xd7\\\x07\x00\xc4=,L\xc0,>\x01@2\xf6\x12\xde\x9c\xde[t/\xb3\x0e\x87\xa2\xe2\xc2\xe0A<\xca\xb26\xd5(\x1b\xa9\xd3\xe8\x0e\xf5\x86\x17\xceE\xdarV\xae\xb7_\xf3AR\r!I\xf7(\x06m\xaaE\xbb\xb6\xac\r*\x9b$e<\xb8\xd7\xa2\x0e\x00\xd0l\x92\xb2\xd5\x15\xcc\xae'\x00\xf4m\x08O'+\xc2y\x9f\x8d\xc9\x15\x80\xfe\x99[q\x962@CN|i\xf7\xa9!=\xd7\xab\x19\x00\xc8\xd6\xb8\xeb\xa1\xf0\xd8l\xca\xfb]\xee\xfb]*\x9fV\xe1\x07\xb7\xc9\x8b55\xe7M\xef\xb0\x04\xc0\xfd&\x89\x01<\xbe\xf9\x03*\x8a\xf5\x81\x7f\xaa/2y\x87ks\xec\x1e\xc1\x00\x00\x00\x00IEND\xaeB`\x82">, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=2>}
```

En este punto, puedes leer el TFRecord sin procesar y decodificarlo como un PNG utilizando una biblioteca decodificadora de PNG como Pillow.

## El proceso ETL para gestionar datos en TensorFlow
ETL es el patrón principal que TensorFlow utiliza para entrenar, sin importar la escala. Hemos estado explorando la construcción de modelos a pequeña escala en una sola computadora en este libro, pero la misma tecnología puede usarse para entrenamiento a gran escala en múltiples máquinas con conjuntos de datos masivos.

La fase de Extracción del proceso ETL es cuando los datos sin procesar se cargan desde donde están almacenados y se preparan de manera que puedan ser transformados. La fase de Transformación es cuando los datos se manipulan de una forma que los hace adecuados o mejorados para el entrenamiento. Por ejemplo, el agrupamiento en lotes (batching), la ampliación de imágenes, el mapeo a columnas de características y otras lógicas aplicadas a los datos pueden considerarse parte de esta fase. La fase de Carga es cuando los datos se cargan en la red neuronal para el entrenamiento.

Considera el código completo para entrenar el clasificador de "Caballos o Humanos", mostrado aquí. He añadido comentarios para mostrar dónde tienen lugar las fases de Extracción, Transformación y Carga:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# INICIO DE LA DEFINICIÓN DEL MODELO
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
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# FIN DE LA DEFINICIÓN DEL MODELO

# INICIO DE LA FASE DE EXTRACCIÓN
data = tfds.load('horses_or_humans', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
# FIN DE LA FASE DE EXTRACCIÓN

# INICIO DE LA FASE DE TRANSFORMACIÓN
def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    return image, label

train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(32)
validation_batches = val_data.batch(32)
# FIN DE LA FASE DE TRANSFORMACIÓN

# INICIO DE LA FASE DE CARGA
history = model.fit(
    train_batches, epochs=10, validation_data=validation_batches, validation_steps=1
)
# FIN DE LA FASE DE CARGA
```

Usar este proceso puede hacer que tus tuberías de datos sean menos susceptibles a cambios en los datos y en el esquema subyacente. Cuando usas TFDS para extraer datos, la misma estructura subyacente se utiliza sin importar si los datos son lo suficientemente pequeños como para caber en la memoria, o tan grandes que no pueden contenerse ni siquiera en una máquina sencilla.

Las APIs tf.data para la transformación también son consistentes, por lo que puedes usar unas similares sin importar la fuente de datos subyacente. Y, por supuesto, una vez transformados, el proceso de cargar los datos también es consistente, ya sea que estés entrenando en una sola CPU, una GPU, un clúster de GPUs o incluso pods de TPUs.

Sin embargo, cómo cargas los datos puede tener un gran impacto en la velocidad de entrenamiento. Echemos un vistazo a eso a continuación.

### Optimizando la fase de carga
Echemos un vistazo más de cerca al proceso de Extracción-Transformación-Carga cuando entrenamos un modelo. Podemos considerar que la extracción y transformación de los datos son posibles en cualquier procesador, incluyendo una CPU. De hecho, el código utilizado en estas fases para realizar tareas como descargar datos, descomprimirlos y procesarlos registro por registro no es para lo que están diseñados los GPUs o TPUs, por lo que este código probablemente se ejecutará en la CPU de todos modos. Sin embargo, cuando se trata de entrenar, puedes obtener grandes beneficios de un GPU o TPU, por lo que tiene sentido usar uno para esta fase si es posible. Así, en la situación en la que tengas un GPU o TPU disponible, lo ideal sería dividir la carga de trabajo entre la CPU y el GPU/TPU, con la Extracción y Transformación ocurriendo en la CPU, y la Carga ocurriendo en el GPU/TPU.

Supongamos que estás trabajando con un conjunto de datos grande. Suponiendo que es tan grande que debes preparar los datos (es decir, realizar la extracción y transformación) en lotes, terminarás con una situación como la que se muestra en la Figura 4-3. Mientras se está preparando el primer lote, el GPU/TPU está inactivo. Cuando ese lote esté listo, se puede enviar al GPU/TPU para el entrenamiento, pero ahora la CPU está inactiva hasta que se termine el entrenamiento, momento en el que puede comenzar a preparar el segundo lote. Hay mucho tiempo inactivo aquí, por lo que podemos ver que hay espacio para optimización.

![Figura 4-3. Entrenamiento en un CPU/GPU](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure4.3.png)

La solución lógica es hacer el trabajo en paralelo, preparando y entrenando de manera simultánea. Este proceso se llama pipelining y se ilustra en la Figura 4-4.

![Figura 4-4. Pipelining](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure4.4.png)

En este caso, mientras la CPU prepara el primer lote, el GPU/TPU nuevamente no tiene nada en lo que trabajar, por lo que está inactivo. Cuando el primer lote esté listo, el GPU/TPU puede comenzar el entrenamiento, pero en paralelo con esto, la CPU preparará el segundo lote. Por supuesto, el tiempo que tarda en entrenar el lote n - 1 y preparar el lote n no siempre será el mismo. Si el tiempo de entrenamiento es más rápido, tendrás períodos de tiempo inactivo en el GPU/TPU. Si es más lento, tendrás períodos de tiempo inactivo en la CPU. Elegir el tamaño de lote correcto puede ayudarte a optimizar esto, y como el tiempo en el GPU/TPU probablemente sea más costoso, probablemente querrás reducir su tiempo inactivo tanto como sea posible.

Probablemente notaste que cuando pasamos de conjuntos de datos simples como Fashion MNIST en Keras a usar las versiones de TFDS, tenías que agruparlos antes de poder entrenar. Esta es la razón: el modelo de pipelining está en su lugar para que, sin importar cuán grande sea tu conjunto de datos, sigas utilizando un patrón consistente para el ETL en él.

### Paralelizando ETL para mejorar el rendimiento del entrenamiento
TensorFlow te da todas las API que necesitas para paralelizar el proceso de Extracción y Transformación. Echemos un vistazo a cómo se ven utilizando Dogs vs. Cats y las estructuras subyacentes de TFRecord.

Primero, usas tfds.load para obtener el conjunto de datos:

```python
train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)
```

Si quieres usar los TFRecords subyacentes, necesitarás acceder a los archivos originales que fueron descargados. Como el conjunto de datos es grande, está fragmentado a través de varios archivos (ocho, en la versión 4.0.0). Puedes crear una lista de estos archivos y usar tf.Data.Dataset.list_files para cargarlos:

```python
file_pattern = f'/root/tensorflow_datasets/cats_vs_dogs/4.0.0/cats_vs_dogs-train.tfrecord*'
files = tf.data.Dataset.list_files(file_pattern)
```

Una vez que tengas los archivos, puedes cargarlos en un conjunto de datos usando files.interleave así:

```python
train_dataset = files.interleave(
    tf.data.TFRecordDataset, 
    cycle_length=4,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
```

Aquí hay algunos conceptos nuevos, así que tomémonos un momento para explorarlos.
El parámetro cycle_length especifica el número de elementos de entrada que se procesan concurrentemente. Entonces, en un momento verás la función de mapeo que decodifica los registros mientras se cargan desde el disco. Como cycle_length está establecido en 4, este proceso manejará cuatro registros a la vez. Si no especificas este valor, se derivará del número de núcleos de CPU disponibles.

El parámetro num_parallel_calls, cuando se establece, especificará el número de llamadas paralelas que se ejecutarán. Usar tf.data.experimental.AUTOTUNE, como se hace aquí, hará que tu código sea más portátil porque el valor se establece dinámicamente, basado en los CPUs disponibles. Cuando se combina con cycle_length, estás estableciendo el grado máximo de paralelismo. Entonces, por ejemplo, si num_parallel_calls se ajusta automáticamente a 6 y cycle_length es 4, tendrás seis hilos separados, cada uno cargando cuatro registros a la vez.

Ahora que el proceso de Extracción está paralelizado, exploremos cómo paralelizar la transformación de los datos. Primero, crea la función de mapeo que carga el TFRecord crudo y lo convierte en contenido utilizable, por ejemplo, decodificando una imagen JPEG en un búfer de imagen:

```python
def read_tfrecord(serialized_example):
    feature_description={
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (300,300))
    return image, example['label']
```

Como puedes ver, esta es una función de mapeo típica sin trabajo específico hecho para que funcione en paralelo. Eso se hará cuando llamemos la función de mapeo. Así es como hacerlo:

```python
cores = multiprocessing.cpu_count()
print(cores)
train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
train_dataset = train_dataset.cache()
```

Primero, si no quieres autotunear, puedes usar la biblioteca multiprocessing para obtener un conteo de tus CPUs. Luego, cuando llames a la función de mapeo, solo pasas esto como el número de llamadas paralelas que deseas hacer. Es realmente tan simple como eso.
El método cache almacenará en caché el conjunto de datos en la memoria. Si tienes mucha RAM disponible, esto es una aceleración realmente útil. Intentar esto en Colab con Dogs vs. Cats probablemente hará que tu VM se caiga debido a que el conjunto de datos no cabe en la RAM. Después de eso, si está disponible, la infraestructura de Colab te dará una nueva máquina con más RAM.

El cargado y entrenamiento también pueden ser paralelizados. Además de mezclar y agrupar los datos, puedes prefetch según el número de núcleos de CPU disponibles. Aquí está el código:

```python
train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

Una vez que tu conjunto de entrenamiento esté completamente paralelizado, puedes entrenar el modelo como antes:

```python
model.fit(train_dataset, epochs=10, verbose=1)
```

Cuando probé esto en Google Colab, descubrí que este código adicional para paralelizar el proceso de ETL redujo el tiempo de entrenamiento a aproximadamente 40 segundos por época, en lugar de 75 segundos sin él. ¡Estos simples cambios redujeron mi tiempo de entrenamiento casi a la mitad!

## Resumen
Este capítulo presentó TensorFlow Datasets, una biblioteca que te da acceso a una gran variedad de conjuntos de datos, desde pequeños hasta grandes conjuntos utilizados en la investigación. Viste cómo usan una API común y un formato común para ayudar a reducir la cantidad de código que tienes que escribir para acceder a los datos. También viste cómo usar el proceso ETL, que está en el corazón del diseño de TFDS, y en particular exploramos cómo paralelizar la extracción, transformación y carga de datos para mejorar el rendimiento del entrenamiento. En el próximo capítulo, tomarás lo que has aprendido y comenzarás a aplicarlo a problemas de procesamiento de lenguaje natural.