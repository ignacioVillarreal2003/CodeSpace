# Introducción al Procesamiento del Lenguaje Natural
El procesamiento del lenguaje natural (NLP, por sus siglas en inglés) es una técnica de inteligencia artificial que se ocupa de la comprensión del lenguaje humano. Implica técnicas de programación para crear un modelo que pueda entender el lenguaje, clasificar contenido, e incluso generar y crear nuevas composiciones en lenguaje humano. Exploraremos estas técnicas en los próximos capítulos. También hay muchos servicios que utilizan NLP para crear aplicaciones como chatbots, pero eso no está dentro del alcance de este libro. En su lugar, analizaremos los fundamentos del NLP y cómo modelar el lenguaje para que puedas entrenar redes neuronales que comprendan y clasifiquen texto. Como un poco de diversión, también verás cómo usar los elementos predictivos de un modelo de aprendizaje automático para escribir algo de poesía.

Comenzaremos este capítulo examinando cómo descomponer el lenguaje en números y cómo esos números pueden usarse en redes neuronales.

## Codificando el lenguaje en números
Puedes codificar el lenguaje en números de muchas maneras. La más común es codificar por letras, como se hace naturalmente cuando las cadenas de texto se almacenan en tu programa. En memoria, sin embargo, no se almacena la letra a como tal, sino una codificación de ella—quizás un valor ASCII, Unicode u otro. Por ejemplo, considera la palabra listen. Esto puede codificarse en ASCII como los números 76, 73, 83, 84, 69 y 78. Esto es útil, ya que ahora puedes usar valores numéricos para representar la palabra. Pero luego considera la palabra silent, que es un antigram de listen. Los mismos números representan esa palabra, aunque en un orden diferente, lo que podría dificultar la construcción de un modelo que entienda el texto.

> Un antigram es una palabra que es un anagrama de otra, pero tiene un significado opuesto. Por ejemplo, united y untied son antigrams, al igual que restful y fluster, Santa y Satan, forty-five y over fifty. Mi antiguo título laboral solía ser Developer Evangelist, pero desde entonces cambió a Developer Advocate, lo cual es algo positivo porque Evangelist es un antigram de Evil’s Agent!

Una mejor alternativa podría ser usar números para codificar palabras completas en lugar de las letras dentro de ellas. En ese caso, silent podría ser el número x y listen el número y, y no se superpondrían entre sí.

Usando esta técnica, considera una oración como "I love my dog." Podrías codificarla con los números [1, 2, 3, 4]. Si luego quisieras codificar "I love my cat.", podría ser [1, 2, 3, 5]. Ya has llegado al punto en el que puedes notar que las oraciones tienen un significado similar porque son similares numéricamente—[1, 2, 3, 4] se parece mucho a [1, 2, 3, 5].

Este proceso se llama tokenización, y a continuación explorarás cómo hacerlo en código.

### Introducción a la Tokenización
TensorFlow Keras contiene una biblioteca llamada preprocessing que ofrece una serie de herramientas extremadamente útiles para preparar datos para el aprendizaje automático. Una de ellas es un Tokenizer, que te permite convertir palabras en tokens. Veámoslo en acción con un ejemplo simple:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

En este caso, creamos un objeto Tokenizer y especificamos el número máximo de palabras que puede tokenizar. Este será el máximo número de tokens que se generarán a partir del corpus de palabras. Aquí tenemos un corpus muy pequeño que contiene solo seis palabras únicas, por lo que estaremos muy por debajo del límite de cien especificado.

Una vez que tenemos un tokenizer, llamar a fit_on_texts creará el índice de palabras tokenizadas. Imprimir esto mostrará un conjunto de pares clave/valor para las palabras en el corpus, como este:

```python
{'today': 1, 'is': 2, 'a': 3, 'day': 4, 'sunny': 5, 'rainy': 6}
```

El tokenizer es bastante flexible. Por ejemplo, si ampliáramos el corpus con otra oración que contenga la palabra “today” pero con un signo de interrogación al final, los resultados mostrarían que sería lo suficientemente inteligente como para filtrar “today?” y tratarlo simplemente como “today”:

```python
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]
```

El resultado será:

```python
{'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6, 'it': 7}
```

Este comportamiento está controlado por el parámetro filters del tokenizer, que por defecto elimina toda la puntuación excepto el apóstrofe. Por ejemplo, “Today is a sunny day” se convertiría en una secuencia que contiene [1, 2, 3, 4, 5] con las codificaciones mencionadas anteriormente, y “Is it sunny today?” se convertiría en [2, 7, 4, 1]. Una vez que las palabras en tus oraciones estén tokenizadas, el siguiente paso es convertirlas en listas de números, donde el número corresponde al valor asociado a cada palabra en el índice.

### Convirtiendo Oraciones en Secuencias
Ahora que has visto cómo convertir palabras en números tokenizados, el siguiente paso es codificar las oraciones en secuencias de números. El tokenizer tiene un método llamado text_to_sequences—todo lo que tienes que hacer es pasarle tu lista de oraciones, y te devolverá una lista de secuencias. Por ejemplo, si modificas el código anterior de la siguiente manera:

```python
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
```

Obtendrás las secuencias que representan las tres oraciones. Recordando que el índice de palabras es:

```python
{'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6, 'it': 7}
```

El resultado será algo como esto:

```python
[[1, 2, 3, 4, 5], [1, 2, 3, 6, 5], [2, 7, 4, 1]]
```

Puedes sustituir los números por las palabras correspondientes y verás que las oraciones tienen sentido.

Ahora considera qué sucede si estás entrenando una red neuronal con un conjunto de datos. El patrón típico es que tienes un conjunto de datos para entrenar que sabes que no cubrirá el 100% de tus necesidades, pero esperas que cubra lo máximo posible. En el caso del NLP, podrías tener miles de palabras en tus datos de entrenamiento, utilizadas en diferentes contextos, pero no puedes tener cada posible palabra en cada posible contexto.

Entonces, cuando muestras a tu red neuronal un texto nuevo, previamente desconocido, que contiene palabras no vistas antes, ¿qué podría suceder? Lo adivinaste: se confundirá porque simplemente no tiene contexto para esas palabras, y, como resultado, cualquier predicción que haga se verá afectada negativamente.

#### Uso de tokens fuera del vocabulario
Una herramienta para manejar situaciones donde hay palabras desconocidas es un token fuera del vocabulario (OOV). Este puede ayudar a tu red neuronal a entender el contexto de datos con texto previamente no visto. Por ejemplo, dado el pequeño corpus anterior, supongamos que quieres procesar oraciones como estas:

```python
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]
```

Recuerda que no estás agregando esta entrada al corpus de texto existente (que puedes considerar como tus datos de entrenamiento), sino pensando cómo una red preentrenada podría interpretar este texto. Si lo tokenizas con las palabras ya usadas y el tokenizador existente, así:

```python
test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
```

Tus resultados se verán así:

```python
{'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6, 'it': 7}
[[1, 2, 3, 5], [7, 6]]
```

Por lo tanto, las nuevas oraciones, al cambiar los tokens por palabras, serían “today is a day” y “it rainy.”

Como puedes ver, se pierde todo el contexto y significado. Un token fuera del vocabulario podría ayudar aquí, y puedes especificarlo en el tokenizador. Esto se hace agregando un parámetro llamado oov_token. Puedes asignarle cualquier cadena, asegurándote de que no aparezca en el corpus:

```python
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  
sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
```

Verás que la salida ha mejorado un poco:

```python
{'<OOV>': 1, 'today': 2, 'is': 3, 'a': 4, 'sunny': 5, 'day': 6, 'rainy': 7, 'it': 8}
[[2, 3, 4, 1, 6], [1, 8, 1, 7, 1]]
```

Tu lista de tokens tiene un nuevo elemento, < OOV >, y las oraciones de prueba mantienen su longitud. Al revertir la codificación, obtendrás: “today is a < OOV > day” y “< OOV > it < OOV > rainy < OOV >.”

La primera está más cerca del significado original. La segunda, debido a que la mayoría de sus palabras no están en el corpus, aún carece de mucho contexto, pero es un paso en la dirección correcta.

#### Entendiendo el padding
Al entrenar redes neuronales, típicamente necesitas que todos tus datos tengan la misma forma. Recuerda de capítulos anteriores que, al entrenar con imágenes, reformateaste las imágenes para que tuvieran el mismo ancho y alto. Con texto enfrentas el mismo problema: una vez que has tokenizado tus palabras y convertido tus oraciones en secuencias, estas pueden tener diferentes longitudes. Para que tengan el mismo tamaño y forma, puedes usar padding.

Para explorar el padding, agreguemos otra oración mucho más larga al corpus:

```python
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]
```

Cuando conviertes estas oraciones en secuencias, verás que las listas de números tienen diferentes longitudes:

```python
[
    [2, 3, 4, 5, 6],
    [2, 3, 4, 7, 6],
    [3, 8, 5, 2],
    [9, 10, 11, 12, 13, 14, 15, 2]
]
```

(Cuando imprimas las secuencias, todas estarán en una sola línea, pero aquí las he dividido en líneas separadas para mayor claridad). Si quieres que todas tengan la misma longitud, puedes usar la API pad_sequences. Primero, necesitas importarla:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

Usar esta API es muy sencillo. Para convertir tus secuencias (sin padding) en un conjunto con padding, simplemente llama a pad_sequences de esta forma:

```python
padded = pad_sequences(sequences)
print(padded)
```

Obtendrás un conjunto de secuencias con formato uniforme. También estarán en líneas separadas, como esta:

```python
[[ 0  0  0  2  3  4  5  6]
 [ 0  0  0  2  3  4  7  6]
 [ 0  0  0  0  3  8  5  2]
 [ 9 10 11 12 13 14 15  2]]
```

Las secuencias se rellenan con ceros, que no son un token en nuestra lista de palabras. Si te habías preguntado por qué la lista de tokens comenzaba en 1 cuando típicamente los programadores cuentan desde 0, ¡ahora lo sabes!

Ahora tienes algo con forma regular que puedes usar para el entrenamiento. Pero antes de ir allí, exploremos esta API un poco, porque te ofrece muchas opciones para mejorar tus datos.

Primero, puede que hayas notado que, en el caso de las oraciones más cortas, para que tengan la misma forma que la más larga, se añadieron la cantidad necesaria de ceros al principio. Esto se llama prepadding y es el comportamiento predeterminado. Puedes cambiar esto usando el parámetro padding. Por ejemplo, si quieres que tus secuencias se rellenen con ceros al final, puedes usar:

```python
padded = pad_sequences(sequences, padding='post')
```

El resultado será:

```python
[[ 2  3  4  5  6  0  0  0]
 [ 2  3  4  7  6  0  0  0]
 [ 3  8  5  2  0  0  0  0]
 [ 9 10 11 12 13 14 15  2]]
```

Ahora las palabras están al inicio de las secuencias con padding, y los caracteres 0 están al final.

Otro comportamiento predeterminado que puede que hayas observado es que todas las oraciones se ajustaron a la longitud de la más larga. Es un valor predeterminado razonable porque significa que no pierdes datos. La desventaja es que obtienes mucho padding. Pero, ¿qué pasa si no quieres esto, tal vez porque tienes una oración extremadamente larga que provoca demasiado padding? Para solucionar eso, puedes usar el parámetro maxlen, especificando la longitud máxima deseada al llamar a pad_sequences, como este ejemplo:

```python
padded = pad_sequences(sequences, padding='post', maxlen=6)
```

El resultado será:

```python
[[ 2  3  4  5  6  0]
 [ 2  3  4  7  6  0]
 [ 3  8  5  2  0  0]
 [11 12 13 14 15  2]]
```

Ahora tus secuencias con padding tienen todas la misma longitud, y no hay demasiado padding. Sin embargo, has perdido algunas palabras de la oración más larga, que se han truncado desde el principio. ¿Qué pasa si no quieres perder las palabras del principio y prefieres que se trunquen al final de la oración? Puedes cambiar el comportamiento predeterminado con el parámetro truncating, así:

```python
padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
```

El resultado mostrará que la oración más larga ahora se trunca al final en lugar de al principio:

```python
[[ 2  3  4  5  6  0]
 [ 2  3  4  7  6  0]
 [ 3  8  5  2  0  0]
 [ 9 10 11 12 13 14]]
```

> TensorFlow admite el entrenamiento usando tensores "irregulares" (de formas diferentes), lo cual es perfecto para las necesidades del procesamiento del lenguaje natural (NLP). Usarlos es un poco más avanzado que lo que cubrimos en este libro, pero una vez que completes la introducción al NLP proporcionada en los próximos capítulos, puedes explorar la documentación para aprender más.

## Eliminando Stopwords y Limpiando Texto
En la siguiente sección verás algunos conjuntos de datos del mundo real, y descubrirás que a menudo hay texto que no quieres en tu dataset. Es posible que desees filtrar las llamadas stopwords, que son demasiado comunes y no aportan significado, como “el”, “y” y “pero”. También podrías encontrar muchas etiquetas HTML en tu texto, y sería útil tener una forma limpia de eliminarlas. Otras cosas que podrías querer filtrar incluyen palabras groseras, signos de puntuación o nombres. Más adelante exploraremos un conjunto de datos de tweets, que a menudo contienen el ID de usuario de alguien, y querremos filtrar esos también.

Aunque cada tarea es diferente según tu corpus de texto, hay tres cosas principales que puedes hacer para limpiar tu texto de manera programática.

La primera es eliminar etiquetas HTML. Afortunadamente, hay una biblioteca llamada BeautifulSoup que facilita esto. Por ejemplo, si tus oraciones contienen etiquetas HTML como < br >, se eliminarán con este código:

```python
from bs4 import BeautifulSoup  
soup = BeautifulSoup(sentence)  
sentence = soup.get_text()  
```

Una forma común de eliminar stopwords es tener una lista de stopwords y preprocesar tus oraciones, eliminando instancias de estas. Aquí tienes un ejemplo abreviado:

```python
stopwords = ["a", "about", "above", ... "yours", "yourself", "yourselves"]
```

Una lista completa de stopwords se puede encontrar en algunos de los ejemplos en línea para este capítulo. Luego, mientras iteras a través de tus oraciones, puedes usar un código como este para eliminar las stopwords de tus oraciones:

```python
words = sentence.split()  
filtered_sentence = ""  
for word in words:  
    if word not in stopwords:  
        filtered_sentence = filtered_sentence + word + " "  
sentences.append(filtered_sentence)  
```

Otra cosa que podrías considerar es eliminar la puntuación, que puede confundir a un eliminador de stopwords. El código mostrado anteriormente busca palabras rodeadas por espacios, por lo que una stopword seguida inmediatamente de un punto o una coma no será detectada.

Arreglar este problema es fácil con las funciones de traducción proporcionadas por la biblioteca de cadenas de Python. También incluye una constante, string.punctuation, que contiene una lista de signos de puntuación comunes, por lo que para eliminarlos de una palabra puedes hacer lo siguiente:

```python
import string  
table = str.maketrans('', '', string.punctuation)  
words = sentence.split()  
filtered_sentence = ""  
for word in words:  
    word = word.translate(table)  
    if word not in stopwords:  
        filtered_sentence = filtered_sentence + word + " "  
sentences.append(filtered_sentence)  
```

Aquí, antes de filtrar las stopwords, se elimina la puntuación de cada palabra en la oración. Así que, si al dividir una oración obtienes la palabra “it;”, esta se convertirá en “it” y luego se eliminará como una stopword. Sin embargo, ten en cuenta que al hacer esto podrías necesitar actualizar tu lista de stopwords. Es común que estas listas incluyan palabras abreviadas y contracciones como “you’ll”. El traductor cambiará “you’ll” a “youll”, y si quieres que esa palabra se filtre, deberás actualizar tu lista de stopwords para incluirla.

Seguir estos tres pasos te dará un conjunto de texto mucho más limpio para usar. Pero, por supuesto, cada conjunto de datos tendrá sus peculiaridades con las que necesitarás trabajar.

## Trabajando con Fuentes de Datos Reales
Ahora que has visto los conceptos básicos de obtener oraciones, codificarlas con un índice de palabras y secuenciar los resultados, puedes llevar esto al siguiente nivel utilizando algunos conjuntos de datos públicos bien conocidos y las herramientas que Python proporciona para convertirlos en un formato fácilmente secuenciable. Comenzaremos con un conjunto de datos donde gran parte del trabajo ya ha sido realizado para ti en TensorFlow Datasets: el conjunto de datos de IMDb. Después, adoptaremos un enfoque más práctico, procesando un conjunto de datos basado en JSON y un par de conjuntos de datos con valores separados por comas (CSV) que contienen datos de emociones.

### Obteniendo Texto de TensorFlow Datasets
Exploramos TFDS en el Capítulo 4, por lo que, si tienes problemas con alguno de los conceptos en esta sección, puedes revisarlo rápidamente allí. El objetivo de TFDS es facilitar el acceso a datos de una manera estandarizada. Proporciona acceso a varios conjuntos de datos basados en texto; exploraremos imdb_reviews, un conjunto de 50,000 reseñas de películas etiquetadas del Internet Movie Database (IMDb), cada una determinada como positiva o negativa en cuanto a sentimiento.

Este código cargará la partición de entrenamiento del conjunto de datos de IMDb e iterará sobre él, agregando el campo de texto que contiene la reseña a una lista llamada imdb_sentences. Las reseñas son una tupla que incluye el texto y una etiqueta que contiene el sentimiento de la reseña. Nota que al envolver la llamada tfds.load en tfds.as_numpy, aseguras que los datos se carguen como cadenas, no como tensores:

```python
imdb_sentences = []  
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))  
for item in train_data:  
    imdb_sentences.append(str(item['text']))  
```

Una vez que tengas las oraciones, puedes crear un tokenizer y ajustarlo a ellas como antes, además de crear un conjunto de secuencias:

```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)  
tokenizer.fit_on_texts(imdb_sentences)  
sequences = tokenizer.texts_to_sequences(imdb_sentences)  
```

También puedes imprimir tu índice de palabras para inspeccionarlo:

```python
print(tokenizer.word_index)  
```

Es demasiado grande para mostrar todo el índice, pero aquí están las 20 palabras principales. Nota que el tokenizer las lista en orden de frecuencia en el conjunto de datos, por lo que palabras comunes como "the," "and," y "a" están indexadas:

```python
{'the': 1, 'and': 2, 'a': 3, 'of': 4, 'to': 5, 'is': 6, 'br': 7, 'in': 8,  
'it': 9, 'i': 10, 'this': 11, 'that': 12, 'was': 13, 'as': 14, 'for': 15,  
'with': 16, 'movie': 17, 'but': 18, 'film': 19, "'s": 20, ...}  
```

Estas son stopwords, como se describió en la sección anterior. Tenerlas presentes puede afectar la precisión de tu entrenamiento porque son las palabras más comunes y no son distintivas.

También nota que “br” está incluido en esta lista porque es comúnmente usado en este corpus como la etiqueta HTML < br >.

Puedes actualizar el código para usar BeautifulSoup para eliminar las etiquetas HTML, agregar una traducción de cadenas para eliminar la puntuación y eliminar las stopwords de la lista proporcionada como sigue:

```python
from bs4 import BeautifulSoup  
import string  
stopwords = ["a", ... , "yourselves"]  
table = str.maketrans('', '', string.punctuation)  
imdb_sentences = []  
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))  
for item in train_data:  
    sentence = str(item['text'].decode('UTF-8').lower())  
    soup = BeautifulSoup(sentence)  
    sentence = soup.get_text()  
    words = sentence.split()  
    filtered_sentence = ""  
    for word in words:  
        word = word.translate(table)  
        if word not in stopwords:  
            filtered_sentence = filtered_sentence + word + " "  
    imdb_sentences.append(filtered_sentence)  

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)  
tokenizer.fit_on_texts(imdb_sentences)  
sequences = tokenizer.texts_to_sequences(imdb_sentences)  
print(tokenizer.word_index)  
```

Nota que las oraciones se convierten a minúsculas antes del procesamiento porque todas las stopwords se almacenan en minúsculas. Cuando imprimes tu índice de palabras ahora, verás esto:

```python
{'movie': 1, 'film': 2, 'not': 3, 'one': 4, 'like': 5, 'just': 6, 'good': 7,  
'even': 8, 'no': 9, 'time': 10, 'really': 11, 'story': 12, 'see': 13,  
'can': 14, 'much': 15, ...}  
```

Puedes ver que esto es mucho más limpio que antes. Siempre hay espacio para mejorar, sin embargo, y algo que noté al observar el índice completo fue que algunas de las palabras menos comunes hacia el final no tenían sentido. A menudo, los revisores combinaban palabras, por ejemplo, con un guion (“annoying-conclusion”) o una barra (“him/her”), y al eliminar la puntuación, estas se convertían incorrectamente en una sola palabra.

Puedes evitar esto con un poco de código que agregue espacios alrededor de estos caracteres, por lo que añadí lo siguiente inmediatamente después de que la oración fue creada:

```python
sentence = sentence.replace(",", " , ")  
sentence = sentence.replace(".", " . ")  
sentence = sentence.replace("-", " - ")  
sentence = sentence.replace("/", " / ")  
```

Esto convierte palabras combinadas como “him/her” en “him / her”, que luego elimina el “/” y las tokeniza en dos palabras. Esto podría llevar a mejores resultados de entrenamiento más adelante.

Ahora que tienes un tokenizer para el corpus, puedes codificar tus oraciones. Por ejemplo, las oraciones simples que estábamos viendo anteriormente en el capítulo se verán así:

```python
sentences = [  
    'Today is a sunny day',  
    'Today is a rainy day',  
    'Is it sunny today?'  
]  
sequences = tokenizer.texts_to_sequences(sentences)  
print(sequences)  
[[516, 5229, 147], [516, 6489, 147], [5229, 516]]  
```

Si decodificas estas, verás que las stopwords son eliminadas y obtienes las oraciones codificadas como “today sunny day,” “today rainy day,” y “sunny today.”

Si quieres hacer esto en código, puedes crear un nuevo diccionario con las claves y valores invertidos (es decir, para un par clave/valor en el índice de palabras, hacer que el valor sea la clave y la clave el valor) y hacer la búsqueda desde ahí. Aquí está el código:

```python
reverse_word_index = dict(  
    [(value, key) for (key, value) in tokenizer.word_index.items()])  
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])  
print(decoded_review)  
```

Esto dará el siguiente resultado:

```python
today sunny day  
```

#### Usando los datasets de subpalabras de IMDb
TFDS también contiene un par de datasets preprocesados de IMDb que utilizan subpalabras. Aquí no es necesario dividir las oraciones por palabras, ya que estas ya están divididas en subpalabras. Usar subpalabras es un punto intermedio entre dividir el corpus en letras individuales (pocos tokens con bajo significado semántico) y palabras individuales (muchos tokens con alto significado semántico). Este enfoque puede ser muy efectivo para entrenar un clasificador de lenguaje. Estos datasets también incluyen los codificadores y decodificadores utilizados para dividir y codificar el corpus.

Para acceder a ellos, puedes llamar a tfds.load y pasarle imdb_reviews/subwords8k o imdb_reviews/subwords32k de esta manera:

```python
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)
```

Puedes acceder al codificador en el objeto info así. Esto te ayudará a ver el tamaño del vocabulario:

```python
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))
```

Esto imprimirá 8185, ya que el vocabulario en este caso está compuesto por 8,185 tokens. Si deseas ver la lista de subpalabras, puedes obtenerla con la propiedad encoder.subwords:

```python
print(encoder.subwords)
['the_', ', ', '. ', 'a_', 'and_', 'of_', 'to_', 's_', 'is_', 'br', 'in_', 'I_', 'that_',...]
```

Algunas cosas que podrías notar aquí son que las palabras vacías, puntuación y gramática están en el corpus, así como etiquetas HTML como < br >. Los espacios se representan con guiones bajos, por lo que el primer token es la palabra "the".

Si deseas codificar una cadena, puedes hacerlo con el codificador de esta forma:

```python
sample_string = 'Today is a sunny day'
encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))
```

El resultado será una lista de tokens:

```python
Encoded string is [6427, 4869, 9, 4, 2365, 1361, 606]
```

Tus cinco palabras se codifican en siete tokens. Para ver los tokens, puedes usar la propiedad subwords en el codificador, que devuelve un arreglo. Está basado en índices cero, así que "Tod" en "Today" fue codificado como 6427, siendo el ítem 6,426 en el arreglo:

```python
print(encoder.subwords[6426])
Tod
```

Si deseas decodificar, puedes usar el método decode del codificador:

```python
encoded_string = encoder.encode(sample_string)
original_string = encoder.decode(encoded_string)
test_string = encoder.decode([6427, 4869, 9, 4, 2365, 1361, 606])
```

Las últimas líneas tendrán un resultado idéntico porque encoded_string, a pesar de su nombre, es una lista de tokens igual que la siguiente línea codificada manualmente.

### Obtener texto desde archivos CSV
Aunque TFDS tiene muchos datasets excelentes, no lo tiene todo, y a menudo tendrás que gestionar la carga de datos tú mismo. Uno de los formatos más comunes en los que se encuentra la información para procesamiento de lenguaje natural (NLP) son los archivos CSV.

En los próximos capítulos, usarás un CSV con datos de Twitter que adapté del dataset de código abierto Sentiment Analysis in Text. Usarás dos datasets diferentes: uno donde las emociones se han reducido a "positivo" o "negativo" para clasificación binaria, y otro donde se utiliza el rango completo de etiquetas emocionales. La estructura de ambos es idéntica, así que solo mostraré la versión binaria aquí.

La biblioteca csv de Python facilita el manejo de archivos CSV. En este caso, los datos se almacenan con dos valores por línea: el primero es un número (0 o 1) que indica si el sentimiento es negativo o positivo, y el segundo es una cadena con el texto.

El siguiente código leerá el archivo CSV y hará un preprocesamiento similar al que vimos en la sección anterior. Agrega espacios alrededor de la puntuación en palabras compuestas, usa BeautifulSoup para eliminar contenido HTML y luego elimina todos los caracteres de puntuación:

```python
import csv
sentences = []
labels = []
with open('/tmp/binary-emotion.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(int(row[0]))
        sentence = row[1].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
```

Esto generará una lista con 35,327 oraciones.

#### Crear subconjuntos de entrenamiento y prueba
Ahora que el corpus de texto se ha leído en una lista de oraciones, deberás dividirlo en subconjuntos de entrenamiento y prueba para entrenar un modelo. Por ejemplo, si deseas usar 28,000 oraciones para entrenamiento y el resto para pruebas, puedes usar este código:

```python
training_size = 28000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

Ahora que tienes un conjunto de entrenamiento, necesitas crear el índice de palabras a partir de él. Aquí está el código para usar el tokenizer y crear un vocabulario con un máximo de 20,000 palabras. Estableceremos la longitud máxima de una oración en 10 palabras, truncaremos las más largas al final, rellenaremos las más cortas al final y usaremos "< OOV >":

```python
vocab_size = 20000
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                 padding=padding_type, truncating=trunc_type)
```

Puedes inspeccionar los resultados observando training_sequences y training_padded. Por ejemplo, aquí imprimimos el primer ítem en la secuencia de entrenamiento, y puedes ver cómo está rellenado hasta una longitud máxima de 10:

```python
print(training_sequences[0])
print(training_padded[0])
```

Salida:

```python
[18, 3257, 47, 4770, 613, 508, 951, 423]
[  18 3257   47 4770  613  508  951  423    0    0]
```

También puedes inspeccionar el índice de palabras imprimiéndolo:

```python
print(word_index)
{'<OOV>': 1, 'just': 2, 'not': 3, 'now': 4, 'day': 5, 'get': 6, 'no': 7, 
 'good': 8, 'like': 9, 'go': 10, 'dont': 11, ...}
```

Aquí encontrarás muchas palabras que tal vez quieras considerar eliminar como palabras vacías, como "like" y "dont". Siempre es útil inspeccionar el índice de palabras.

### Obteniendo texto de archivos JSON
Otro formato muy común para archivos de texto es JavaScript Object Notation (JSON).
Este es un formato de archivo estándar abierto utilizado con frecuencia para el intercambio de datos, particularmente en aplicaciones web. Es legible por humanos y está diseñado para usar pares de nombre/valor. Como tal, es especialmente adecuado para texto etiquetado. Una búsqueda rápida de conjuntos de datos en JSON en Kaggle arroja más de 2,500 resultados. Conjuntos de datos populares como el Stanford Question Answering Dataset (SQuAD), por ejemplo, están almacenados en JSON.

JSON tiene una sintaxis muy simple, donde los objetos están contenidos entre llaves como pares de nombre/valor separados por comas. Por ejemplo, un objeto JSON que representa mi nombre sería: 

```python
{"firstName" : "Laurence",  
 "lastName" : "Moroney"}  
```

JSON también soporta arrays, que son muy similares a las listas de Python, y están denotados por la sintaxis de corchetes. Aquí tienes un ejemplo:

```python
[
 {"firstName" : "Laurence",  
 "lastName" : "Moroney"},  
 {"firstName" : "Sharon",  
 "lastName" : "Agathon"}  
]
```

Los objetos también pueden contener arrays, así que esto es un JSON perfectamente válido:

```python
[
 {"firstName" : "Laurence",  
 "lastName" : "Moroney",  
 "emails": ["lmoroney@gmail.com", "lmoroney@galactica.net"]},  
 {"firstName" : "Sharon",  
 "lastName" : "Agathon",  
 "emails": ["sharon@galactica.net", "boomer@cylon.org"]}  
]
```

Un conjunto de datos más pequeño almacenado en JSON y muy divertido para trabajar es el News Headlines Dataset for Sarcasm Detection de Rishabh Misra, disponible en Kaggle. Este conjunto de datos recopila titulares de noticias de dos fuentes: The Onion para titulares divertidos o sarcásticos, y HuffPost para titulares normales.

La estructura del archivo en el conjunto de datos de sarcasmo es muy simple:

```python
{"is_sarcastic": 1 or 0,  
 "headline": String containing headline,  
 "article_link": String Containing link}  
```

El conjunto de datos consta de aproximadamente 26,000 elementos, uno por línea. Para hacerlo más legible en Python, he creado una versión que encierra estos elementos en un array para que pueda leerse como una lista única, que se usa en el código fuente de este capítulo.

#### Leyendo archivos JSON
La biblioteca json de Python hace que leer archivos JSON sea simple. Dado que JSON utiliza pares de nombre/valor, puedes indexar el contenido basado en el nombre. Por ejemplo, para el conjunto de datos de sarcasmo puedes crear un manejador de archivo para el archivo JSON, abrirlo con la biblioteca json, iterarlo, leer cada campo línea por línea y obtener el elemento de datos utilizando el nombre del campo.

Aquí está el código:

```python
import json  
with open("/tmp/sarcasm.json", 'r') as f:  
    datastore = json.load(f)  
    for item in datastore:  
        sentence = item['headline'].lower()  
        label = item['is_sarcastic']  
        link = item['article_link']  
```

Esto hace que sea sencillo crear listas de oraciones y etiquetas como has hecho a lo largo de este capítulo, y luego tokenizar las oraciones. También puedes hacer preprocesamiento en tiempo real mientras lees una oración, eliminando stopwords, etiquetas HTML, puntuación y más.

Aquí tienes el código completo para crear listas de oraciones, etiquetas y URLs, mientras las oraciones se limpian de palabras y caracteres no deseados:

```python
with open("/tmp/sarcasm.json", 'r') as f:  
    datastore = json.load(f)  
    sentences = []  
    labels = []  
    urls = []  
    for item in datastore:  
        sentence = item['headline'].lower()  
        sentence = sentence.replace(",", " , ")  
        sentence = sentence.replace(".", " . ")  
        sentence = sentence.replace("-", " - ")  
        sentence = sentence.replace("/", " / ")  
        soup = BeautifulSoup(sentence)  
        sentence = soup.get_text()  
        words = sentence.split()  
        filtered_sentence = ""  
        for word in words:  
            word = word.translate(table)  
            if word not in stopwords:  
                filtered_sentence = filtered_sentence + word + " "  
        sentences.append(filtered_sentence)  
        labels.append(item['is_sarcastic'])  
        urls.append(item['article_link'])  
```

Como antes, estos pueden dividirse en conjuntos de entrenamiento y prueba. Si deseas usar 23,000 de los 26,000 elementos en el conjunto de datos para el entrenamiento, puedes hacer lo siguiente:

```python
training_size = 23000  
training_sentences = sentences[0:training_size]  
testing_sentences = sentences[training_size:]  
training_labels = labels[0:training_size]  
testing_labels = labels[training_size:]  
```
Para tokenizar los datos y prepararlos para el entrenamiento, puedes seguir el mismo enfoque que antes. Aquí, especificamos nuevamente un tamaño de vocabulario de 20,000 palabras, una longitud máxima de secuencia de 10 con truncamiento y padding al final, y un token OOV de “< OOV >”:

```python
vocab_size = 20000  
max_length = 10  
trunc_type = 'post'  
padding_type = 'post'  
oov_tok = "<OOV>"  
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  
tokenizer.fit_on_texts(training_sentences)  
word_index = tokenizer.word_index  
training_sequences = tokenizer.texts_to_sequences(training_sentences)  
padded = pad_sequences(training_sequences, padding='post')  
print(word_index)  
```

La salida será todo el índice, en orden de frecuencia de palabras:

```python
{'<OOV>': 1, 'new': 2, 'trump': 3, 'man': 4, 'not': 5, 'just': 6, 'will': 7,  
'one': 8, 'year': 9, 'report': 10, 'area': 11, 'donald': 12, ...}  
```

Con suerte, el código de apariencia similar te ayudará a identificar el patrón que puedes seguir al preparar texto para que las redes neuronales lo clasifiquen o generen. En el próximo capítulo verás cómo construir un clasificador de texto usando embeddings, y en el Capítulo 7 llevarás eso un paso más allá, explorando redes neuronales recurrentes. Luego, en el Capítulo 8, verás cómo mejorar aún más los datos de secuencia para crear una red neuronal capaz de generar nuevo texto.

## Resumen
En capítulos anteriores usaste imágenes para construir un clasificador. Las imágenes, por definición, son altamente estructuradas. Sabes sus dimensiones. Sabes su formato. El texto, en cambio, puede ser mucho más difícil de trabajar. A menudo no está estructurado, puede contener contenido no deseado como instrucciones de formato, no siempre contiene lo que necesitas, y frecuentemente debe ser filtrado para eliminar contenido irrelevante o sin sentido.

En este capítulo viste cómo tomar texto y convertirlo en números usando tokenización de palabras, y luego exploraste cómo leer y filtrar texto en una variedad de formatos.

Con estas habilidades, ahora estás listo para dar el siguiente paso y aprender cómo se puede inferir significado a partir de las palabras: el primer paso para entender el lenguaje natural.