# Usando TensorFlow para Crear Texto

```python
Sabes nada, Jon Snow
el lugar donde está destinado
ya sea Cork o en el hijo del pájaro azul
navegó hacia el verano
vieja dulce larga y los anillos de alegría
así que esperaré a la salvaje colleen muriendo
```

Este texto fue generado por un modelo muy simple entrenado en un pequeño corpus.
Lo he mejorado un poco añadiendo saltos de línea y puntuación, pero aparte de la primera línea, el resto fue generado completamente por el modelo que aprenderás a construir en este capítulo.
Es algo genial que mencione a una salvaje colleen muriendo—¡si has visto el programa del que proviene Jon Snow, entenderás por qué!

En los últimos capítulos, viste cómo puedes usar TensorFlow con datos basados en texto, primero tokenizándolos en números y secuencias que pueden ser procesados por una red neuronal, luego usando embeddings para simular sentimientos mediante vectores, y finalmente utilizando redes neuronales profundas y recurrentes para clasificar texto. Usamos el conjunto de datos Sarcasm, uno pequeño y simple, para ilustrar cómo funciona todo esto. En este capítulo, cambiaremos de enfoque: en lugar de clasificar texto existente, crearás una red neuronal que pueda predecir texto.

Dado un corpus de texto, intentará entender los patrones de palabras dentro de él, de modo que pueda, dado un nuevo fragmento de texto llamado semilla, predecir qué palabra debería venir después. Una vez que tenga eso, la semilla y la palabra predicha se convierten en la nueva semilla, y se puede predecir la siguiente palabra. Así, cuando se entrena con un corpus de texto, una red neuronal puede intentar escribir nuevo texto en un estilo similar.

Para crear el poema anterior, recopilé letras de varias canciones tradicionales irlandesas, entrené una red neuronal con ellas y la usé para predecir palabras. Comenzaremos con algo simple, usando una pequeña cantidad de texto para ilustrar cómo construir un modelo predictivo, y terminaremos creando un modelo completo con mucho más texto. Después de eso, ¡podrás probarlo para ver qué tipo de poesía puede crear!

Para comenzar, tendrás que tratar el texto de manera un poco diferente de lo que has hecho hasta ahora. En los capítulos anteriores, tomaste oraciones y las convertiste en secuencias que luego fueron clasificadas según los embeddings de los tokens dentro de ellas.

Cuando se trata de crear datos que puedan usarse para entrenar un modelo predictivo como este, hay un paso adicional donde las secuencias deben transformarse en secuencias de entrada y etiquetas, donde la secuencia de entrada es un grupo de palabras y la etiqueta es la siguiente palabra en la oración. Luego puedes entrenar un modelo para hacer coincidir las secuencias de entrada con sus etiquetas, de modo que las predicciones futuras puedan seleccionar una etiqueta que se acerque a la secuencia de entrada.

## Convirtiendo Secuencias en Secuencias de Entrada
Cuando predices texto, necesitas entrenar una red neuronal con una secuencia de entrada (característica) que tenga una etiqueta asociada. Emparejar secuencias con etiquetas es clave para predecir texto.

Por ejemplo, si en tu corpus tienes la frase "Hoy hay un hermoso cielo azul", podrías dividirla en "Hoy hay un hermoso azul" como la característica y "cielo" como la etiqueta. Entonces, si obtuvieras una predicción para el texto "Hoy hay un hermoso azul", probablemente sería "cielo". Si en los datos de entrenamiento también tienes "Ayer había un hermoso cielo azul", dividido de la misma manera, y obtuvieras una predicción para el texto "Mañana habrá un hermoso azul", hay una alta probabilidad de que la próxima palabra sea "cielo".

Dadas muchas frases, entrenando con secuencias de palabras donde la palabra siguiente es la etiqueta, puedes construir rápidamente un modelo predictivo donde se pueda predecir la palabra más probable siguiente en la oración a partir de un texto existente.

Comenzaremos con un corpus de texto muy pequeño: un fragmento de una canción tradicional irlandesa de la década de 1860, cuyas letras son las siguientes:

```python
In the town of Athy one Jeremy Lanigan
Battered away til he hadnt a pound.
His father died and made him a man again
Left him a farm and ten acres of ground.

He gave a grand party for friends and relations
Who didnt forget him when come to the wall,
And if youll but listen Ill make your eyes glisten
Of the rows and the ructions of Lanigan’s Ball.

Myself to be sure got free invitation,
For all the nice girls and boys I might ask,
And just in a minute both friends and relations
Were dancing round merry as bees round a cask.

Judy ODaly, that nice little milliner,
She tipped me a wink for to give her a call,
And I soon arrived with Peggy McGilligan
Just in time for Lanigans Ball.
```

Crea un único string con todo el texto y configúralo como tus datos. Usa \n para los saltos de línea. Luego este corpus puede ser fácilmente cargado y tokenizado de esta forma:

```python
tokenizer = Tokenizer()
data = "In the town of Athy one Jeremy Lanigan \n Battered away ... ..."
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
```

El resultado de este proceso es reemplazar las palabras por sus valores de token, como se muestra en la Figura 8-1.

![Figura 8-1. Tokenizando una oración](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.1.png)

Para entrenar un modelo predictivo, debemos dar un paso adicional aquí: dividir la oración en múltiples secuencias más pequeñas, de modo que, por ejemplo, podamos tener una secuencia que consista en los primeros dos tokens, otra de los primeros tres, etc. (Figura 8-2).

![Figura 8-2. Convirtiendo una secuencia en un número de secuencias de entrada](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.2.png)

Para hacerlo, necesitas recorrer cada línea del corpus y convertirla en una lista de tokens usando texts_to_sequences. Luego puedes dividir cada lista iterando sobre cada token y haciendo una lista de todos los tokens hasta ese punto.

Aquí está el código:

```python
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
print(input_sequences[:5])
```

Una vez que tengas estas secuencias de entrada, puedes rellenarlas para que tengan una forma regular. Usaremos pre-padding (Figura 8-3).

![Figura 8-3. Rellenando las secuencias de entrada](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.3.png)

Para hacerlo, necesitas encontrar la oración más larga en las secuencias de entrada y rellenar todo al tamaño de esa longitud. Aquí está el código:

```python
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
```

Finalmente, una vez que tengas un conjunto de secuencias de entrada rellenadas, puedes dividirlas en características y etiquetas, donde la etiqueta es simplemente el último token en la secuencia de entrada (Figura 8-4).

![Figura 8-4. Convirtiendo las secuencias rellenadas en características (x) y etiquetas (y)](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.4.png)

Al entrenar una red neuronal, vas a emparejar cada característica con su etiqueta correspondiente. Por ejemplo, la etiqueta para [0 0 0 0 4 2 66 8 67 68 69] será [70].

Aquí está el código para separar las etiquetas de las secuencias de entrada:

```python
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
```

A continuación, necesitas codificar las etiquetas. Actualmente son solo tokens—por ejemplo, el número 2 en la parte superior de la Figura 8-4. Pero si quieres usar un token como etiqueta en un clasificador, tendrá que estar mapeado a una neurona de salida. Por lo tanto, si vas a clasificar n palabras, con cada palabra siendo una clase, necesitarás tener n neuronas.

Aquí es donde es importante controlar el tamaño del vocabulario, porque cuantas más palabras tengas, más clases necesitarás. Recuerda en los capítulos 2 y 3 cuando clasificaste elementos de moda con el conjunto de datos Fashion MNIST, y tenías 10 tipos de prendas de vestir. Eso requería tener 10 neuronas en la capa de salida. En este caso, ¿qué pasa si quieres predecir hasta 10,000 palabras del vocabulario? Necesitarás una capa de salida con 10,000 neuronas.

Además, necesitas codificar tus etiquetas con one-hot encoding para que coincidan con la salida deseada de una red neuronal. Considera la Figura 8-4. Si a una red neuronal se le alimenta la entrada X que consiste en una serie de ceros seguida por un 4, querrás que la predicción sea 2. Sin embargo, la forma en que la red entrega esto es teniendo una capa de salida con vocabulary_size neuronas, donde la segunda tiene la probabilidad más alta.

Para codificar tus etiquetas en un conjunto de Ys que luego puedas usar para entrenar, puedes usar la utilidad to_categorical en tf.keras:

```python
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

Puedes verlo visualmente en la Figura 8-5.

![Figura 8-5. Codificación one-hot de etiquetas](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.5.png)

Esta es una representación muy dispersa, que, si tienes muchos datos de entrenamiento y muchas palabras potenciales, consumirá memoria rápidamente. ¡Supón que tienes 100,000 oraciones de entrenamiento con un vocabulario de 10,000 palabras! Necesitarías 1,000,000,000 bytes solo para almacenar las etiquetas. Pero es la forma en que debemos diseñar nuestra red si vamos a clasificar y predecir palabras.

## Creando el Modelo
Ahora vamos a crear un modelo simple que pueda ser entrenado con estos datos de entrada. Consistirá en una capa de embedding, seguida de una LSTM, y luego una capa dense. Para el embedding, necesitarás un vector por cada palabra, por lo que los parámetros serán el número total de palabras y el número de dimensiones en las que deseas realizar el embedding. En este caso, no tenemos muchas palabras, así que ocho dimensiones deberían ser suficientes.

Puedes hacer que la LSTM sea bidireccional, y el número de pasos puede ser la longitud de una secuencia, que es nuestra longitud máxima menos 1 (porque quitamos un token al final para hacer la etiqueta).

Finalmente, la capa de salida será una capa dense con el número total de palabras como parámetro, activada con softmax. Cada neurona en esta capa representará la probabilidad de que la siguiente palabra coincida con la palabra del valor de índice correspondiente:

```python
model = Sequential()
model.add(Embedding(total_words, 8))
model.add(Bidirectional(LSTM(max_sequence_len-1)))
model.add(Dense(total_words, activation='softmax'))
```

Compila el modelo con una función de pérdida categórica como categorical cross entropy y un optimizador como Adam. También puedes especificar que deseas capturar métricas:

```python
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
```

Es un modelo muy simple y con pocos datos, por lo que puedes entrenarlo durante un largo tiempo, por ejemplo, 1,500 épocas:

```python
history = model.fit(xs, ys, epochs=1500, verbose=1)
```

Después de 1,500 épocas, verás que alcanza una precisión muy alta (Figura 8-6).

```python
Figura 8-6. Precisión del entrenamiento
```

Con el modelo alcanzando alrededor del 95% de precisión, podemos estar seguros de que si tenemos un fragmento de texto que ya ha visto, predecirá la siguiente palabra correctamente alrededor del 95% de las veces. Sin embargo, ten en cuenta que al generar texto, verá continuamente palabras que no ha visto previamente, por lo que, a pesar de este buen resultado, encontrarás que la red rápidamente terminará produciendo texto sin sentido. Exploraremos esto en la siguiente sección.

## Generando Texto
Ahora que has entrenado una red que puede predecir la siguiente palabra en una secuencia, el siguiente paso es darle una secuencia de texto y hacer que prediga la próxima palabra. Veamos cómo hacerlo.

### Prediciendo la Siguiente Palabra
Comenzarás creando una frase llamada seed text. Esta es la expresión inicial sobre la cual la red basará todo el contenido que genere. Lo hará prediciendo la siguiente palabra.

Empieza con una frase que la red ya haya visto, como "in the town of athy":

```python
seed_text = "in the town of athy"
```

A continuación, necesitas tokenizarla usando texts_to_sequences. Esto devuelve un array, incluso si solo hay un valor, por lo que toma el primer elemento de ese array:

```python
token_list = tokenizer.texts_to_sequences([seed_text])[0]
```

Luego necesitas rellenar esa secuencia para darle la misma forma que los datos usados en el entrenamiento:

```python
token_list = pad_sequences([token_list], 
                           maxlen=max_sequence_len-1, padding='pre')
```

Ahora puedes predecir la siguiente palabra para esta lista de tokens llamando a model.predict en la lista de tokens. Esto devolverá las probabilidades para cada palabra en el corpus, así que pasa los resultados a np.argmax para obtener la más probable:

```python
predicted = np.argmax(model.predict(token_list), axis=-1)
print(predicted)
```

Esto debería darte el valor 68. Si miras el índice de palabras, verás que esta es la palabra "one":

```python
'town': 66, 'athy': 67, 'one': 68, 'jeremy': 69, 'lanigan': 70,
```

Puedes buscarlo en el código recorriendo los elementos del índice de palabras hasta encontrar el valor predicho e imprimirlo:

```python
for word, index in tokenizer.word_index.items():
    if index == predicted:
        print(word)
        break
```

Entonces, comenzando con el texto "in the town of athy", la red predijo que la siguiente palabra debería ser "one", lo cual, si observas los datos de entrenamiento, es correcto, porque la canción comienza con la línea:

```python
In the town of Athy one Jeremy Lanigan  
Battered away til he hadnt a pound  
```

Ahora que has confirmado que el modelo funciona, puedes ser creativo y usar diferentes textos iniciales. Por ejemplo, cuando usé el texto inicial "sweet jeremy saw dublin", la siguiente palabra que predijo fue "then". (Este texto fue elegido porque todas esas palabras están en el corpus. Deberías esperar resultados más precisos, al menos al principio, para las palabras predichas en tales casos.)

### Combinando Predicciones para Generar Texto
En la sección anterior viste cómo usar el modelo para predecir la siguiente palabra dada un texto inicial (seed text). Para que la red neuronal cree un nuevo texto, simplemente repite la predicción, agregando nuevas palabras cada vez.

Por ejemplo, anteriormente, cuando utilicé la frase "sweet jeremy saw dublin", el modelo predijo que la siguiente palabra sería "then". Puedes construir sobre esto agregando "then" al texto inicial para obtener "sweet jeremy saw dublin then" y obtener otra predicción. Repetir este proceso generará una cadena de texto creada por la IA.

Aquí está el código actualizado de la sección anterior que realiza este bucle varias veces, con el número determinado por el parámetro next_words:

```python
seed_text = "sweet jeremy saw dublin"
next_words = 10
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], 
                               maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    print(seed_text)
```

Esto terminará creando una cadena algo como esta:

```python
sweet jeremy saw dublin then got there as me me a call doing me
```

Rápidamente desciende en incoherencias. ¿Por qué?

- El cuerpo del texto de entrenamiento es muy pequeño. Tiene muy poco contexto con el que trabajar.
- La predicción depende de las palabras previas. Si hay una coincidencia pobre en las palabras anteriores, incluso la mejor opción para la siguiente palabra tendrá una baja probabilidad. Cuando agregas esto a la secuencia y predices la siguiente palabra, la probabilidad de tener una baja coincidencia es aún mayor, lo que hace que las palabras predichas parezcan casi aleatorias.

Por ejemplo, aunque todas las palabras en la frase "sweet jeremy saw dublin" existen en el corpus, nunca aparecen en ese orden. Cuando se realizó la primera predicción, la palabra "then" fue elegida como la candidata más probable, con una probabilidad bastante alta (89%). Al agregarla al texto inicial para formar "sweet jeremy saw dublin then", obtuvimos otra frase que no estaba en los datos de entrenamiento, por lo que la predicción dio la probabilidad más alta a la palabra "got", con un 44%. Continuar agregando palabras a la oración reduce aún más la probabilidad de una coincidencia en los datos de entrenamiento, y como resultado, la precisión de las predicciones sufre, llevando a un texto que se siente más "aleatorio".

Esto lleva al fenómeno de que el contenido generado por IA se vuelve cada vez más incoherente con el tiempo.

Para un ejemplo, echa un vistazo al excelente cortometraje de ciencia ficción Sunspring, que fue escrito completamente por una red basada en LSTM como la que estás construyendo aquí, entrenada en guiones de películas de ciencia ficción. Se le dio contenido inicial y se le encargó generar un nuevo guion. Los resultados fueron hilarantes, y verás que, aunque el contenido inicial tiene sentido, a medida que avanza la película, se vuelve cada vez menos comprensible.

## Extendiendo el Dataset
El mismo patrón que usaste para el conjunto de datos codificado directamente puede extenderse para usar un archivo de texto de manera muy simple. He alojado un archivo de texto que contiene alrededor de 1,700 líneas de texto recopiladas de varias canciones, que puedes usar para experimentar. Con una pequeña modificación, puedes usar esto en lugar de la única canción codificada directamente.

Para descargar los datos en Colab, utiliza el siguiente código:

```python
!wget --no-check-certificate \
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/ \
irish-lyrics-eof.txt -O /tmp/irish-lyrics-eof.txt
```

Luego simplemente puedes cargar el texto en tu corpus de esta manera:

```python
data = open('/tmp/irish-lyrics-eof.txt').read()
corpus = data.lower().split("\n")
```

¡El resto de tu código funcionará sin modificaciones! Entrenar esto durante mil épocas te lleva a aproximadamente un 60% de precisión, con la curva aplanándose (Figura 8-7).

![Figura 8-7. Entrenamiento con un conjunto de datos más grande](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.7.png)

Probar nuevamente la frase "in the town of athy" produce una predicción de "one", pero esta vez con solo un 40% de probabilidad.

Para "sweet jeremy saw dublin", la siguiente palabra predicha es "drawn", con una probabilidad del 59%. Predecir las siguientes 10 palabras produce:

```python
sweet jeremy saw dublin drawn and fondly i am dead and the parting graceful
```

¡Se ve un poco mejor! ¿Pero podemos mejorarlo aún más?

## Cambiar la Arquitectura del Modelo
Una forma de mejorar el modelo es cambiar su arquitectura, utilizando múltiples LSTMs apiladas. Esto es bastante sencillo: solo asegúrate de configurar return_sequences en True en la primera de ellas. Aquí está el código:

```python
model = Sequential()
model.add(Embedding(total_words, 8))
model.add(Bidirectional(LSTM(max_sequence_len-1, return_sequences=True)))
model.add(Bidirectional(LSTM(max_sequence_len-1)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=1000, verbose=1)
```

Puedes ver el impacto que esto tiene en el entrenamiento durante mil épocas en la Figura 8-8. No es significativamente diferente de la curva anterior.

![Figura 8-8. Agregar una segunda capa LSTM](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.8.png)

Al probar con las mismas frases de antes, esta vez obtuve "more" como la siguiente palabra después de "in the town of athy" con un 51% de probabilidad, y después de "sweet jeremy saw dublin" obtuve "cailín" (la palabra gaélica para “chica”) con un 61% de probabilidad. Nuevamente, al predecir más palabras, la salida rápidamente descendió en incoherencias.

Aquí tienes algunos ejemplos:

```python
sweet jeremy saw dublin cailín loo ra fountain plundering that fulfill
you mccarthy you mccarthy down
you know nothing jon snow johnny cease and she danced that put to smother well
i must the wind flowers
dreams it love to laid ned the mossy and night i weirs
```

Si obtienes resultados diferentes, no te preocupes—no hiciste nada mal. La inicialización aleatoria de las neuronas impactará los resultados finales.

## Mejorando los datos
Hay un pequeño truco que puedes usar para ampliar el tamaño de este conjunto de datos sin añadir nuevas canciones, llamado ventana de los datos. En este momento, cada línea de cada canción se lee como una sola línea y luego se convierte en secuencias de entrada, como viste en la Figura 8-2. Mientras que los humanos leen las canciones línea por línea para escuchar la rima y el ritmo, el modelo no tiene que hacerlo, especialmente cuando se usan LSTMs bidireccionales.

Entonces, en lugar de tomar la línea "In the town of Athy, one Jeremy Lanigan", procesarla y luego pasar a la siguiente línea ("Battered away till he hadn’t a pound") y procesarla, podríamos tratar todas las líneas como un solo texto largo y continuo. Luego podemos crear una "ventana" en ese texto de n palabras, procesarla, y luego mover la ventana una palabra hacia adelante para obtener la siguiente secuencia de entrada (Figura 8-9).

![Figura 8-9. Una ventana móvil de palabras](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.9.png)

En este caso, se puede generar muchos más datos de entrenamiento en forma de un mayor número de secuencias de entrada. Mover la ventana a través de todo el corpus de texto nos daría ((número_de_palabras - tamaño_ventana) × tamaño_ventana) secuencias de entrada con las que podríamos entrenar.

El código es bastante simple: al cargar los datos, en lugar de dividir cada línea de la canción en una "oración", podemos crearlas sobre la marcha a partir de las palabras en el corpus:

```python
window_size=10  
sentences=[]  
alltext=[]  
data = open('/tmp/irish-lyrics-eof.txt').read()  
corpus = data.lower()  
words = corpus.split(" ")  
range_size = len(words)-max_sequence_len  
for i in range(0, range_size):  
    thissentence=""  
    for word in range(0, window_size-1):  
        word = words[i+word]  
        thissentence = thissentence + word  
        thissentence = thissentence + " "  
    sentences.append(thissentence)
```

En este caso, como ya no tenemos oraciones y estamos creando secuencias del mismo tamaño que la ventana móvil, max_sequence_len es el tamaño de la ventana. Se lee el archivo completo, se convierte a minúsculas y se divide en un array de palabras usando la división de cadenas. Luego, el código recorre las palabras y crea oraciones con cada palabra desde el índice actual hasta el índice actual más el tamaño de la ventana, agregando cada una de esas oraciones recién construidas al array de oraciones.

Cuando entrenes, notarás que los datos adicionales hacen que cada época sea mucho más lenta, pero los resultados mejoran considerablemente, y el texto generado se convierte en galimatías mucho más lentamente.

Aquí hay un ejemplo que me llamó la atención, ¡particularmente la última línea!

```python
you know nothing, jon snow is gone  
and the young and the rose and wide  
to where my love i will play  
the heart of the kerry  
the wall i watched a neat little town
```

Hay muchos hiperparámetros que puedes intentar ajustar. Cambiar el tamaño de la ventana cambiará la cantidad de datos de entrenamiento: un tamaño de ventana más pequeño puede generar más datos, pero habrá menos palabras para asignar a una etiqueta, así que si lo estableces demasiado pequeño terminarás con poesía sin sentido. También puedes cambiar las dimensiones en el embedding, el número de LSTMs, o el tamaño del vocabulario para usar en el entrenamiento. Dado que la precisión porcentual no es la mejor medida, querrás hacer un examen más subjetivo de cuánto “sentido” tiene la poesía, no hay una regla estricta para determinar si tu modelo es “bueno” o no.

Por ejemplo, cuando intenté usar un tamaño de ventana de 6, aumentar el número de dimensiones para el embedding a 16, cambiar el número de LSTMs del tamaño de la ventana (que sería 6) a 32, y aumentar la tasa de aprendizaje en el optimizador Adam, obtuve una bonita curva de aprendizaje suave (Figura 8-10) y algo de la poesía empezó a tener más sentido.

![Figura 8-10. Curva de aprendizaje con hiperparámetros ajustados](./assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure8.10.png)

Al usar "sweet jeremy saw dublin" como semilla (recuerda, todas las palabras en la semilla están en el corpus), obtuve este poema:

```python
sweet jeremy saw dublin  
whack fol  
all the watch came  
and if ever you love get up from the stool  
longs to go as i was passing my aged father  
if you can visit new ross  
gallant words i shall make  
such powr of her goods  
and her gear  
and her calico blouse  
she began the one night  
rain from the morning so early  
oer railroad ties and crossings  
i made my weary way  
through swamps and elevations  
my tired feet  
was the good heavens
```

Mientras que la frase "whack fol" puede no tener sentido para muchos lectores, es común en algunas canciones irlandesas, algo así como "la la la" o "doobie-doobie-doo". Lo que realmente me gustó de esto es cómo algunas de las frases posteriores mantuvieron algún tipo de sentido, como "such power of her good and her gear, and her calico blouse", pero esto podría ser debido al sobreajuste a las frases que ya existen en las canciones dentro del corpus. Por ejemplo, las líneas que comienzan con "oer railroad ties..." hasta "my tired feet" son tomadas directamente de una canción llamada "The Lakes of Pontchartrain" que está en el corpus. Si te encuentras con problemas como este, lo mejor es reducir la tasa de aprendizaje y tal vez disminuir el número de LSTMs. Pero sobre todo, ¡experimenta y diviértete!

## Codificación basada en caracteres

En los últimos capítulos hemos estado viendo NLP utilizando codificación basada en palabras. Encuentro que es mucho más fácil comenzar con esto, pero cuando se trata de generar texto, también podrías considerar la codificación basada en caracteres porque el número de caracteres únicos en un corpus tiende a ser mucho menor que el número de palabras únicas. Como tal, puedes tener muchos menos neuronas en tu capa de salida, y tus predicciones de salida se distribuyen a través de menos probabilidades. Por ejemplo, al ver el conjunto de datos de las obras completas de Shakespeare, verás que solo hay 65 caracteres únicos en todo el conjunto. Entonces, cuando haces predicciones, en lugar de observar las probabilidades de la siguiente palabra entre 2,700 palabras como en el conjunto de datos de canciones irlandesas, solo estás observando 65. ¡Esto hace que tu modelo sea un poco más sencillo!

Lo que también es agradable acerca de la codificación basada en caracteres es que los caracteres de puntuación también están incluidos, por lo que saltos de línea, etc., pueden ser predichos. Como ejemplo, cuando utilicé una RNN entrenada con el corpus de Shakespeare para predecir el texto que sigue a mi línea favorita de Game of Thrones, obtuve:

```python
YGRITTE: 
You know nothing, Jon Snow. 
Good night, we’ll prove those body’s servants to
The traitor be these mine:
So diswarl his body in hope in this resceins,
I cannot judg appeal’t.

MENENIUS: 
Why, ’tis pompetsion.

KING RICHARD II:
I think he make her thought on mine;
She will not: suffer up thy bonds:
How doched it, I pray the gott,
We’ll no fame to this your love, and you were ends
```

Es algo genial que ella lo identifique como un traidor y quiera atarlo ("diswarl his body"), ¡pero no tengo idea de qué significa "resceins"! Si ves el programa, esta es parte de la trama, ¡así que tal vez Shakespeare estaba en algo sin darse cuenta!

Por supuesto, creo que tendemos a ser un poco más indulgentes cuando usamos algo como los textos de Shakespeare como nuestros datos de entrenamiento, porque el lenguaje ya es un poco desconocido.

Como con el modelo de canciones irlandesas, la salida rápidamente degenera en texto sin sentido, pero aún es divertido jugar con ello. Para intentarlo por ti mismo, puedes revisar el Colab.

## Resumen
En este capítulo exploramos cómo hacer una generación básica de texto utilizando un modelo entrenado basado en LSTM. Viste cómo puedes dividir el texto en características de entrenamiento y etiquetas, usando palabras como etiquetas, y crear un modelo que, al recibir un texto de inicio, pueda predecir la siguiente palabra probable. Iteraste sobre esto para mejorar el modelo y obtener mejores resultados, explorando un conjunto de datos de canciones tradicionales irlandesas. También viste un poco sobre cómo esto podría mejorarse potencialmente con generación de texto basada en caracteres con un ejemplo que usa texto shakesperiano. ¡Espero que haya sido una introducción divertida a cómo los modelos de aprendizaje automático pueden sintetizar texto!
