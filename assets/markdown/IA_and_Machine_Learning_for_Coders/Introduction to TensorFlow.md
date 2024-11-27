# Introducción a TensorFlow
Cuando se trata de crear inteligencia artificial (IA), el aprendizaje automático (ML) y el aprendizaje profundo son un excelente punto de partida. Sin embargo, al comenzar, es fácil sentirse abrumado por las opciones y toda la nueva terminología. Este libro tiene como objetivo desmitificar estos temas para los programadores, guiándolos a través de la escritura de código para implementar conceptos de aprendizaje automático y aprendizaje profundo; y construir modelos que se comporten más como lo haría un humano, en escenarios como visión por computadora, procesamiento de lenguaje natural (NLP) y más. De esta manera, estos modelos se convierten en una forma de inteligencia sintetizada o artificial.

Pero cuando hablamos de aprendizaje automático, ¿qué es en realidad este fenómeno? Vamos a analizarlo rápidamente y a considerarlo desde la perspectiva de un programador antes de continuar. Luego, este capítulo te mostrará cómo instalar las herramientas necesarias, desde TensorFlow hasta los entornos donde puedes escribir y depurar tus modelos de TensorFlow.

## ¿Qué es el aprendizaje automático?
Antes de adentrarnos en los detalles del aprendizaje automático (ML), veamos cómo evolucionó a partir de la programación tradicional. Comenzaremos examinando qué es la programación tradicional, luego consideraremos los casos en los que esta tiene limitaciones. A partir de allí, veremos cómo el aprendizaje automático evolucionó para abordar esos casos y, como resultado, abrió nuevas oportunidades para implementar escenarios innovadores, desbloqueando muchos de los conceptos de la inteligencia artificial.

La programación tradicional implica escribir reglas, expresadas en un lenguaje de programación, que actúan sobre datos y nos proporcionan respuestas. Esto se aplica prácticamente a cualquier situación en la que algo pueda programarse con código.

Por ejemplo, considera un juego como el popular Breakout. El código determina el movimiento de la pelota, el puntaje y las diversas condiciones para ganar o perder el juego. Piensa en el escenario donde la pelota rebota contra un ladrillo, como se muestra en la Figura 1-1.

![Figura 1-1. Código en un juego de Breakout](assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.1.png)

Aquí, el movimiento de la pelota puede determinarse mediante sus propiedades dx y dy. Cuando la pelota golpea un ladrillo, este se elimina, y la velocidad de la pelota aumenta y cambia de dirección. El código actúa sobre los datos relacionados con la situación del juego.

Alternativamente, considera un escenario de servicios financieros. Tienes datos sobre las acciones de una empresa, como su precio actual y sus ganancias actuales. Puedes calcular un valioso indicador llamado P/E (precio dividido entre ganancias) utilizando un código como el que se muestra en la Figura 1-2.

![Figura 1-2. Código en un escenario de servicios financieros](assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.2.png)

Tu código lee el precio, lee las ganancias y devuelve un valor que resulta de dividir el primero por el segundo.

Si intentara resumir la programación tradicional en un solo diagrama, se parecería a lo que se muestra en la Figura 1-3.

![Figura 1-3. Vista de alto nivel de la programación tradicional](assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.3.png)

Como puedes ver, tienes reglas expresadas en un lenguaje de programación. Estas reglas actúan sobre los datos, y el resultado son las respuestas.

## Limitaciones de la programación tradicional
El modelo de la Figura 1-3 ha sido la columna vertebral del desarrollo desde sus inicios. Sin embargo, tiene una limitación inherente: solo se pueden implementar escenarios para los cuales es posible derivar reglas. ¿Qué sucede con otros escenarios? Por lo general, son inviables de desarrollar porque el código es demasiado complejo. Simplemente, no es posible escribir código para manejar dichos casos.

Por ejemplo, considera la detección de actividades físicas. Los monitores de actividad, como los que detectan si estamos caminando, son una innovación reciente, no solo debido a la disponibilidad de hardware pequeño y económico, sino también porque los algoritmos necesarios para esa detección no eran factibles anteriormente. Exploremos por qué.

La Figura 1-4 muestra un algoritmo ingenuo para detectar si una persona está caminando. Este puede basarse en la velocidad de la persona. Si la velocidad es menor a cierto valor, podemos determinar que probablemente está caminando.

![Figura 1-4. Algoritmo para la detección de actividad (caminar)](assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.4.png)

Si tomamos como base la velocidad, podríamos extender este algoritmo para detectar si una persona está corriendo (Figura 1-5).

![Figura 1-5. Extensión del algoritmo para detectar si alguien está corriendo](assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.5.png)

Como se puede observar, usando la velocidad, podríamos decir que si es menor a un valor específico (por ejemplo, 4 mph), la persona está caminando; de lo contrario, está corriendo. Aún funciona, de cierta forma.

Extendiendo a otras actividades
Ahora, supongamos que queremos extender este algoritmo para otra actividad popular de fitness, como andar en bicicleta. El algoritmo podría lucir como en la Figura 1-6.

![Figura 1-6. Extensión del algoritmo para detectar si alguien está andando en bicicleta](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.6.png)

Reconozco que es un enfoque ingenuo, ya que solo considera la velocidad. Por ejemplo, algunas personas corren más rápido que otras, o podrías correr cuesta abajo más rápido de lo que pedaleas cuesta arriba. Pero, en general, el algoritmo aún funciona. Sin embargo, ¿qué sucede si queremos implementar otro escenario, como jugar golf (Figura 1-7)?

![Figura 1-7. ¿Cómo escribir un algoritmo para detectar golf?](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.7.png)

Aquí nos encontramos con un problema. ¿Cómo determinamos que alguien está jugando golf usando esta metodología? Una persona podría caminar un poco, detenerse, realizar alguna actividad, caminar nuevamente, detenerse, etc. ¿Cómo podemos identificar que esto es golf?

Las limitaciones de las reglas tradicionales
Nuestra capacidad para detectar esta actividad usando reglas tradicionales ha llegado a un límite. Pero tal vez haya una mejor forma de abordar este problema.

Entra en escena el aprendizaje automático

## De la Programación al Aprendizaje
Veamos nuevamente el diagrama que utilizamos para demostrar qué es la programación tradicional (Figura 1-8). Aquí tenemos reglas que actúan sobre los datos y nos dan respuestas. En nuestro escenario de detección de actividad, los datos eran la velocidad a la que la persona se movía; a partir de ello, podíamos escribir reglas para detectar su actividad, ya sea caminar, andar en bicicleta o correr. Nos encontramos con un obstáculo cuando se trataba de jugar golf, porque no pudimos formular reglas que describieran cómo sería esa actividad.

![Figura 1-8. El flujo de programación tradicional](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.8.png)

Pero, ¿qué pasaría si invertimos los ejes de este diagrama? En lugar de que nosotros definamos las reglas, ¿qué tal si proporcionamos las respuestas y, junto con los datos, encontramos una manera de descubrir cuáles podrían ser las reglas?

La Figura 1-9 muestra cómo se vería esto. Podemos considerar este diagrama de alto nivel como una definición de aprendizaje automático.

![Figura 1-9. Cambiando los ejes para obtener aprendizaje automático](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.9.png)

¿Cuáles son las implicaciones de esto? Ahora, en lugar de intentar descifrar cuáles son las reglas, recopilamos muchos datos sobre nuestro escenario, etiquetamos esos datos, y la computadora puede determinar qué reglas hacen que una pieza de datos corresponda a una etiqueta en particular, mientras que otra pieza corresponde a una etiqueta diferente.

¿Cómo funcionaría esto en nuestro escenario de detección de actividad? Podemos observar todos los sensores que nos proporcionan datos sobre esta persona. Si tienen un dispositivo portátil que detecta información como frecuencia cardíaca, ubicación, velocidad, etc., y recopilamos muchas instancias de estos datos mientras realizan diferentes actividades, terminamos con un conjunto de datos que dice: “Esto es cómo se ve caminar”, “Esto es cómo se ve correr”, y así sucesivamente (Figura 1-10).

![Figura 1-10. De la codificación al aprendizaje automático: recopilación y etiquetado de datos](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.10.png)

Ahora nuestro trabajo como programadores cambia: ya no se trata de formular reglas para determinar las actividades, sino de escribir el código que haga coincidir los datos con las etiquetas. Si logramos hacer esto, podemos ampliar los escenarios que podemos implementar con código.

El aprendizaje automático es una técnica que nos permite lograr esto, pero para comenzar necesitaremos un marco de trabajo, y aquí es donde entra TensorFlow. En la siguiente sección veremos qué es y cómo instalarlo, y más adelante en este capítulo escribirás tu primer código que aprenda el patrón entre dos valores, como en el escenario anterior. Es un simple escenario de "Hola Mundo", pero tiene el mismo patrón básico de código que se utiliza en escenarios extremadamente complejos.

El campo de la inteligencia artificial es amplio y abstracto, abarcando todo lo relacionado con hacer que las computadoras piensen y actúen como lo hacen los seres humanos. Una de las formas en que los humanos adquieren nuevos comportamientos es aprendiendo por ejemplo. Por lo tanto, la disciplina del aprendizaje automático puede considerarse como una vía de acceso al desarrollo de inteligencia artificial. A través de ella, una máquina puede aprender a ver como un humano (un campo llamado visión por computadora), leer texto como un humano (procesamiento de lenguaje natural) y mucho más. En este libro, cubriremos los conceptos básicos del aprendizaje automático utilizando el marco de trabajo TensorFlow.

## ¿Qué es TensorFlow?
TensorFlow es una plataforma de código abierto para crear y utilizar modelos de aprendizaje automático. Implementa muchos de los algoritmos y patrones comunes necesarios para el aprendizaje automático, lo que te ahorra la necesidad de aprender todas las matemáticas y la lógica subyacentes. Así, puedes centrarte exclusivamente en tu escenario. Está dirigido a un amplio público, desde aficionados hasta desarrolladores profesionales e investigadores que buscan expandir los límites de la inteligencia artificial. Lo más importante es que también permite el despliegue de modelos en la web, la nube, dispositivos móviles y sistemas integrados. En este libro, cubriremos cada uno de estos escenarios.

Arquitectura de alto nivel de TensorFlow

![Figura 1-11. Arquitectura de alto nivel de TensorFlow](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.11.png)

El proceso de crear modelos de aprendizaje automático se llama entrenamiento. Aquí, una computadora utiliza un conjunto de algoritmos para aprender sobre las entradas y qué las diferencia entre sí. Por ejemplo, si deseas que una computadora reconozca gatos y perros, puedes usar muchas imágenes de ambos para crear un modelo. La computadora utilizará ese modelo para determinar qué hace que un gato sea un gato y qué hace que un perro sea un perro. Una vez que el modelo está entrenado, el proceso de usarlo para reconocer o clasificar entradas futuras se llama inferencia.

Para entrenar modelos, necesitas varias herramientas de soporte. El primer requisito es un conjunto de APIs para diseñar los modelos. TensorFlow ofrece tres formas principales de hacerlo:

- Codificar todo a mano: Aquí diseñas la lógica para que una computadora aprenda e implementas esto en código (no recomendado).
- Usar estimadores incorporados: Son redes neuronales ya implementadas que puedes personalizar.
- Usar Keras: Una API de alto nivel que encapsula paradigmas comunes de aprendizaje automático en código. Este libro se enfocará principalmente en usar las APIs de Keras para crear modelos.

Existen muchas formas de entrenar un modelo. Normalmente, usarás un único chip, ya sea una unidad central de procesamiento (CPU), una unidad de procesamiento gráfico (GPU) o algo más reciente llamado unidad de procesamiento tensorial (TPU). En entornos avanzados y de investigación, puede emplearse un entrenamiento paralelo en múltiples chips mediante una estrategia de distribución. TensorFlow también admite este enfoque.

Como se discutió anteriormente, si deseas crear un modelo que pueda reconocer gatos y perros, necesitas entrenarlo con muchos ejemplos de ambos. Pero, ¿cómo puedes gestionar estos ejemplos? A menudo, esto puede requerir más codificación que la creación de los propios modelos. TensorFlow incluye APIs para simplificar este proceso, conocidas como TensorFlow Data Services. Estas incluyen:

- Conjuntos de datos preprocesados que puedes usar con una sola línea de código.
- Herramientas para procesar datos en bruto y facilitar su uso.

Más allá de crear modelos, también necesitas ponerlos en manos de los usuarios. Para ello, TensorFlow incluye diversas APIs:

- TensorFlow Serving: Permite ofrecer inferencia de modelos a través de una conexión HTTP para usuarios en la nube o la web.
- TensorFlow Lite: Proporciona herramientas para la inferencia de modelos en sistemas móviles (Android, iOS) y sistemas integrados basados en Linux, como Raspberry Pi.
- TensorFlow Lite Micro (TFLM): Una variante de TensorFlow Lite que permite la inferencia en microcontroladores, dentro del concepto emergente conocido como TinyML.
- TensorFlow.js: Ofrece la capacidad de entrenar y ejecutar modelos directamente en navegadores o en Node.js.

- A continuación, te mostraré cómo instalar TensorFlow para que puedas empezar a crear y utilizar modelos de aprendizaje automático con esta plataforma.

## Usando TensorFlow
En esta sección, veremos las tres formas principales de instalar y usar TensorFlow. Comenzaremos con cómo instalarlo en tu equipo de desarrollo usando la línea de comandos. Luego exploraremos cómo usar el popular IDE PyCharm (entorno de desarrollo integrado) para instalar TensorFlow. Finalmente, veremos Google Colab y cómo se puede utilizar para acceder al código de TensorFlow con un backend basado en la nube desde tu navegador.

### Instalando TensorFlow en Python
TensorFlow admite la creación de modelos utilizando múltiples lenguajes, incluyendo Python, Swift, Java y más. En este libro nos enfocaremos en usar Python, que es el lenguaje de facto para el aprendizaje automático debido a su amplio soporte para modelos matemáticos. Si aún no lo tienes, te recomiendo encarecidamente que visites Python para comenzar y learnpython.org para aprender la sintaxis del lenguaje Python.

Con Python hay muchas formas de instalar frameworks, pero la predeterminada que admite el equipo de TensorFlow es pip.

Entonces, en tu entorno de Python, instalar TensorFlow es tan sencillo como usar:

```python
pip install tensorflow  
```

Ten en cuenta que a partir de la versión 2.1, esto instalará la versión GPU de TensorFlow por defecto. Antes de eso, se usaba la versión CPU. Por lo tanto, antes de instalar, asegúrate de tener una GPU compatible y todos los controladores necesarios para ella. Los detalles sobre esto están disponibles en TensorFlow.

Si no tienes la GPU requerida o los controladores, aún puedes instalar la versión CPU de TensorFlow en cualquier sistema Linux, PC o Mac con:

```python
pip install tensorflow-cpu  
```

Una vez que estés en funcionamiento, puedes probar tu versión de TensorFlow con el siguiente código:

```python
import tensorflow as tf  
print(tf.__version__)  
```

Deberías ver una salida como la de la Figura 1-12. Esto imprimirá la versión actualmente en ejecución de TensorFlow; aquí puedes ver que está instalada la versión 2.0.0.

![Figura 1-12. Ejecución de TensorFlow en Python](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.12.png)

### Usando TensorFlow en PyCharm
Soy particularmente partidario de usar la versión gratuita de la comunidad de PyCharm para construir modelos con TensorFlow. PyCharm es útil por muchas razones, pero una de mis favoritas es que facilita la gestión de entornos virtuales. Esto significa que puedes tener entornos de Python con versiones de herramientas como TensorFlow específicas para tu proyecto en particular. Por ejemplo, si deseas usar TensorFlow 2.0 en un proyecto y TensorFlow 2.1 en otro, puedes separar estos con entornos virtuales y evitar tener que instalar/desinstalar dependencias al cambiar de proyecto. Además, con PyCharm puedes realizar una depuración paso a paso de tu código en Python, algo imprescindible, especialmente si estás comenzando.

Por ejemplo, en la Figura 1-13 tengo un nuevo proyecto llamado example1 y estoy especificando que voy a crear un nuevo entorno usando Conda. Al crear el proyecto, tendré un entorno virtual limpio de Python en el cual puedo instalar cualquier versión de TensorFlow que desee.

![Figura 1-13. Creación de un nuevo entorno virtual mediante PyCharm](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.13.png)

Una vez que has creado un proyecto, puedes abrir el cuadro de diálogo de File → Settings y elegir la entrada para "Project: <nombre de tu proyecto>" en el menú de la izquierda. Luego verás opciones para cambiar la configuración del Project Interpreter y la Project Structure. Elige el enlace de Project Interpreter, y verás el intérprete que estás utilizando, así como una lista de los paquetes que están instalados en este entorno virtual (Figura 1-14).

![Figura 1-14. Adición de paquetes a un entorno virtual](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.14.png)

Haz clic en el botón + a la derecha, y se abrirá un cuadro de diálogo mostrando los paquetes que están disponibles actualmente. Escribe "tensorflow" en el cuadro de búsqueda y verás todos los paquetes disponibles con "tensorflow" en el nombre (Figura 1-15).

![Figura 1-15. Instalación de TensorFlow con PyCharm](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.15.png)

Una vez que hayas seleccionado TensorFlow, o cualquier otro paquete que desees instalar, haz clic en el botón Install Package, y PyCharm hará el resto. 

Una vez que TensorFlow esté instalado, ya puedes escribir y depurar tu código de TensorFlow en Python.

### Usando TensorFlow en Google Colab
Otra opción, que quizás sea la más fácil para comenzar, es usar Google Colab, un entorno de Python alojado al que puedes acceder a través de un navegador. Lo realmente interesante de Colab es que proporciona GPU y TPU como backends, lo que te permite entrenar modelos usando hardware de última generación sin costo alguno.

Cuando visitas el sitio web de Colab, tendrás la opción de abrir Colabs anteriores o iniciar un nuevo cuaderno, como se muestra en la Figura 1-16.

![Figura 1-16. Introducción a Google Colab](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.16.png)

Al hacer clic en el enlace New Python 3 Notebook, se abrirá el editor, donde puedes agregar paneles de código o texto (Figura 1-17). Puedes ejecutar el código haciendo clic en el botón Play (la flecha) a la izquierda del panel.

![Figura 1-17. Ejecución del código TensorFlow en Colab](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.17.png)

Siempre es una buena idea verificar la versión de TensorFlow, como se muestra aquí, para asegurarte de que estás ejecutando la versión correcta. A menudo, la versión incorporada de TensorFlow en Colab estará una o dos versiones detrás de la última versión. Si ese es el caso, puedes actualizarla con pip install, como se mostró anteriormente, simplemente usando un bloque de código como este:

```python
!pip install tensorflow==2.1.0   
```

Una vez que ejecutes este comando, tu entorno actual dentro de Colab usará la versión deseada de TensorFlow.

## Introducción al Aprendizaje Automático
Como vimos antes en el capítulo, el paradigma del aprendizaje automático es aquel en el que tienes datos, esos datos están etiquetados, y quieres descubrir las reglas que relacionan los datos con las etiquetas. El escenario más simple para mostrar esto en código es el siguiente. Considera estos dos conjuntos de números:

X = –1, 0, 1, 2, 3, 4

Y = –3, –1, 1, 3, 5, 7

Existe una relación entre los valores de X e Y (por ejemplo, si X es –1 entonces Y es –3, si X es 3 entonces Y es 5, y así sucesivamente). ¿Puedes verla?

Después de unos segundos probablemente notaste que el patrón aquí es Y = 2X – 1. ¿Cómo lo descubriste? Diferentes personas lo deducen de maneras distintas, pero típicamente se observa que X incrementa en 1 en su secuencia, y Y incrementa en 2; por lo tanto, Y = 2X +/- algo. Luego, miran el caso donde X = 0 y ven que Y = –1, así que concluyen que la respuesta podría ser Y = 2X – 1. Después, verifican los otros valores y ven que esta hipótesis "encaja", y la respuesta es Y = 2X – 1.

Esto es muy similar al proceso de aprendizaje automático. Vamos a echar un vistazo a algún código de TensorFlow que podrías escribir para que una red neuronal descubra esto por ti. Aquí está el código completo, usando las APIs de TensorFlow Keras. No te preocupes si aún no tiene sentido; lo analizaremos línea por línea:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
```

Has oído hablar de redes neuronales y probablemente hayas visto diagramas que las explican usando capas de neuronas interconectadas, algo como la Figura 1-18.

![Figura 1-18. Una red neuronal típica](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.18.png)

Cuando ves una red neuronal como esta, considera cada uno de los "círculos" como una neurona, y cada columna de círculos como una capa. Entonces, en la Figura 1-18, hay tres capas: la primera tiene cinco neuronas, la segunda tiene cuatro, y la tercera tiene dos.

En nuestro código, definimos la red neuronal más simple posible: tiene solo una capa y contiene únicamente una neurona:

```python
model = Sequential([Dense(units=1, input_shape=[1])])
```

Cuando usas TensorFlow, defines tus capas usando Sequential. Dentro de Sequential, especificas cómo se ve cada capa. Solo tenemos una línea dentro de nuestro Sequential, así que solo tenemos una capa.

Luego defines cómo se ve la capa usando las API de keras.layers. Hay muchos tipos diferentes de capas, pero aquí estamos usando una capa Dense. "Dense" significa un conjunto de neuronas totalmente (o densamente) conectadas, lo cual puedes ver en la Figura 1-18 donde cada neurona está conectada a cada neurona en la siguiente capa. Es la forma más común de tipo de capa. Nuestra capa Dense tiene units=1 especificado, por lo que tenemos solo una capa densa con una neurona en toda nuestra red neuronal. Finalmente, cuando especificas la primera capa en una red neuronal (en este caso, es nuestra única capa), tienes que decirle cuál es la forma de los datos de entrada. En este caso, nuestros datos de entrada son nuestra X, que es solo un valor único, por lo que especificamos que esa es su forma.

La siguiente línea es donde realmente comienza la diversión. Veámosla de nuevo:

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

Si has hecho algo con aprendizaje automático antes, probablemente has visto que implica muchas matemáticas. Si no has hecho cálculo en años, podría haberte parecido una barrera de entrada. Aquí es donde entran las matemáticas: es el núcleo del aprendizaje automático.

En un escenario como este, la computadora no tiene idea de cuál es la relación entre X e Y. Así que hará una suposición. Por ejemplo, podría adivinar que Y = 10X + 10. Luego necesita medir qué tan buena o mala es esa suposición. Ese es el trabajo de la función de pérdida (loss function).

Ya sabe las respuestas cuando X es –1, 0, 1, 2, 3 y 4, por lo que la función de pérdida puede compararlas con las respuestas para la relación adivinada. Si adivinó Y = 10X + 10, entonces cuando X es –1, Y será 0. La respuesta correcta allí era –3, por lo que está un poco equivocada. Pero cuando X es 4, la respuesta adivinada es 50, mientras que la correcta es 7. Eso está realmente lejos.

Con esta información, la computadora puede hacer otra suposición. Ese es el trabajo del optimizador. Aquí es donde se usa el cálculo intensivo, pero con TensorFlow, eso puede ocultarse de ti. Solo seleccionas el optimizador adecuado para usar en diferentes escenarios. En este caso elegimos uno llamado sgd, que significa descenso de gradiente estocástico (stochastic gradient descent)—una función matemática compleja que, al recibir los valores, la suposición previa y los resultados de calcular los errores (o pérdida) en esa suposición, puede generar otra. Con el tiempo, su trabajo es minimizar la pérdida, y al hacerlo, acercar la fórmula adivinada cada vez más a la respuesta correcta.

A continuación, simplemente formateamos nuestros números en el formato de datos que esperan las capas. En Python, hay una biblioteca llamada Numpy que TensorFlow puede usar, y aquí ponemos nuestros números en un array de Numpy para facilitar su procesamiento:

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```

El proceso de aprendizaje comenzará entonces con el comando model.fit, así:

```python
model.fit(xs, ys, epochs=500)
```

Puedes leer esto como "ajusta los Xs a los Ys, e inténtalo 500 veces". Entonces, en el primer intento, la computadora adivinará la relación (es decir, algo como Y = 10X + 10) y medirá qué tan buena o mala fue esa suposición. Luego alimentará esos resultados al optimizador, que generará otra suposición. Este proceso se repetirá, con la lógica de que la pérdida (o error) disminuirá con el tiempo, y como resultado, la "suposición" será cada vez mejor.

La Figura 1-19 muestra una captura de pantalla de esto ejecutándose en un cuaderno Colab. Echa un vistazo a los valores de pérdida a lo largo del tiempo.

![Figura 1-19. Entrenando la red neuronal](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.19.png)

Podemos ver que durante las primeras 10 épocas, la pérdida pasó de 3.2868 a 0.9682. Es decir, después de solo 10 intentos, la red estaba funcionando tres veces mejor que con su suposición inicial. Luego observa lo que ocurre en la quingentésima época (Figura 1-20).

![Figura 1-20. Entrenando la red neuronal: las últimas cinco épocas](/assets/markdown/IA_and_Machine_Learning_for_Coders/img/figure1.20.png)

Ahora podemos ver que la pérdida es 2.61 × 10⁻⁵. La pérdida se ha vuelto tan pequeña que el modelo prácticamente ha descubierto que la relación entre los números es Y = 2X – 1. La máquina ha aprendido el patrón entre ellos.

Nuestra última línea de código luego usó el modelo entrenado para obtener una predicción como esta:

```python
print(model.predict(np.array([10.0])))
```

> El término "predicción" se utiliza típicamente cuando se trata de modelos de aprendizaje automático. ¡No pienses en ello como mirar al futuro! Este término se usa porque estamos tratando con una cierta cantidad de incertidumbre. Piensa en el escenario de detección de actividad que hablamos anteriormente. Cuando la persona se movía a cierta velocidad, probablemente estaba caminando. De manera similar, cuando un modelo aprende sobre los patrones entre dos cosas, nos dirá cuál es probablemente la respuesta. En otras palabras, está prediciendo la respuesta. (Más adelante también aprenderás sobre la inferencia, donde el modelo elige una respuesta entre muchas, e infiere que ha elegido la correcta).

¿Qué crees que será la respuesta cuando le pidamos al modelo que prediga Y cuando X es 10? Podrías pensar instantáneamente en 19, pero eso no es correcto. Elegirá un valor muy cercano a 19. Hay varias razones para esto. Primero, nuestra pérdida no era 0. Aún era una cantidad muy pequeña, por lo que debemos esperar que cualquier respuesta predicha esté desviada por una cantidad muy pequeña. En segundo lugar, la red neuronal está entrenada solo con una pequeña cantidad de datos, en este caso, solo seis pares de valores (X, Y).

El modelo solo tiene una sola neurona, y esa neurona aprende un peso y un sesgo, de modo que Y = WX + B. Esto parece exactamente la relación Y = 2X – 1 que queremos, donde querríamos que aprenda que W = 2 y B = –1. Dado que el modelo fue entrenado con solo seis datos, nunca se podría esperar que la respuesta fuera exactamente estos valores, pero algo muy cercano a ellos.

Ejecuta el código por ti mismo para ver qué obtienes. Yo obtuve 18.977888 cuando lo ejecuté, pero tu respuesta puede diferir ligeramente porque cuando la red neuronal se inicializa por primera vez hay un elemento aleatorio: tu suposición inicial será ligeramente diferente de la mía y de la de una tercera persona.

## Viendo lo que la red aprendió
Este es obviamente un escenario muy simple, donde estamos emparejando Xs con Ys en una relación lineal. Como se mencionó en la sección anterior, las neuronas tienen parámetros de peso y sesgo que aprenden, lo que hace que una sola neurona sea suficiente para aprender una relación como esta: es decir, cuando Y = 2X – 1, el peso es 2 y el sesgo es –1. Con TensorFlow, en realidad podemos echar un vistazo a los pesos y sesgos que se aprendieron, con un simple cambio en nuestro código como este:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print("Here is what I learned: {}".format(l0.get_weights()))
```

La diferencia es que creé una variable llamada l0 para contener la capa Dense. Luego, después de que la red termina de aprender, puedo imprimir los valores (o pesos) que la capa aprendió.

En mi caso, la salida fue la siguiente:

```python
Here is what I learned: [array([[1.9967953]], dtype=float32), array([-0.9900647], dtype=float32)]
```

Por lo tanto, la relación aprendida entre X e Y fue Y = 1.9967953X – 0.9900647.

Esto está bastante cerca de lo que esperaríamos (Y = 2X – 1), y podríamos argumentar que es incluso más cercano a la realidad, porque estamos asumiendo que la relación se mantendrá para otros valores.

## Resumen
Eso es todo para tu primer "Hello World" de aprendizaje automático. Podrías estar pensando que esto parece una exageración masiva para algo tan simple como determinar una relación lineal entre dos valores. Y tendrías razón. Pero lo interesante de esto es que el patrón de código que hemos creado aquí es el mismo patrón que se utiliza para escenarios mucho más complejos. Verás estos a partir del Capítulo 2, donde exploraremos algunas técnicas básicas de visión por computadora: la máquina aprenderá a "ver" patrones en imágenes e identificar lo que hay en ellas.
