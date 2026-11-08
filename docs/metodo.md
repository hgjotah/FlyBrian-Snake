# Método y reproducibilidad

## Del conectoma al grafo

Un conectoma describe neuronas y contactos observados en una reconstrucción anatómica. Brian usa todos los IDs revisados de FAFB 783 y todas las filas de la tabla oficial de conectividad por neuropilo. El manifiesto identifica los archivos fuente, tamaños, hashes y revisión de las anotaciones. Las filas agregadas distinguen neuropilo, presinapsis, postsinapsis y número de contactos. Los pares dirigidos se cuentan una vez a través de neuropilos. El importador conserva IDs de 64 bits y utiliza índices compactos para el CSR.

Para la dinámica rápida se prioriza el transmisor curado cuando determina un signo. La predicción cubre los casos restantes. ACh suma impulso y GABA y glutamato lo restan por la convención del modelo de referencia. Otros transmisores quedan representados en el grafo con eficacia rápida nula. En las KC gamma, la anotación curada de ACh corrige la etiqueta predicha de dopamina que aparece en gran parte de la población. Los datos de neurotransmisor conservan su incertidumbre y el efecto de glutamato depende de receptores que este modelo simplifica.

## Dinámica y precisión

La dinámica usa `dv/dt = (v_rest - v + g) / τ_m` y `dg/dt = -g / τ_s`. El motor compara el voltaje con el umbral cada 0,1 ms. El impulso que llega después del retraso añade `0,275 mV × número de contactos × signo × escala plástica` a `g`. El disparo restaura el voltaje y pone `g` a cero. Durante el periodo refractario el estado continuo queda congelado y puede acumular impulsos sinápticos. La entrada externa usa eventos Bernoulli con probabilidad `1 - exp(-Hz × dt / 1000)` y salto de 68,75 mV.

La solución lineal se evalúa en doble precisión. Si `v ≤ v_umbral` y `g ≤ v_umbral - v_rest`, la neurona permanece bajo umbral mientras no llegue otra entrada. El motor avanza ese estado de forma analítica al llegar una entrada o al terminar la ventana. Las neuronas candidatas a disparar sí se comprueban en cada tick. Se conserva el orden de las emisiones para la acumulación reproducible. Un ensayo de referencia en Brian2 comprueba signos, retardo, spikes y voltajes. La comparación con el escaneo anterior de precisión simple se registra en los resultados.

La validación SugarR adapta el cuaderno de Shiu a FAFB 783. El estímulo usa 20 de los 21 root IDs de la publicación original. El ID restante pertenece a una revisión distinta y se registra como ausente. La comparación utiliza la respuesta de MN9 y la propagación fuera de las neuronas estimuladas. El cuaderno original promedia más ensayos, usa v630 para ese ejemplo y aplica convenciones distintas para parte de las entradas.

## Memoria local

Se seleccionan filas observadas de KC gamma excitadoras hacia MBON01 en los neuropilos MB_ML y hacia MBON11 en MB_PED. Cada lado se trata como canal independiente y cada DAN contribuye según la fracción de contactos salientes de su tipo en ese neuropilo. La anatomía publicada de γ5 y γ1/pedc sustenta esta selección. Dentro de cada neuropilo la ubicación de un contacto respecto al subcompartimento queda aproximada.

En un tick con spikes de DAN, el motor calcula `e = exp(-edad_pre/τ_pre) × exp(-edad_post/τ_post)`. Por defecto exige un spike previo en ambas neuronas y las constantes son 350 y 100 ms. La escala decrece por `η × puerta_DAN × e × (escala - mínimo)`, con mínimo 0,20. Una conexión conserva su signo. Los estados transitorios se reinician entre partidas y los pesos se conservan. Se simula depresión y la elegibilidad depende del orden de actividad y refuerzo. La regla concreta es una hipótesis implementable inspirada en los resultados fisiológicos y deja fuera la cinética de receptores, la potenciación y las dinámicas de largo plazo.

## Interfaz experimental

La posición relativa de comida se proyecta sobre los ejes anterior y lateral de la serpiente. El codificador excita LC10a lateralizadas y LC10d para delante. El perfil anterior también activaba LPLC2 para la comida detrás. El perfil actual utiliza una etiqueta KC para esa señal y hace más visibles las manzanas próximas a los lados. La pared lateral usa LC4. Al tener la pared justo delante, la señal visual llega a LC6 con más intensidad en el lado cerrado. En ensayos aislados del conectoma, LC6 activó la salida descendente contraria. En el punto medio se aplica una asimetría sensorial de 1,5 en el canal izquierdo para reducir empates bilaterales. Con una manzana en la línea de visión lateral y la pared justo enfrente, el perfil actual atenúa LC4 y LC6 en el flanco de la manzana. La señal frontal permanece activa y los disparos descendentes determinan el giro. Subconjuntos KC fijos reciben etiquetas sintéticas de rumbo, comida y borde. Estas son convenciones experimentales de estímulo directo.

El decodificador compara los spikes medios de DNa01/DNa02 por hemisferio. El lado más activo determina el giro relativo. Un empate continúa recto. Esta lateralización tiene respaldo anatómico y fisiológico. La transformación a Snake es fija. [Yang et al., Cell](https://doi.org/10.1016/j.cell.2024.08.033)

La recompensa graduada suma +30 por una manzana y −60 por una colisión. El progreso suma 0,25 por cada paso que mejora la mejor distancia legal alcanzada hacia la manzana actual. Volver a una posición ya alcanzada en un ciclo no vuelve a generar ese premio. Con una manzana en el interior, tocar el perímetro resta 12 puntos por defecto. En las tres casillas próximas a una pared, acercarse o quedarse cerca sin necesidad resta 3, 6 o 9 puntos según la distancia. El alcance se reduce en tableros pequeños para dejar libre la zona central. Si la manzana está en cualquier borde, se suspende el castigo de proximidad y contacto durante ese turno y los dos movimientos posteriores a recogerla. Cada tres movimientos, o al comer, una ruta óptima suma +2 y una ruta con hasta dos pasos de exceso que aún se acerca suma +1. El segundo movimiento que se aparta de la ruta legal mínima resta 2 una sola vez. El contador se reinicia al recoger la manzana. La última manzana del objetivo de una por casilla añade +150 y eleva por separado la tasa del pulso apetitivo. El entrenamiento procesa ese pulso antes de cerrar el episodio.

El saldo neto activa PAM01 si es positivo y PPL101 si es negativo mediante `1 − exp(−magnitud / escala)`, con escalas de 8 y 18 respectivamente. Al límite de 120 Hz, una manzana activa aproximadamente 117 Hz de señal apetitiva, la proximidad máxima 10 Hz de señal aversiva y el choque 116 Hz de señal aversiva. La modificación de pesos responde a los spikes y a la elegibilidad local.

Al reanudar un checkpoint se conserva su perfil. En el perfil graduado anterior, el progreso y el castigo podían activar los dos canales a la vez. En el perfil histórico, con +10 por manzana, −20 por colisión, progreso de 0,1, la cercanía era de `−0,05 / (1 + distancia mínima)²` y escalas de señal 2 y 4.

## Experimentos A–I

| Condición | Intervención |
| --- | --- |
| A | Pesos iniciales |
| B | Aprendizaje intacto |
| C | Pesos congelados durante el entrenamiento |
| D | Refuerzo dopaminérgico externo deshabilitado en entrenamiento. La evaluación comparte el protocolo de A y B |
| E | Permutación de destinos dentro de cada neuropilo. Conserva el grado de filas por fuente y destino, el número de contactos por fila y el sustrato plástico. Puede producir multiaristas y lazos |
| F | Secuencia de refuerzos permutada de episodios donantes con semillas separadas |
| G | Restauración de la sesión B desde disco y evaluación con estado transitorio reiniciado |
| H | Entrenamiento en celdas de paridad par y evaluación en celdas de paridad impar con semillas nuevas. La referencia inicial usa esas mismas celdas impares |
| I | Ablación funcional de MBON durante entrenamiento y evaluación |

La evaluación congela pesos y usa semillas separadas. Las diferencias se calculan por semilla y el intervalo percentil del 95 % usa 2.000 remuestreos de semillas de evaluación a partir de cinco semillas. Esto mide variación entre tableros para las redes observadas. Inferir la variación del aprendizaje entre inicializaciones requeriría repetir cada condición con múltiples semillas de entrenamiento.

Las partidas manuales, los pulsos y los silenciamientos interactivos se identifican en el diario. Los resultados conductuales, la actividad y las modificaciones sinápticas son medidas distintas. Un cambio local y reproducible de peso establece que la regla actúa. La mejora en Snake se evalúa por separado, frente a pesos iniciales, controles y situaciones reservadas.


## Continuar la investigación

Ejecuta estudios separados con varias semillas de entrenamiento y archiva un JSON de configuración por corrida. Mantén el mismo límite de movimientos para A, B y sus controles. Compara primero manzanas por semilla y después supervivencia, manzanas junto a pared y actividad descendente.

Para investigar una mejora conductual, comprueba la respuesta de las DNa con una secuencia de observaciones idéntica antes y después de cargar los pesos. Las comparaciones de actividad KC-A en el visor aíslan la respuesta asociativa. Las ablaciones, los controles temporales y las reservas de posiciones ayudan a comprobar si esa respuesta también altera las decisiones de juego.


En el perfil edge_target, la señal LC10d de una manzana alineada de frente se reduce a la mitad. La señal visual de pared lateral se reduce cuando esa manzana está de frente y hay espacio para avanzar. Las dos ganancias son calibraciones de la interfaz sensorial y se mantienen fijas durante el aprendizaje. Las etiquetas espaciales KC mantienen la distancia de la pared. La práctica `routes` modifica las posiciones iniciales y la condición de fin de cada episodio. El segundo desvío entrega el castigo una vez por manzana y el intento continúa. El preset y la opción de escritorio entrenan la coincidencia KC–DAN sin requerir un spike postsináptico. Esta variante de la regla de plasticidad es una hipótesis del modelo. Las decisiones usan el mismo motor que las partidas completas.
