# Resultados de Brian

## Respuesta del teclado en la Arena

En Windows, una prueba Qt reprodujo tres giros válidos en un tablero de 7×7 y midió el tiempo entre la pulsación y la actualización del tablero del jugador. Los tiempos fueron 615, 265 y 171 ms en la versión anterior y 13, 77 y 127 ms con el reloj nuevo. Son tres pulsaciones de prueba en el mismo equipo. Los turnos que requieren más cálculo neuronal pueden superar el intervalo objetivo de 135 ms.

La primera flecha inicia el duelo y los movimientos del jugador se dibujan al abrir cada turno. El siguiente turno espera a que Brian termine el actual y conserva el orden de hasta dos giros pendientes.

## Cuadrados repetidos en el borde

Con los pesos de la sesión de 1.000 partidas y las mismas 32 semillas, la corrección visual redujo de 11 a 0 las partidas con dos vueltas consecutivas por las mismas cuatro casillas y con la misma manzana. Cada partida duró como máximo 60 movimientos. La media de manzanas pasó de 6,28 a 8,16 y los choques bajaron de 16 a 12.

En la semilla 2, Brian estaba en (5, 0) mirando hacia arriba con la manzana en (6, 0). Antes, la señal visual frontal lateral derecha tenía 128,38 Hz y Brian giraba a la izquierda. Al reconocer que la manzana está en un flanco transitable, esa señal bajó a 31,50 Hz. Brian giró a la derecha y obtuvo la manzana, con el mismo saldo de +28,50 previsto por las reglas. [Comparación por semilla](../data/runs/arena-loop-comparison.json)

## Arena en vivo y evitación de paredes

Se comparó la Arena anterior con la nueva en las mismas 16 semillas, con tablero 7×7, tope de 60 movimientos y plasticidad congelada. La comparación de pesos entrenados reutilizó exactamente el checkpoint de 1.000 partidas.

| Pesos de Brian | Choques antes → ahora | Manzanas medias antes → ahora | Supervivencia media antes → ahora |
| --- | ---: | ---: | ---: |
| Iniciales | 16 → 11 | 3,06 → 6,44 | 20,88 → 42,69 |
| Checkpoint de 1.000 partidas | 16 → 8 | 3,06 → 6,38 | 20,88 → 44,00 |

En una partida reproducida, un paso hacia el borde tenía retorno −0,139 y activaba a la vez las señales apetitiva y aversiva. El nuevo perfil aplica una sola señal según el saldo y suprime el premio de progreso al dirigirse hacia una pared próxima. La calibración de las señales visuales frontales también modifica el estímulo antes del movimiento. Los resultados de la tabla reflejan el conjunto de esos cambios. [Partidas por semilla](../data/runs/arena-wall-comparison.json) · [Captura durante el duelo](../data/runs/arena-en-vivo.png)

## Revisión de la arena y la ráfaga

El modo rápido conserva el conectoma completo, la regla de plasticidad y la semilla de entrenamiento. Acorta las ventanas neuronales a 36 y 18 ms y limita las partidas a 12 movimientos. La validación específica de la ráfaga se guardó por separado.

| Corrida | Episodios | Movimientos | Tiempo de cálculo | Conexiones modificadas al final |
| --- | ---: | ---: | ---: | ---: |
| Ventanas estándar, límite de 12 | 40 | 370 | 34,7 s | 758 |
| Ráfaga, límite de 12 | 40 | 303 | 16,9 s | 724 |
| Ráfaga, límite de 12 | 100 | 759 | 53,4 s | 752 |

La corrida de 100 episodios tardó unos 60 segundos de principio a fin, incluidos validación y guardado. Esto sitúa 1.000 partidas cortas en el orden de 7 a 10 minutos en este equipo. Es una estimación a partir de las dos muestras de ráfaga. Cambiar las ventanas y el límite de turnos cambia el protocolo experimental. [Validación de ráfaga](../data/runs/validation-burst.json) · [Sesión de 100 partidas](../data/runs/rafaga-100.sqlite)

## Resultados históricos

Los resultados de las secciones siguientes se obtuvieron antes del nuevo perfil de recompensas, con el protocolo original.

## Integridad y validación neuronal

FlyWire FAFB v783 se importó y verificó en Windows 11. El grafo contiene **139.255 neuronas**, **16.847.997 filas por neuropilo**, **15.091.983 pares dirigidos** y **54.492.922 contactos anatómicos**. Hay 139.241 neuronas con anotaciones y coordenadas representativas. La regla de memoria puede actuar en 4.157 filas KC→MBON. [Manifiesto](../data/processed/manifest.json)

- **Propagación.** En 1 s neuronal, SugarR produjo 12.020 spikes en 372 neuronas. De ellos, 9.765 ocurrieron fuera del grupo estimulado. MN9 disparó a 70 Hz. La semilla repetida produjo los mismos conteos. El reposo tuvo cero spikes y la estimulación de LC10a izquierda produjo 9 spikes en las salidas descendentes observadas. [Validación](../data/runs/validation.json)
- **Condicionamiento.** Doce emparejamientos modificaron 388 conexiones. En las filas vinculadas al patrón A, el cambio absoluto medio de escala fue 15,6 %. En las filas vinculadas al patrón B fue 0 %. La respuesta MBON de prueba pasó de 50 a 48 spikes para A y permaneció en 54 para B. Los tres controles, pesos congelados, refuerzo omitido y dopamina aislada, produjeron cero cambios. La restauración del checkpoint recuperó los pesos exactamente. [Validación](../data/runs/validation.json)
- **Referencia numérica.** El escaneo anterior en precisión simple produjo 12.198 spikes SugarR y el motor actual en doble precisión produjo 12.020, una diferencia de 1,46 %. Ambos midieron 70 Hz en MN9 y los mismos cambios de condicionamiento. Brian2 coincidió en los spikes del circuito de verificación y en su voltaje dentro de la tolerancia probada. [Escaneo anterior](../data/runs/validation-active-scan.json)
- **Currículo.** Se recorrieron las etapas 1 a 4 en una sesión independiente, con ocho ensayos de asociación y dos episodios por etapa de juego. Se verificó la transición con memoria conservada. [Sesión del currículo](../data/sessions/Brian_curriculo.sqlite)

## Controles de Snake

Etapa 4, tablero 7×7, hasta 100 movimientos por episodio, ocho episodios de entrenamiento por condición cuando correspondía y diez semillas nuevas de evaluación. Los intervalos son percentiles bootstrap emparejados al 95 %. En H, la referencia se evalúa en la misma reserva de posiciones impares que la condición entrenada.

| Condición | Manzanas medias | Diferencia frente a la referencia inicial (IC 95 %) | Filas modificadas |
| --- | ---: | ---: | ---: |
| A · Inicial | 2,6 | 0,0 [0,0, 0,0] | 0 |
| B · Entrenada | 2,6 | 0,0 [0,0, 0,0] | 270 |
| C · Congelada | 2,6 | 0,0 [0,0, 0,0] | 0 |
| D · Refuerzo DA omitido | 2,6 | 0,0 [0,0, 0,0] | 0 |
| E · Destinos permutados | 0,0 | −2,6 [−4,0, −1,4] | 570 |
| F · Refuerzos desordenados | 2,6 | 0,0 [0,0, 0,0] | 516 |
| G · Restaurada desde disco | 2,6 | 0,0 [0,0, 0,0] | 270 |
| H · Reserva de posiciones impares | 3,3 | 0,0 [0,0, 0,0] | 492 |
| I · Ablación MBON | 2,8 | +0,2 [0,0, 0,5] | 0 |

Los cambios sinápticos del entrenamiento B persistieron en G. A y B tuvieron la misma puntuación y la misma supervivencia media, 16,3 movimientos. La pérdida de rendimiento en E se refiere a la perturbación del grafo bajo el protocolo de permutación aplicado. La mejora atribuible al entrenamiento fue de 0 en esta evaluación. Los intervalos describen los tableros evaluados para una sola semilla de entrenamiento. [Informe A–I](../data/runs/controles-v783/report.json) · [Evaluación CSV](../data/runs/controles-v783/evaluation.csv)

## Checkpoints de 100 y 1.000 partidas

Esta corrida de etapa 4 usó un límite de 12 movimientos por episodio. Se guardaron 1.000 episodios y 9.340 movimientos, equivalentes a 840,6 segundos neuronales y 863,8 segundos de cálculo acumulado por movimiento. SQLite devolvió `ok` en su comprobación de integridad. La evaluación congeló los pesos y usó las mismas 20 semillas nuevas para las tres condiciones.

| Pesos | Conexiones modificadas | Reducción media entre las modificadas | Manzanas medias | Supervivencia media |
| --- | ---: | ---: | ---: | ---: |
| Iniciales | 0 | — | 1,20 | 9,20 |
| Tras 100 partidas | 762 | 15,1 % | 1,20 | 9,05 |
| Tras 1.000 partidas | 762 | 38,3 % | 1,20 | 9,05 |

En ambos checkpoints, la diferencia emparejada frente a los pesos iniciales fue 0 manzanas, con IC 95 % [0, 0]. La supervivencia cambió −0,15 movimientos, con IC 95 % [−0,40, 0]. El aprendizaje sináptico medido se intensificó entre los hitos. El rendimiento en Snake permaneció sin mejora bajo este protocolo. [Sesión con los checkpoints](../data/sessions/Brian_v783_1000.sqlite) · [Informe emparejado](../data/runs/hitos-v783/report.json) · [Evaluación CSV](../data/runs/hitos-v783/evaluation.csv)

## Rendimiento

Medición en Windows 11 con Intel Core i5-12450H, 12 CPU lógicas y 15,7 GiB de RAM disponible para el sistema. La apertura del grafo y los circuitos tomó 0,86 s. La primera ventana con el kernel en caché tomó 0,42 s.

| Operación | Medición |
| --- | ---: |
| Ventana de 30 ms neuronales, pesos congelados, mediana de diez | 37,0 ms de reloj |
| Ventana de 30 ms neuronales, plasticidad habilitada, mediana de diez | 37,4 ms de reloj |
| CPU en el benchmark | 99,4 % de un núcleo lógico |
| Pico de RAM del proceso | 493 MiB |
| Checkpoint comprimido de la prueba | 521 kB en 89 ms |
| Exportación de una muestra de tres movimientos | 6,2 ms |
| Repintado nativo de 3.200 puntos en el visor, mediana | 24,8 ms |

El tiempo por movimiento depende de las poblaciones activas y de la cantidad de propagación. El benchmark neuronal y el del visor se miden por separado. [Benchmark CPU](../data/runs/benchmark.json) · [Benchmark de Qt nativo](../data/runs/desktop-benchmark.json)

## Pruebas

La suite automatizada ejecutada incluye reglas de Snake, IDs de 64 bits, signos, demora, Brian2, elegibilidad, congelación, reserva de manzanas, permutación, recuperación de descarga, persistencia, el grafo completo y el bucle de eventos de Qt durante una partida simulada. La suite completa terminó con 21 pruebas aprobadas. La instalación PowerShell se probó con los datos oficiales disponibles y comprobación de hashes. En un entorno separado sin datos, el instalador ejecutó las 19 pruebas independientes del conectoma y omitió las dos que lo requieren. [Registro de la suite completa](../data/runs/tests.txt)


## Rectas y giros

En 24 tramos de cinco casillas, la interfaz previa completó las rectas de borde en once movimientos. La calibración del estímulo frontal y de la señal de pared paralela completó los 24 tramos en los cinco movimientos mínimos. En seis partidas de hasta cien movimientos con las mismas semillas, los giros con manzana alineada delante bajaron de 30 a 1. Se recogieron 69 manzanas frente a 70 y una de las seis partidas terminó por choque. Estos resultados miden la calibración sensorial. Un ensayo separado de 120 episodios modificó 917 pesos y mantuvo 12 de 24 casos reservados óptimos, sin mejora adicional demostrada. [Datos de la revisión](../data/runs/zigzag-revision.json).

## Práctica de paredes

Se añadió un protocolo de seis escenarios en cuatro orientaciones, tablero 10×10 y límite de 20 movimientos por ejercicio. La evaluación usa la misma señal sensorial de peligro antes y después del entrenamiento, con los pesos congelados. Un éxito exige salir a una zona segura o recoger la manzana en el control de avance paralelo. Sobrevivir hasta el límite no cuenta como éxito.

Un ensayo de 72 episodios sin exigir disparo postsináptico modificó 561 conexiones y redujo los éxitos de 42/48 a 40/48, del 87,5 % al 83,3 %. Los controles congelado y sin dopamina mantuvieron 42/48 y cero cambios de pesos. Este candidato empeoró la evasión. [Primer ensayo](../data/runs/paredes-estudio-20261107_170427/report.json).

El protocolo con coincidencia postsináptica seleccionó checkpoints mediante 24 escenarios de validación independientes. Tras 12 de 48 episodios, la validación pasó del 83,3 % al 87,5 % y los episodios posteriores no la aumentaron. El checkpoint seleccionado modificó 290 conexiones. En los 48 casos de prueba permaneció en 42/48, con diferencia emparejada cero e intervalo bootstrap del 95 % [0, 0]. Ambos controles mantuvieron los mismos resultados y cero cambios de pesos. La validación mejoró, pero la prueba reservada no demuestra una mejora conductual atribuible al aprendizaje. [Ensayo con selección](../data/runs/paredes-validacion-20261107_171024/report.json).

Estos resultados corresponden al protocolo anterior, con la señal auxiliar activada. No justifican afirmar que Brian ha aprendido a evitar todas las paredes ni que llegará a hacerlo simplemente acumulando episodios.

## Corrección de la regresión en paredes

La revisión encontró tres errores de diseño. La señal auxiliar se propagaba automáticamente a partidas completas y Arena sin compararla con el perfil previo. El selector priorizaba los éxitos y podía aceptar un aumento de choques. Además, el ejercicio podía premiar salir de la pared original mientras Brian seguía pegado a otro borde, convirtiendo un castigo de pared en refuerzo neto positivo.

La comparación utilizó una copia en RAM de los mismos pesos activos, 12 semillas comunes y hasta 150 movimientos por partida, con plasticidad congelada. Cambió únicamente la tasa de la señal auxiliar.

| Señal auxiliar | Partidas con choque | Manzanas totales | Turnos evaluados |
| --- | ---: | ---: | ---: |
| 180 Hz, versión anterior | 10/12 | 115 | 931 |
| Desactivada | 5/12 | 192 | 1.423 |

La diferencia demuestra una regresión por la interfaz sensorial en esta muestra. Su corrección conserva los pesos y no representa aprendizaje adicional. [Comparación reproducible y resultados por semilla](../data/runs/wall-regression-audit.json).

Se retiró la activación automática de la señal, se corrigió la condición del premio de evasión y se situaron las manzanas de ese ejercicio en el interior seguro. Los candidatos deben superar controles de choques por semilla y de transferencia a partidas completas antes de entrar en la memoria activa. Los pesos pendientes se guardan aparte y no pasan a Arena al interrumpir un bloque. Las pruebas incluyen el caso que producía un premio positivo junto a otro borde, el rechazo de más choques pese a más éxitos y la conservación de la memoria aceptada al guardar y volver a cargar.

La copia de memoria usada en la revisión tenía 1.210 conexiones modificadas. De ellas, 1.207 estaban a menos de 0,001 del mínimo de 0,2. Sobre esa memoria, 48 ejercicios con exigencia postsináptica cambiaron cero pesos y la variante KC+dopamina cambió 18. Con pesos iniciales, 24 ejercicios cambiaron 884 y 917 conexiones respectivamente. Esto distingue una memoria casi saturada de un motor incapaz de modificar sinapsis. La práctica adopta la variante KC+dopamina sin introducir la señal auxiliar que causaba la regresión.

El candidato de 48 episodios no mejoró la validación y fue rechazado. La evaluación reservada de 48 ejercicios conservó 45 éxitos y tres choques con los pesos aceptados. El resultado confirma la conservación de la referencia; no demuestra aprendizaje conductual adicional. La copia de origen y los checkpoints de la comprobación quedan en el [directorio de verificación](../data/runs/wall-fix-learning/report.json). Las actualizaciones por episodio se pueden inspeccionar en la gráfica de entrenamiento.

La suite completa terminó con 77 pruebas aprobadas. La comprobación nativa recorrió un bloque de paredes, su validación y el regreso a partidas completas sin límite. La memoria en entrenamiento permaneció separada de la memoria activa. Las validaciones atienden interrupciones entre movimientos para permitir pausar o entrar en Arena.

## Partidas consecutivas

Se revisó la sesión guardada `Brian_20261107_191559_968319.sqlite`, con semilla base 10 y tablero 10×10. Incluía las caídas de 58 a 1 manzana y de 74 a 6. Las dos partidas cortas se reprodujeron aisladamente, con plasticidad congelada, tanto con el checkpoint anterior a las partidas buenas como con la memoria posterior.

| Episodio | Resultado registrado | Con pesos anteriores | Con pesos posteriores |
| --- | --- | --- | --- |
| 3 | 1 manzana, 10 movimientos | 1 manzana, 10 movimientos | 1 manzana, 10 movimientos |
| 9 | 6 manzanas, 34 movimientos | 6 manzanas, 34 movimientos | 6 manzanas, 34 movimientos |

Estas dos caídas no se explican por deterioro de los pesos en la partida previa. Las mismas semillas ya producen esos resultados con los pesos anteriores. [Reproducciones de los casos registrados](../data/runs/consecutive-games-audit/recorded-drops.json).

Una comparación separada utilizó diez semillas nuevas y un límite de evaluación de 120 movimientos. La secuencia de manzanas fue idéntica con aprendizaje y con pesos congelados: 19, 18, 15, 15, 15, 4, 14, 18, 13 y 3. Cada uno de los 20 episodios se reprodujo en otro simulador recién construido con sus mismos pesos de entrada. Coincidieron manzanas, movimientos, choques, actualizaciones sinápticas y pesos finales. Esto comprueba la independencia de la actividad temporal en la muestra evaluada. No garantiza ausencia de efectos del aprendizaje en otros pesos, semillas o partidas más largas. [Auditoría y copia de origen](../data/runs/consecutive-games-audit/report.json).

Se corrigieron fallos relacionados comprobables: la cola diferida faltaba en los checkpoints y no se limpiaba al iniciar otro intento; los controles de aprendizaje podían ignorarse al reanudar; volver de Arena conservaba la actividad del tablero pausado pese a compartir pesos actualizados por otro simulador; y una carga fallida podía dejar un entrenador sin sesión operativa. Las pruebas nuevas verifican la limpieza temporal sin perder pesos, el orden de partidas congeladas, la restauración de recompensas pendientes, la aplicación de interruptores y el retorno desde Arena conservando el tablero.

La suite completa tras esta revisión terminó con 84 pruebas aprobadas, incluidas las pruebas nativas de escritorio con cambios de los interruptores de aprendizaje durante una sesión pausada.

## Choque inminente y manzana

El protocolo 2 comienza con un choque en el siguiente movimiento recto. Las 24 combinaciones distribuyen seis posiciones relativas de manzana entre cuatro orientaciones. Solo esquivar y recoger la manzana completa el ejercicio. Chocar o agotar los movimientos lo termina con castigo. El premio se entrega antes del reinicio. Las pruebas verifican el castigo en el siguiente paso recto en las 24 combinaciones, el éxito en el último paso permitido, la ausencia de premio por girar sin comer y los reinicios automáticos de un bloque.

La anticipación visual usa LC6 una casilla antes del borde, con un generador de pulsos separado del de comida y refuerzo. En diez semillas de calibración, con pesos congelados y hasta 120 movimientos, los choques pasaron de 4 a 1 y las manzanas de 134 a 160. [Calibración](../data/runs/boundary-vision-one-cell/report.json).

La comprobación posterior utilizó ocho semillas nuevas, el mismo modelo y hasta 100 movimientos. Los choques pasaron de 3 a 1 y las manzanas de 94 a 103. Estos resultados corresponden al cambio de entradas sensoriales con pesos congelados. No demuestran aprendizaje adicional ni ausencia de choques en otros escenarios.

Una auditoría independiente ejecutó 48 intentos autónomos del ejercicio de choque inminente. Se completaron 46 y se comprobaron todos los premios y castigos terminales. El candidato cambió cinco conexiones respecto a la memoria de origen, que ya estaba casi saturada. Los ensayos se guardaron en una sesión separada y no sustituyeron la memoria activa del usuario. [Semillas, resultados y auditoría de refuerzos](../data/runs/escape-apple-verification/report.json).

La suite completa terminó con 109 pruebas aprobadas. La comprobación de escritorio verificó tres inicios de choque inmediato, sus finales y la vuelta a partidas completas sin límite. También se comprobó que la actualización de la percepción de límites conserva los pesos y guarda una copia de la memoria anterior.
