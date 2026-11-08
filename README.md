# FlyBrain Snake

FlyBrain Snake es el laboratorio de escritorio de **Brian**, un modelo del cerebro de una mosca de la fruta. Brian juega a Snake mediante la actividad simulada de FlyWire FAFB v783. El control usa la diferencia de disparos entre neuronas descendentes de izquierda y derecha. El aprendizaje modifica un conjunto anatómicamente delimitado de conexiones KC→MBON.

Las manzanas de **Brian** aparecen únicamente en las casillas interiores, en Arena y en todas las prácticas de entrenamiento. El jugador humano de Arena puede recibir manzanas también en el perímetro. Ambos tableros comparten la semilla base, pero sus posiciones de comida pueden ser distintas. Al cargar una sesión de Brian con una manzana en el borde, se sustituye por una interior conservando los pesos.

## Abrir en Windows

Haz doble clic en **Brian.cmd**. En **Arena**, pulsa **Jugar**. Brian juega en el tablero izquierdo y las flechas mueven tu serpiente en el derecho. El duelo comienza con la primera flecha. **Tablero** ajusta ambos lados entre 4 y 100 casillas para la próxima partida. El tamaño inicial es 11×11. Los dos tableros comparten una semilla nueva en cada duelo y avanzan al mismo ritmo. Al entrenar o cargar una sesión, Arena adopta su tamaño de tablero como punto de partida. Puedes cambiarlo para un duelo. El cerebro central muestra una muestra de contactos de FlyWire y los disparos de cada turno con círculos que se expanden y se apagan. El primero en alcanzar la **Meta** gana, igual que quien sobrevive a un choque del rival contra la pared. Dos choques o dos metas en el mismo turno dan empate. La meta inicial es 50 manzanas. Puedes ajustarla para el próximo duelo hasta la capacidad del tablero. El duelo dura todos los turnos necesarios. Brian usa y actualiza la misma memoria activa en **Arena** y **Entrenamiento**. Se guarda entre aperturas. Para importar los pesos de una sesión anterior, usa **Sesión…**.

En **Entrenamiento**, elige una etapa y pulsa **Nuevo bloque**. En la etapa 4, **Práctica → Rectas y giros** alterna rectas en los cuatro bordes, rectas interiores y giros frente a una pared. Cada episodio termina al recoger su manzana, chocar o alcanzar el límite de movimientos. Cambiar **Práctica** actualiza la sesión activa y empieza un intento del modo elegido conservando los pesos. **Reanudar** respeta esa selección y crea un bloque si todavía no hay una sesión. Al elegir **Partidas completas**, el límite empieza en **Sin límite**. Puedes fijar un límite opcional y modificarlo durante la partida. Al cargar una sesión de partidas completas se descarta el límite heredado y se conservan los pesos. En esta práctica se entrena la coincidencia de actividad KC y dopamina sin exigir un disparo de la MBON. Conserva los pesos para volver después a **Partidas completas**. Usa **Pausa** y **Un paso** para inspeccionar una decisión. **Manual** en esa pestaña permite mover el tablero de entrenamiento. **Reanudar** devuelve ese tablero a Brian.

**Reiniciar aprendizaje** devuelve la memoria activa a los pesos iniciales y conserva las sesiones guardadas. **Recompensas** ajusta los valores del siguiente bloque de entrenamiento. Arena toma esos valores del perfil activo, o directamente del panel si aún no hay una sesión. Brian puede recibir un premio de ruta cada tres movimientos, o antes si come. Se le castiga una sola vez por manzana, en el segundo paso que se desvía del camino legal más corto. Si la manzana está en el interior, se castiga tocar el perímetro y acercarse sin necesidad a sus tres casillas próximas. Una manzana en el perímetro desactiva ese castigo y da dos movimientos de salida al recogerla. Un tablero se completa al comer tantas manzanas como casillas tiene, y la última manzana va acompañada de un refuerzo mayor.

En **Práctica → Evitar paredes**, Brian siempre empieza mirando hacia fuera desde la última casilla: el siguiente movimiento recto choca. La manzana aparece a un lado, en diagonal hacia el interior o detrás, con variantes en las cuatro orientaciones. Tiene que esquivar y recoger esa manzana. Entonces recibe el premio de manzana, más **Evasión** —30 + 18 por defecto— y comienza otro intento. Girar sin comer no termina el ejercicio ni entrega ese premio especial. Chocar o agotar el límite sin completar el objetivo termina el intento con castigo —−60 por defecto— y vuelve a empezar hasta completar el número de **Episodios** elegido.

Los dos primeros movimientos tienen margen para salir del borde impuesto por el ejercicio. El choque sigue castigándose durante ese margen. Los castigos habituales de partidas completas mantienen sus reglas. El premio conjunto tiene un pulso dopaminérgico mayor que una manzana ordinaria para que el refuerzo adicional también se distinga dentro del simulador.

**Práctica → Perímetro estricto** empieza una partida normal en el centro, con manzanas interiores y sin límite de movimientos por defecto. En cuanto la cabeza entra en cualquier fila o columna exterior, el intento termina con un castigo del doble de **Colisión** —−120 por defecto— y una señal aversiva reforzada. El castigo se aplica antes del reinicio y se conservan los pesos aprendidos. El bloque repite hasta completar los **Episodios** elegidos. La gráfica **Finales por borde** distingue estos finales de un choque real. Esta regla pertenece a esa práctica; Arena y **Partidas completas** mantienen su funcionamiento habitual. El preset de consola es `configs/perimetro.json`.

Brian recibe señales de las manzanas y de los límites en partidas completas, Arena y entrenamiento. La señal frontal de pared se anticipa una casilla mediante las poblaciones visuales LC6 ya identificadas; al estar sobre el borde conserva la alerta de choque existente. Las señales laterales y las etiquetas KC de distancia también se mantienen. **Telemetría → Límites percibidos** muestra las casillas libres frente a Brian y a cada lado cuando tomó la decisión. Los giros los decide la lectura neuronal habitual.

La práctica de paredes valida los pesos con 24 escenarios reservados antes de empezar, cada 24 episodios y al completar el bloque. Solo se acepta una mejora si no aumenta los choques, tampoco al comparar cada escenario por separado. Un candidato debe conservar las manzanas, supervivencia y ausencia de nuevos choques en seis partidas ordinarias de hasta 64 movimientos. Se puede conservar una mejora de partidas normales aunque el ejercicio ya estuviera dominado, siempre que mantenga sus resultados. Esos límites corresponden a las pruebas y no se aplican a Arena. Los pesos quedan congelados al evaluar. La validación se ejecuta en segundo plano y puede tardar más de un minuto si llega a comprobar las partidas completas.

Los pesos en entrenamiento están separados de la memoria activa. Guardar, pausar, cerrar o iniciar otro bloque conserva en la memoria activa el último modelo aceptado. Al pasar a Arena o cambiar de práctica se guarda el candidato pendiente y se usa el modelo aceptado. Cargar una sesión de paredes recupera ese modelo desde **paredes-mejor** y permite seguir entrenando el candidato en privado. La gráfica **Evasiones correctas** muestra los ejercicios de entrenamiento. **Actualizaciones por episodio** muestra los cambios sinápticos de ese intento, a diferencia del total histórico de conexiones modificadas. Los candidatos y las decisiones quedan registrados en la sesión.

La señal auxiliar de pared de la versión anterior causaba una regresión en partidas completas y se ha desactivado en la aplicación. Al abrir una memoria que la conservaba se crea una copia **memoria-antes-retirar-senal-pared-….npz**, se retira la señal y se mantienen exactamente sus pesos. La práctica utiliza la regla local disponible de coincidencia KC y dopamina sin exigir un disparo postsináptico, para evitar bloquear el aprendizaje cuando las MBON permanecen silenciosas. La regla actual solo deprime conexiones hasta un mínimo. Una memoria con las conexiones estimuladas ya en ese mínimo tiene poco margen de aprendizaje adicional. Este protocolo todavía no demuestra aprendizaje perfecto ni garantiza ausencia de choques. Consulta los [resultados de paredes](docs/resultados.md#corrección-de-la-regresión-en-paredes).

La nueva anticipación visual utiliza un generador de estímulos independiente del de comida y refuerzo, conservado en los checkpoints. Al actualizar la memoria del escritorio se guarda **memoria-antes-vision-limites-….npz** sin modificar sus pesos. Los perfiles históricos guardados siguen usando `wall_preview_cells=0` al reproducirlos por consola; el escritorio los actualiza a 1.

Para una instalación nueva, abre PowerShell en la carpeta del proyecto y ejecuta lo siguiente.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Test
.\Brian.cmd
```

El instalador local utiliza Python 3.12.11 x64 y crea `.venv`. Descarga unos 885 MB de datos oficiales y deja archivos procesados en `data/processed`. Reserva al menos 3 GB de disco para datos, dependencias y primeros ensayos. Una descarga cortada vuelve a intentarse con rangos de bytes y cada archivo se verifica antes de usarlo. El motor se ejecuta en CPU en Windows 11.

## Protocolos

| Etapa | Protocolo |
| --- | --- |
| 0 | Propagación, estabilidad, salida descendente, controles de asociación y retención en checkpoint |
| 1 | Condicionamiento del patrón KC-A con PAM01 o PPL101. El patrón KC-B actúa como control en la validación |
| 2 | Búsqueda de comida en un toroide, con una sola celda de serpiente |
| 3 | Búsqueda con paredes terminales y castigo gradual |
| 4 | Snake con crecimiento, manzanas sucesivas y paredes terminales |

El tablero empieza en 7×7. La cabeza puede atravesar el cuerpo y el motor del juego bloquea las inversiones de 180°. El cronómetro científico separa el tiempo simulado del reloj de pared. Un límite de pasos de 0 permite continuar hasta una condición terminal.

Las partidas consecutivas conservan los pesos y usan semillas independientes derivadas de la semilla base y el número de episodio. Al empezar se limpian voltajes, impulsos, trazas de elegibilidad, contadores de actividad y recompensas pendientes del intento anterior. Una partida buena puede ir seguida de una corta porque cambian la comida y los estímulos aleatorios. La [revisión de partidas consecutivas](docs/resultados.md#partidas-consecutivas) distingue esa variación del efecto de aprender.

Los interruptores **Plasticidad habilitada** y **Refuerzo dopaminérgico** se aplican al siguiente movimiento, también en **Un paso**, **Reanudar** y un duelo ya abierto. Al volver de Arena se conserva el tablero de entrenamiento pausado y los pesos actuales, y se limpia la actividad neuronal anterior. Ese episodio interrumpido queda marcado como interactivo y el cambio se registra en la sesión. Guardar y cargar dentro de una partida también conserva la cola experimental de recompensas diferidas; comenzar otra partida la vacía.

En **Cerebro 3D**, arrastra para rotar, usa el botón derecho para desplazar y la rueda para acercarte. El fondo usa las coordenadas reales de FlyWire. Selecciona una neurona o escribe su root ID para ver sus contactos pre y postsinápticos. **Memoria** permite comparar un checkpoint real con pesos iniciales en el visor. **Experimentos** ejecuta A–I en segundo plano y muestra la ruta del informe. Puedes cargar y exportar sesiones SQLite desde la aplicación. Las acciones manuales llevan una etiqueta propia en las trazas.

## Anatomía de FlyWire v783

Abre **Anatomía** para explorar los contactos individuales en 3D. Filtra por neuropilo, neurotransmisor predicho, revisión, neurona, pareja o sentido. **Mostrar contactos** distribuye una muestra de hasta 6.000 puntos y **Páginas** recorre todas las coincidencias. Pulsa un punto o usa **Localizar** para ver el ID, los dos extremos medidos, los valores de confianza y las seis probabilidades del contacto. Al buscar por neurona también aparecen sus entradas y salidas por neuropilo, según los recuentos publicados. El color puede representar el neurotransmisor predicho o qué extremos están revisados.

El atlas lee los cinco archivos oficiales de [FlyWire v783](https://zenodo.org/records/10676866). Conserva los Feather originales para consultar el detalle completo y genera un índice local de 5,43 GB en `data/anatomy`. En una instalación nueva con el conectoma ya preparado, puedes reconstruirlo así.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m flybrain_snake anatomy-prepare --source .
& $py -m flybrain_snake anatomy-show --id 2021
```

**Arena** requiere el atlas preparado. Usa la transmisión, la estimulación y las recompensas del perfil entrenado. Antes de entrenar o cargar una sesión usa la transmisión por conexión. En **Entrenamiento → Estimulación → Transmisión** puedes elegir **Por conexión** para un bloque nuevo. El preset de consola es `--config configs/contactos.json`. El modo aplica a cada grupo dirigido neurona–neurona–neuropilo la contribución rápida esperada `P(ACh) − P(GABA) − P(Glu)` a partir de las probabilidades medias publicadas. Las asignaciones rápidas curadas tienen prioridad. Es una hipótesis de simulación y su rendimiento en Snake se mide por separado.

La simulación usa el grafo de neuronas revisadas. Los contactos ligados a segmentos sin revisar aparecen en el atlas con su estado correspondiente. Las coordenadas pertenecen a los extremos de los contactos. Para simular propagación dentro de cada rama harían falta datos adicionales de morfología y propiedades eléctricas.

## Línea de comandos

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m flybrain_snake doctor
& $py -m flybrain_snake prepare --local-only
& $py -m flybrain_snake validate --trials 12
& $py -m flybrain_snake benchmark --repetitions 10
& $py -m flybrain_snake train data/sessions/mi_brian.sqlite --episodes 20
& $py -m flybrain_snake train data/sessions/mi_rafaga.sqlite --burst
& $py -m flybrain_snake train data/sessions/mi_brian.sqlite --resume --episodes 20
& $py -m flybrain_snake evaluate --session data/sessions/mi_brian.sqlite --episodes 10
& $py -m flybrain_snake study data/runs/mi_estudio --train 8 --test 10
& $py -m flybrain_snake --config configs/paredes.json wall-study data/runs/mi_estudio_paredes --train 96 --test 48
& $py -m flybrain_snake export data/sessions/mi_brian.sqlite data/runs/mi_exportacion
& $py -m pytest -q
```

Para cambiar de etapa, usa `train --resume --stage N` entre episodios. `--reset-memory` inicia la etapa con los pesos base. `Ctrl+C` en el entrenamiento guarda al terminar el movimiento en curso. Un cierre abrupto deja disponible el último checkpoint confirmado. `compare` evalúa pesos iniciales y varios checkpoints con las mismas semillas. `memory` exporta los pesos y sus root IDs.

```powershell
& $py -m flybrain_snake compare data/sessions/Brian_v783_1000.sqlite data/runs/mi_comparacion --labels etapa4-episodio100 etapa4-episodio1000
& $py -m flybrain_snake gui --session data/sessions/Brian_v783_1000.sqlite
```

`--config configs/default.json` se coloca antes del subcomando y admite parámetros de LIF, plasticidad, recompensa y juego. `configs/milestones.json` describe la corrida histórica de hitos con el perfil de recompensas original. `--data RUTA` permite elegir otro directorio de datos.

`wall-study` compara el mismo cerebro inicial con aprendizaje, pesos congelados y dopamina omitida. Selecciona checkpoints con semillas de validación y mide el resultado en otro conjunto de prueba. `--session RUTA --label latest` usa los pesos de una sesión existente sin modificarla. El informe JSON incluye semillas, resultados por caso, cambios sinápticos e intervalo bootstrap emparejado.

## Resultados y método

- [Resultados medidos](docs/resultados.md)
- [Diseño, supuestos y procedencia](docs/diseno.md)
- [Protocolos, ecuaciones y límites de interpretación](docs/metodo.md)
- [Manifiesto del grafo instalado](data/processed/manifest.json)

Si faltan datos, ejecuta `prepare` sin `--local-only`. Si hay un error de integridad, vuelve a preparar en un directorio de datos nuevo con `--data`. Si una sesión figura como activa, ciérrala en la aplicación o deja que termine el proceso de entrenamiento. La primera ventana puede tardar varios segundos mientras Numba compila el kernel.
