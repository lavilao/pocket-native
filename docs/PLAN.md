# Plan de trabajo de Pocket Native

**Actualizado:** 16 de noviembre de 2026
**Rama estable:** `master` (`7687886`, master fusionado con upstream v0.6.0)
**Rama de integración:** la prueba de rebase quedó obsoleta; `merge-upstream-v0.6.0` ya está en master

## Objetivo

Integrar las mejoras recientes de Pocket Native y Zakuro sin perder el soporte ARM32 del fork, y conseguir que Fantasy Life y otros juegos recompilados funcionen bien en el Redmi 9. Mantener Rune Factory 4 como un problema conocido que se revisa con evidencia, sin dejar que bloquee el resto del trabajo.

## Estado actual

- `master` permanece limpio y en `fab8094`; ese commit se probó en el teléfono.
- La prueba de rebase está detenida al aplicar `ff10871` (12 de 25 commits), con conflictos en `engine/core/src/kernel/svc.rs`, `engine/gpu/shaders/raster.frag` y `engine/gpu/src/raster/gles.rs`. `master` no está dentro de ese rebase.
- Fantasy Life ya se recompiló para ARM32 y se copió a la carpeta pública de Downloads del teléfono. El paquete incluye `game.3ds`, `title.so`, fuentes del sistema y `dlc.cia`. El usuario lo importará manualmente.
- El CIA de Fantasy Life tiene el ID de DLC correspondiente al juego y dos contenidos sin la marca de cifrado de clave de título. Falta validar la estructura NCCH/RomFS con el cargador.
- El fork actual no monta DLC: sus consultas `am:app` contestan `TITLE_NOT_FOUND` y no hay un cargador ni una ruta del sistema de archivos para sus contenidos.
- Zakuro añadió el flujo de updates y DLC en `5d06c80` (7 de octubre de 2026). Agrupa los CIA junto al juego, coteja los IDs, conserva el archivo como imagen y expone los RomFS de cada contenido por su índice. El juego recompilado sigue siendo el programa base; el DLC se carga aparte.
- Rune Factory 4 volvió a caerse en `libusc.so` tras evitar una variante de shader. El filtro actual no resuelve el crash.
- El Redmi 9 expone Vulkan 1.1; el renderer Vulkan actual de Zakuro requiere Vulkan 1.3. Un presenter Vulkan por sí solo dejaría activa la compilación GLES que provoca el crash.

## Prioridades

### 1. Completar la evaluación del rebase en el worktree aislado

1. Resolver los tres conflictos actuales manteniendo las refactorizaciones upstream donde sean aplicables y portando las correcciones de shaders y GLES del fork.
   - En `raster.frag`, conservar el flag `in_place` como `(units[unit].z & 1u) != 0u`: la mitad alta de esa palabra contiene el LOD bias y no debe invertir la coordenada V. Mantener el bias en los seis muestreos incondicionales y aplicar la proyección de la unidad 0 antes de muestrear.
   - En `gles/sources.rs`, adaptar la sustitución HALF a la nueva firma de `sample_unit` con argumentos prefetched y conservar alta precisión en parámetros y retorno de `sample_st`, así como en `st0/st1/st2`. Verificar el GLSL generado en modo normal y HALF; compilar Rust no valida estas sustituciones.
2. Continuar el rebase commit por commit, preservando explícitamente la reserva de fastmem para ARM32, las rutas de lectura de ROM que evitan mapear archivos de más de 4 GiB de espacio virtual y la configuración/build Android de 32 bits.
3. Revisar cada cambio de memoria, recompiler y compilación ARM32 contra `master`. No reemplazar ni reescribir `master` durante esta evaluación.
4. Cuando el rebase termine, construir ambos artefactos ARM32 desde el mismo árbol: APK con `scripts/android/build.sh --32bit --test` y juego con `scripts/build.sh <ROM> --32bit`. Verificar la ABI del APK y de `title.so`.
5. Probar juntos el APK y el juego en una instalación y datos de prueba aislados, sin reemplazar la instalación estable. Verificar primero Fantasy Life sin DLC: arranque, ejecución real de `title.so`, reserva fastmem y ausencia de fallback al intérprete. Comparar FPS bajo las mismas condiciones y registrar revisión, renderer, modo de precisión y errores.
6. Integrar a `master` solo después de revisar el diff completo y registrar que conserva las capacidades ARM32 y las correcciones específicas del Redmi. La compilación por sí sola no satisface este criterio.

### 2. Portar soporte runtime para DLC CIA

**Estado (16 de noviembre): hecho en master, falta verificar en el juego.** El flujo completo de Zakuro está: los CIA junto al ROM se clasifican por id, el update más nuevo adjunta su código y exheader (el RomFS del juego propio sigue siendo el del cartucho, el del update se abre como `patch:/` y por el archivo NCCH con el id del update), `am:app` contesta `GetPatchTitleInfos` además de las consultas DLC, y `3dsrecomp build` acepta un CIA de update y nombra su biblioteca con el id del update (`0004000E<mitad baja>.so`); en Android, `update.so` en la carpeta del juego se copia a `code/` bajo ese nombre. Sin esa biblioteca el juego corre interpretado (el `title.so` del cartucho ya no coincide) y el log lo dice.

Pendiente solo lo de siempre: confirmar en el Redmi que Fantasy Life ve el DLC con el update instalado, y si su DLC exige una versión mínima del juego.

Los pasos originales de la portada quedaron así: el parser de encabezado/TMD con acceso por rangos (1-4) llegó en `c6402ad`; las consultas `am:app` (5) y la apertura por índice (6) con él; el soporte de updates — lo que faltaba del commit `5d06c80` de Zakuro, con `Update`/`attach_update`, el RomFS `patch:/`, `GetPatchTitleInfos` y el build de `3dsrecomp` desde un CIA de update (7) — llegó con la fusión de v0.6.0; la asociación automática por carpeta (8) es cómo funciona todo el flujo, sin pasos manuales.

**Criterio de terminado:** el juego puede consultar el DLC mediante `am:app` y abrir un archivo conocido por su índice de contenido; el log identifica el CIA y los contenidos cargados; Fantasy Life confirma dentro del juego que el DLC está disponible. Registrar la versión del juego y cualquier actualización requerida. Un CIA aceptado por el parser no satisface por sí solo este criterio.

### 3. Tratar Vulkan como una línea separada

1. No usar Vulkan como solución inmediata para el crash GLES del Redmi.
2. Después de estabilizar el rebase, comprobar si el GE8320 expone las extensiones y funciones necesarias para un renderer Vulkan 1.1.
3. Si el camino exige Vulkan 1.3, mantener GLES como ruta del Redmi y limitar el renderer Vulkan a dispositivos compatibles. Un cambio de presenter no cuenta como sustitución del renderer.

### 4. Retomar Rune Factory 4 de forma acotada

Después de estabilizar GLES y las variantes de shader tras el rebase, hacer una prueba controlada con captura de log y variante identificada antes de compilar. Si vuelve a caer en UniFlex, reducir el shader de forma reproducible y conservar el diagnóstico. Evitar más pruebas repetidas sin un cambio que ponga a prueba una hipótesis concreta.

## Orden recomendado

1. Corregir los bloqueantes de flags y precisión del shader; finalizar y revisar el rebase ARM32 únicamente en el worktree de prueba.
2. Construir APK y juego ARM32 desde el mismo árbol. Probar juntos en el Redmi, con instalación y datos aislados, Fantasy Life sin DLC; registrar FPS comparables, renderer, precisión, ejecución recompilada, reserva fastmem y errores.
3. Portar el soporte de DLC CIA con acceso por rangos y validación de ambas capas de cifrado. Confirmar los requisitos de actualización de Fantasy Life y verificar consultas, apertura de un archivo conocido y reconocimiento dentro del juego.
4. Hacer una sola prueba acotada de Rune Factory 4 con captura de log, variante identificada y los cambios GLES ya verificados.
5. Integrar solo después de la revisión completa y de registrar los resultados exigidos; no sustituir la evidencia del Redmi por una compilación correcta.
6. Investigar Vulkan 1.1 como mejora opcional, separada de la ruta de corrección GLES.

## Archivos y estado útil

- Estado detallado previo: `build/WORK_STATUS.md`.
- Rebase de prueba: `/home/luis/Proyectos/pocket-native-rebase-trial`.
- Zakuro upstream con el commit de soporte CIA/DLC: `/home/luis/Proyectos/zakuro-upstream`.
- Paquete Fantasy Life en el PC: `build/Pocket Native/pocket-native/games/0004000000113200/`.
- Paquete copiado al teléfono: `/storage/emulated/0/Download/Pocket Native/pocket-native/games/0004000000113200/`.
- Diagnósticos de Rune Factory 4: `build/diagnostics/rf4/`.

## Referencias de implementación

- Commit que incorporó updates y DLC en Zakuro: [`5d06c80`](https://github.com/fearkov/zakuro/commit/5d06c800ffb81f9ec708d6ef99934a692a01ed1e).
- Parser CIA y estructura de contenidos: [`cia.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro-fs/src/cia.rs) y cargador `Dlc` en [`zakuro-fs/src/lib.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro-fs/src/lib.rs).
- Asociación y carga del DLC: [`library.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro/src/library.rs) y [`loader.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro-core/src/loader.rs).
- Servicios de consultas y acceso a RomFS: [`misc.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro-core/src/services/misc.rs) y [`fs.rs`](https://github.com/fearkov/zakuro/blob/main/crates/zakuro-core/src/services/fs.rs).
