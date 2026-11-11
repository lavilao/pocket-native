# Plan de trabajo de Pocket Native

**Actualizado:** 11 de noviembre de 2026
**Rama estable:** `master` (`fab8094`)
**Rama de integración:** prueba de rebase en `/home/luis/Proyectos/pocket-native-rebase-trial`

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
2. Continuar el rebase commit por commit, preservando explícitamente la reserva de fastmem para ARM32, las rutas de lectura de ROM que evitan mapear archivos de más de 4 GiB de espacio virtual y la configuración/build Android de 32 bits.
3. Revisar cada cambio de memoria, recompiler y compilación ARM32 contra `master`. No reemplazar ni reescribir `master` durante esta evaluación.
4. Cuando el rebase termine, compilar para ARM32 y verificar primero el arranque y la recompilación de Fantasy Life en el teléfono. Comparar FPS y confirmar que no volvió al intérprete.
5. Integrar a `master` solo después de revisar el diff completo y confirmar que conserva las capacidades ARM32 y las correcciones específicas del Redmi.

### 2. Portar soporte runtime para DLC CIA

Basarse en la implementación de Zakuro, adaptándola a las estructuras actuales del fork:

1. Leer el encabezado CIA y el TMD; obtener ID, versión, flags, índices y offsets de contenido sin copiar todo el archivo a memoria.
2. Aceptar DLC solo si corresponde al juego por los 32 bits bajos del ID y si los contenidos que se van a leer están descifrados y contienen NCCH/RomFS válidos.
3. Pasar los CIA encontrados junto al juego al cargador del sistema y conservar sus imágenes durante la ejecución.
4. Implementar las consultas de `am:app` para lista, índices, versión e información de ticket que los juegos esperan.
5. Abrir cada contenido por su ID e índice desde el servicio de archivos NCCH/SDMC; devolver errores claros para archivos cifrados, incompletos o de otro juego.
6. Asociar automáticamente `dlc.cia` con el juego en el paquete de Fantasy Life y comprobar en el juego que sus contenidos estén visibles.

**Criterio de terminado:** el juego puede consultar el DLC y abrir sus archivos, el log muestra qué CIA/contenidos cargó, y Fantasy Life confirma dentro del juego que el DLC está disponible.

### 3. Tratar Vulkan como una línea separada

1. No usar Vulkan como solución inmediata para el crash GLES del Redmi.
2. Después de estabilizar el rebase, comprobar si el GE8320 expone las extensiones y funciones necesarias para un renderer Vulkan 1.1.
3. Si el camino exige Vulkan 1.3, mantener GLES como ruta del Redmi y limitar el renderer Vulkan a dispositivos compatibles. Un cambio de presenter no cuenta como sustitución del renderer.

### 4. Retomar Rune Factory 4 de forma acotada

Después de estabilizar GLES y las variantes de shader tras el rebase, hacer una prueba controlada con captura de log y variante identificada antes de compilar. Si vuelve a caer en UniFlex, reducir el shader de forma reproducible y conservar el diagnóstico. Evitar más pruebas repetidas sin un cambio que ponga a prueba una hipótesis concreta.

## Orden recomendado

1. Finalizar y revisar el rebase ARM32 en el worktree de prueba.
2. Probar en el Redmi la recompilación ARM32 de Fantasy Life y registrar FPS, renderer, estado de recompilación y errores.
3. Portar y verificar el soporte de DLC CIA en runtime.
4. Hacer una sola prueba de Rune Factory 4 con los cambios GLES ya integrados.
5. Investigar Vulkan 1.1 como mejora opcional, separada de la ruta de corrección GLES.

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
