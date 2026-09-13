# On-Orbit Refuelling Mission Planning for Multiple GEO Satellites Using a Single Servicing Spacecraft

Proyecto de planificación de una nave de servicio que visita varios
satélites objetivo en órbitas circulares. El objetivo es minimizar el
`delta-v` total y estudiar cómo la secuencia, el tiempo disponible y la
masa de la nave afectan la misión.

## Modelo actual

El punto de entrada principal es `scripts/run_tsa_demo.py`. Implementa el modelo TSA
(Total Sweep Angle) de Shen y Tsiotras (2002) — específicamente el
**Problem 2** del paper (el servicio no regresa a su posición inicial;
el **Problem 1**, con retorno al depot, no está implementado):

- El servicio parte del `DEPOT` y visita cada target una vez.
- No hay retorno al depot.
- Todos los cuerpos comparten la misma órbita circular: radio,
  inclinación y RAAN.
- Los satélites difieren únicamente en su fase orbital.
- La misión tiene un tiempo total `tf` compartido entre todos los tramos.
- Opcionalmente, se descuenta un **tiempo fijo de acoplamiento/entrega**
  (`dock_time_s`, igual en cada parada — no está en el paper, es una
  extensión propia) del `tf` total ANTES de repartir tiempo entre los
  tramos de transferencia: `tf_transferencias = tf - n_paradas ×
  dock_time_s`. No consume combustible (no es una maniobra), solo tiempo.
- Cada tramo se evalúa mediante Lambert y el tiempo se distribuye con
  un programa entero discretizado (`scipy.optimize.milp`).
- La masa se calcula con la ecuación de Tsiolkovsky, con dos tanques
  separados (propelente de maniobra y carga a entregar) y chequeo de
  factibilidad — ver `orbital_refuel/mass_dynamics.py`.

## Supuestos de la simulación

- Tierra esférica y parámetro gravitacional constante:
  `mu = 398600.4418 km^3/s^2`.
- Órbitas circulares (`e = 0`) y dinámica de dos cuerpos.
- No se modelan perturbaciones, arrastre, maniobras de estación ni
  ventanas de lanzamiento.
- Lambert incluye soluciones multi-revolución (`lambert.find_all_lambert_solutions`,
  validado a precisión de máquina contra `izzo2015` para M=0,1,2). La
  curva de costo de cada tramo (`rdvsq_planning_model.cost_curve`) usa
  esas soluciones más un mínimo acumulado sobre el tof, reproduciendo
  el "coasting" del paper — la curva resultante sí tiene la estructura
  real de rampa-meseta que describe la Fig. 1/2 de las referencias.
- El coste de un tramo es el `delta-v` de la transferencia; el tiempo de
  acoplamiento (si se usa `dock_time_s`) es una pausa fija sin costo de
  combustible, no una operación modelada en detalle.
- **La distribución de tiempo tiene dos modos**: `time_method="grid"`
  (por defecto) discretiza la curva de costo en una malla de tiempos;
  `time_method="exact"` usa los breakpoints β/A exactos (mínimos locales
  de cada rama de Lambert vía optimización 1D) y las dos restricciones
  de tiempo del paper (ecs. 16b/16c), sin error de discretización — más
  lento, nunca peor. Ver Limitaciones para un caso de comparación.
- En el perfil de masa se usa un `Isp` fijo y se aplica Tsiolkovsky
  tramo a tramo.
- El modelo TSA es específico para la constelación de una sola órbita;
  no representa directamente una constelación con radios o planos
  orbitales diferentes.

## Resultados reproducibles

Al ejecutar `python scripts/run_tsa_demo.py` con los casos incluidos se obtiene:

- Secuencia óptima para una constelación equiespaciada de 6 targets:
  `[1, 2, 3, 4, 5, 6]`.
- Perfil de masa con `m0 = 500 kg` e `Isp = 300 s`:
  `500.00 kg -> 248.61 kg`.
- Propelente consumido en la demostración: `251.39 kg`.
- Con `n = 5`, la validación por fuerza bruta muestra que TSA no es
  óptimo para presupuestos cortos y coincide con el óptimo probado desde
  aproximadamente `4.5` periodos totales, es decir, `0.9` periodos por
  tramo en ese experimento.
- La figura de esa comparación se guarda en
  `outputs/figures/tsa_validity_regime.png`.

Estos valores dependen de los parámetros, la semilla y la discretización
utilizada por el programa entero.

## Instalación

Requiere Python 3.10 o posterior.

```bash
cd orbital_refuel_scheduler
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -e .
```

`poliastro` y `astropy` **no son necesarios** para `main.py` ni para el
resto del proyecto — solo los usa `validation/compare_lambert.py` para
comparar contra una implementación externa de Lambert. Viven en un
archivo de dependencias separado; instalar solo si se quiere correr ese
script en particular:

```bash
pip install -r requirements-validation.txt
```

## Ejecución

Demostración principal (secuencia óptima, perfil de masa, validación TSA):

```bash
python scripts/run_tsa_demo.py
```

Validaciones disponibles (cada una imprime sus propios resultados y
asserts; no requieren `main.py` corrido antes):

| Script | Qué valida |
...


### Visualización interactiva de la misión

```bash
python validation/animate_mission_html.py
```

Genera `validation/mission_visualization.html`, un archivo autocontenido
(sin dependencias externas ni servidor) que anima la secuencia óptima
resuelta por `main.py`: los satélites moviéndose según Kepler, el
servicio recorriendo los arcos de Lambert reales entre un objetivo y el
siguiente (coloreados como pendiente/visitando/visitado), una pausa real
de acoplamiento en cada parada (`DOCK_TIME_H`, un anillo alrededor del
marcador indica cuándo está acoplado en vez de en tránsito), y dos
medidores en vivo — propelente de maniobra y carga a entregar —
calculados con `mass_dynamics.simulate_mission_mass`. Se abre
directamente en el navegador (doble clic o `open`/`xdg-open`/`start`).

Como en este modelo todos los cuerpos comparten la misma órbita
(inclinación y RAAN nulos para todos), la vista es puramente plana —
no hace falta una segunda cámara ni una vista 3D.

## Estructura principal

- `src/orbital_refuel/keplerian.py`: propagación de órbitas circulares.
- `src/orbital_refuel/lambert.py`: solver Lambert de cero revoluciones.
- `src/orbital_refuel/transfer.py`: coste y optimización de un tramo.
- `src/orbital_refuel/mission.py`: generación de la constelación de una sola órbita.
- `src/orbital_refuel/rdvsq_planning_model.py`: TSA, distribución de tiempo
  y perfil de masa.
- `src/orbital_refuel/mass_dynamics.py`: dinámica explícita de propelente y
  carga con chequeo de factibilidad (dos tanques, sin simplificar).
- `scripts/`: ejecuciones principales y demos manuales.
- `validation/`: scripts de validación independientes y visualizaciones
  (ver tabla arriba).
- `tests/`: pruebas automáticas rápidas.
- `outputs/`: figuras, videos y reportes generados.

## Limitaciones y trabajo futuro

## Details 

