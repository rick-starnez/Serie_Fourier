# Simulador: Serie Trigonométrica vs Serie Compleja de Fourier
Aplicación de escritorio en Python para visualizar y comparar la reconstrucción de una señal mediante la Serie Trigonométrica de Fourier y la Serie Compleja de Fourier.

## Descripción

El simulador muestra dos columnas lado a lado:

- Columna izquierda: Serie Trigonométrica de Fourier
  - Señal original vs reconstrucción
  - Espectro de magnitud de los coeficientes a_k, b_k

- Columna derecha: Serie Compleja de Fourier
  - Señal original vs reconstrucción
  - Espectro de magnitud de los coeficientes c_k (solo k >= 0)

En ambas columnas se puede variar en tiempo real:

- El tipo de señal de entrada
- El número de muestras N
- El número de coeficientes K empleados en la reconstrucción

El objetivo didáctico es que el alumno observe cómo cambia la reconstrucción
al aumentar K (aparición del fenómeno de Gibbs, convergencia a la señal, etc.)
y que compruebe que ambas series producen el mismo resultado.

## Señales disponibles

- Cuadrada
- Seno
- Triangular
- Random (aleatoria con semilla fija para reproducibilidad)

## Fórmulas implementadas

Serie Trigonométrica de Fourier:

    x(t) ~ a0/2 + sum_{k=1}^{K} [ a_k*cos(2*pi*k*t) + b_k*sin(2*pi*k*t) ]

    a_k = (2/N) * sum_n x[n]*cos(2*pi*k*n/N)
    b_k = (2/N) * sum_n x[n]*sin(2*pi*k*n/N)

Serie Compleja de Fourier:

    x(t) ~ sum_{k=-K}^{K} c_k * exp(j*2*pi*k*t)

    c_k = (1/N) * sum_n x[n] * exp(-j*2*pi*k*n/N)

Los coeficientes c_k se calculan con la FFT de numpy (np.fft.fft(x)/N).

## Controles de la interfaz

- Combo "Señal": selecciona el tipo de señal de entrada.
- Spinbox "N muestras": cantidad de muestras del período (64 a 4096).
- Spinbox "Coeficientes K": número de armónicos usados (1 a 200).
- Botón "Reset zoom": restaura la vista completa de las gráficas.

## Zoom

- Zoom por región: mantener presionado el botón izquierdo del mouse sobre
  la gráfica de reconstrucción y arrastrar para dibujar un rectángulo.
  Al soltar, se hace zoom a esa región en X e Y.
- Reset individual: doble clic sobre la gráfica.
- Reset global: botón "Reset zoom" de la barra superior.

Funciona igual en macOS, Windows y Linux. No requiere teclas modificadoras.

## Requisitos

- Python 3.9 o superior
- numpy
- matplotlib
- tkinter (incluido en la instalación estándar de Python en Windows y macOS;
  en Linux puede requerir el paquete python3-tk del sistema)

Las versiones recomendadas están en el archivo requirements.txt.

## Instalación

1. Clonar el repositorio:

       git clone <URL_DEL_REPOSITORIO>
       cd <NOMBRE_DEL_REPOSITORIO>

2. (Recomendado) Crear y activar un entorno virtual:

       python -m venv venv

   En macOS/Linux:

       source venv/bin/activate

   En Windows:

       venv\Scripts\activate

3. Instalar dependencias:

       pip install -r requirements.txt

## Ejecución

    python series_fourier.py

Se abrirá una ventana con las dos columnas del simulador.

## Nota para Linux

Si al ejecutar aparece un error del tipo "No module named tkinter",
instalar el paquete del sistema correspondiente:

- Debian/Ubuntu:

      sudo apt install python3-tk

- Fedora:

      sudo dnf install python3-tkinter

- Arch:

      sudo pacman -S tk

## Estructura del proyecto

    .
    ├── series_fourier.py     # Aplicación principal
    ├── requirements.txt      # Dependencias de Python
    └── README.md             # Este archivo

## Uso didáctico sugerido

1. Seleccionar la señal "Cuadrada" y observar la reconstrucción con K pequeño.
2. Aumentar K progresivamente y notar:
   - Cómo la reconstrucción se acerca a la señal original.
   - El sobrepaso en los flancos (fenómeno de Gibbs) que no desaparece,
     solo se concentra.
3. Cambiar a "Seno" y "Triangular" para ver que la convergencia es mucho
   más rápida cuando la señal es suave.
4. Comparar la columna izquierda y la derecha: ambas reconstrucciones deben
   coincidir, mostrando la equivalencia entre las dos formas de la serie.
5. Usar el zoom sobre una zona de la señal para observar en detalle el
   comportamiento de la reconstrucción.

## Licencia

Uso libre con fines educativos.
