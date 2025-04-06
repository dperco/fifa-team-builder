
# FIFA Team Builder API

📌 Descripción
API inteligente para crear equipos de fútbol óptimos basados en datos de FIFA 21, utilizando técnicas de IA y recomendación personalizada según criterios tácticos y de presupuesto.

🏗️ Arquitectura del Sistema
Diagrama de Componentes

┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   FastAPI App   │ ◄── │  Team Recommender   │ ◄── │  SentenceBERT    │
└─────────────────┘     └─────────────────────┘     └──────────────────┘
        ▲                       ▲                         ▲
        │                       │                         │
        ▼                       ▼                         ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  REST Endpoints  │     │  Recommendation     │     │   FAISS Index    │
│                 │     │  Engine             │     │                  │
└─────────────────┘     └─────────────────────┘     └──────────────────┘

Decisiones Técnicas

FastAPI: Elegido por su rendimiento, soporte nativo para async/await y generación automática de docs OpenAPI.

Sentence-BERT: Para embeddings semánticos que permiten entender descripciones textuales de equipos.

FAISS: Optimizado para búsqueda de similitud en espacios vectoriales de alta dimensión.

Pandas: Manipulación eficiente de los datos de jugadores.

Historial JSON: Solución liviana para tracking de solicitudes sin necesidad de DB.

🚀 Instalación
Requisitos Previos
Python 3.9+

pip 20+

Git

Pasos de Instalación

## Clonar el repositorio

git clone [https://github.com/tu-usuario/fifa-team-builder.git](https://github.com/tu-usuario/fifa-team-builder.git)

cd fifa-team-builder

# Crear el archvio de embeddings

 cd /app
 python initialize.py

## Crear entorno virtual (recomendado)

python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

## Instalar dependencias

pip install -r requirements.txt

***pueden existir problemas de compatibilidad de versiones de librerias , que hay   actualizar

## Descargar datos (ejemplo)

wget <https://example.com/players_21.csv> -O 'data/players_21.csv'
****   se usa el archivo jugadores_fifa21 de  Kagle ****

🏃 Ejecución

* Servidor de Desarrollo

uvicorn app.main:app --reload

* Servidor en Producción
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app

📊 Endpoints Principales

/api/teams/generate - POST -Genera un nuevo equipo

Como  posible mejora se podria agregar:

/api/teams/history-GET-Obtiene historial de equipos
/api/teams/last-team-GET-Obtiene el último equipo generado

Ejemplo de Request:

POST /api/teams/generate
Headers: {"X-User-ID": "user123"}
Body:
{
    "team_description": "equipo ofensivo con presión alta",
    "team_formation": "4-4-2",
    "budget": 100000000,
    "criteria": {
        "GK": {"min_overall": 80},
        "DEF": {"min_pace": 70, "min_defending": 75},
        "MID": {"min_passing": 75},
        "ATT": {"min_shooting": 80}
    }
}

Ejemplo Response :

{
    "formation": "4-4-2",
    "description": "equipo ofensivo con presión alta",
    "players": [
        {
            "id": 167495,
            "name": "M. Neuer",
            "position": "GK",
            "overall": 90,
            "value": 20500000.0,
            "age": 25,
            "nationality": "Germany",
            "selection_reason": "Mejor portero disponible (GK Score: 68.33) con 90 de overall"
        },
        {
            "id": 155862,
            "name": "Sergio Ramos",
            "position": "CB",
            "overall": 89,
            "value": 33500000.0,
            "age": 25,
            "nationality": "Spain",
            "selection_reason": "Central (CB Score: 95.11) con buen potencial y habilidades defensivas"
        },
        {
            "id": 208333,
            "name": "E. Can",
            "position": "CB",
            "overall": 82,
            "value": 35500000.0,
            "age": 25,
            "nationality": "Germany",
            "selection_reason": "Central (CB Score: 93.44) con buen potencial y habilidades defensivas"
        },
        {
            "id": 215071,
            "name": "M. Casco",
            "position": "FB",
            "overall": 78,
            "value": 9500000.0,
            "age": 25,
            "nationality": "Argentina",
            "selection_reason": "Lateral (FB, FB Score: 76.10) con velocidad y habilidad ofensiva/defensiva"
        },
        {
            "id": 245308,
            "name": "M. Nérez",
            "position": "FB",
            "overall": 80,
            "value": 0.0,
            "age": 25,
            "nationality": "Uruguay",
            "selection_reason": "Lateral (FB, FB Score: 75.10) con velocidad y habilidad ofensiva/defensiva"
        },
        {
            "id": 164459,
            "name": "S. Larsson",
            "position": "CM",
            "overall": 72,
            "value": 950000.0,
            "age": 25,
            "nationality": "Sweden",
            "selection_reason": "Mediocentro (CM Score: 77.83) con equilibrio entre ataque y defensa"
        },
        {
            "id": 223058,
            "name": "D. Kuzyaev",
            "position": "CM",
            "overall": 75,
            "value": 0.0,
            "age": 25,
            "nationality": "Russia",
            "selection_reason": "Mediocentro (CM Score: 77.17) con equilibrio entre ataque y defensa"
        },
        {
            "id": 200094,
            "name": "M. Ozdoev",
            "position": "RM",
            "overall": 77,
            "value": 0.0,
            "age": 25,
            "nationality": "Russia",
            "selection_reason": "Mediocentro defensivo (CAM/CDM Score: 74.13) con habilidades completas"
        },
        {
            "id": 245300,
            "name": "M. Baldona",
            "position": "LM",
            "overall": 80,
            "value": 0.0,
            "age": 25,
            "nationality": "Uruguay",
            "selection_reason": "Mediocentro defensivo (CAM/CDM Score: 73.87) con habilidades completas"
        },
        {
            "id": 184200,
            "name": "M. Arnautović",
            "position": "ST",
            "overall": 81,
            "value": 0.0,
            "age": 25,
            "nationality": "Austria",
            "selection_reason": "Delantero centro (ST Score: 76.23) con habilidades ofensivas completas"
        },
        {
            "id": 245315,
            "name": "E. Aguerro",
            "position": "ST",
            "overall": 78,
            "value": 0.0,
            "age": 25,
            "nationality": "Uruguay",
            "selection_reason": "Delantero centro (ST Score: 73.15) con habilidades ofensivas completas"
        }
    ],
    "total_value": 99950000.0,
    "avg_rating": 80.18,
    "team_analysis": "Análisis del equipo:\n- Portero: M. Neuer\n- Defensas (4): Sergio Ramos, E. Can, M. Casco, M. Nérez\n- Mediocampistas (4): S. Larsson, D. Kuzyaev, M. Ozdoev, M. Baldona\n- Atacantes (2): M. Arnautović, E. Aguerro\n\nFortalezas del equipo: Portería sólida, Defensa consistente, Ataque peligroso\n"
}

🧪 Pruebas Unitarias
Ejecutar todas las pruebas:
pytest --cov=app tests/

Cobertura actual:

Recommendation Engine: 95%
API Endpoints: 90%

💰 Estimación de Costos

Escenario Base (AWS EC2 t3.medium):
Recurso Costo Mensual
Instancia EC2    ~$30
Almacenamiento (50GB)    ~$5
Transferencia de datos (100GB)    ~$9
Total estimado    ~$44/mes

Costos por Operación:

Generación de equipo: ~0.002 CPU-minutos

Búsqueda en historial: ~0.0001 CPU-minutos

Carga inicial: ~1 CPU-minuto (procesamiento de datos)

🧩 Estructura del Proyecto

fifa-team-builder/
├── app/
│   ├── ai_assistant
        |____init__.py                  # Expone clases clave del módulo
        |__ chat_processor.py           # Capa de procesamiento de lenguaje natural
        |__intent_detection.py          # Motor de clasificación de intenciones
│   │   ├── recommendation_engine.py    # Lógica central
│   ├── routers/
        |___init__.py             # Organiza endpoints
        |__ chat.py               # Interfaz de conversación principal
│   │   └── teams.py              # Endpoints API
│   ├── services/
        |__init__.py             # Configura utilidades
        |__data__procesing.py    # ETL de datos de jugadores
│   │   ├── history_manager.py   # Gestión de historial
│   │   └── embeddings.py        # Procesamiento de embeddings
    |__init__.py                 # Inicializa el paquete principal
    |__initialize.py             # Genera el archivo Embeddings
    |__main.py                   # Punto de entrada del sistema
├── data/
│   └── players_21.csv           # Datos de jugadores
|___models/
    |__embeddings.faiss          # Almacenamiento vectorial
├── tests/
│   ├── test_recommendation_engine.py   # test unitarios criterios equipos
│   └── test_routers.py                 # test unitarios endpoints
├── requirements.txt             # Especificación de dependencias
|___ config.sys                  # Gestión centralizada de configuraciones
|___ .gitignore
└── README.m

🛠️ Extensibilidad
El sistema está diseñado para:

Añadir nuevas formaciones: Modificar _get_*_positions() en recommendation_engine.py

Integrar nueva lógica de selección: Añadir métodos _select_* personalizados

Cambiar almacenamiento de historial: Implementar nueva clase que herede de HistoryManager

Escalar horizontalmente: Diseño stateless permite múltiples instancias

############################################## Resumen Aplicacion ###########################

Resumen de la Lógica de Selección de Jugadores y Equipos

1. Flujo General

Entrada del Usuario:

Descripción textual del equipo (ej: "equipo ofensivo con defensa sólida").

Formación táctica (ej: 4-3-3, 4-4-2).

Presupuesto y criterios por posición (ej: DEF: min_pace=70).

Procesamiento:

Embeddings de texto: Se convierte la descripción en un vector numérico usando Sentence-BERT.

Filtrado por posición: Se seleccionan jugadores según la formación (CB, CM, ST, etc.).

Criterios personalizados: Se aplican umbrales (ej: min_pace=70).

Selección Optimizada:

Score por posición: Cada jugador recibe un puntaje basado en atributos relevantes.

Ejemplo para defensores (CB):

CB_Score = (DefendingTotal + PhysicalityTotal + Height + Jumping) / 4
Ejemplo para delanteros (ST):

ST_Score = (ShootingTotal + SprintSpeed + Agility + Positioning) / 4
Búsqueda FAISS: Se comparan los embeddings del equipo deseado con los de los jugadores filtrados.

Restricción de presupuesto: Se priorizan jugadores con mejor relación calidad-precio (Overall / ValueEUR).

Formación del Equipo:

Se seleccionan los mejores jugadores por posición hasta completar la formación.

Ejemplo para 4-3-3:

1 Portero (GK)

4 Defensores (2 CB + 2 Laterales)

3 Mediocampistas (CM + CAM + CDM)

3 Atacantes (ST + LW + RW)

Salida:

Equipo optimizado con:

Jugadores seleccionados.

Valor total y promedio de overall.

Explicación de selección por jugador (ej: "Van Dijk: Defensa sólida (CB_Score: 92.5)").

1. Detalles Clave por Posición
Posición            Atributos Clave                                    Lógica de Selección
GK           GK_Score = (Reflexes + Handling + Positioning)             3    Mejor puntuación en habilidades de portero.
CB           Defensa + Físico + Altura                              Ideal para detener ataques y jugar aéreos.
RB/LB       Resistencia + Velocidad + Centros                      Laterales rápidos para apoyar en ataque y defensa.
CM           Pases + Resistencia + Visión                              Mediocentros equilibrados para distribuir juego.
CAM/CDM    Pases + Defensa/Atq + Agilidad                              Especializados en creación o recuperación de balón.
ST           Remate + Aceleración + Posicionamiento                  Máxima efectividad en área rival.
LW/RW       Regate + Velocidad + Centros                              Extremos habilidosos para desbordar y asistir.

1. Ejemplo de Código (Pseudocódigo)

def generar_equipo(descripción, formación, presupuesto):
    # 1. Convertir descripción a embedding
    embedding_equipo = model_BERT.encode(descripción)
    # 2. Filtrar jugadores por posición y atributos
    jugadores_filtrados = []
    for posición in obtener_posiciones(formación):
        jugadores_posición = df[
            (df["BestPosition"] == posición) &
            (df["ValueEUR"] <= presupuesto)
        ]
        jugadores_filtrados.extend(calcular_scores(jugadores_posición, posición))
    # 3. Ordenar por similitud al equipo deseado (FAISS)
    jugadores_ordenados = FAISS_search(embedding_equipo, jugadores_filtrados)
     # 4. Seleccionar equipo final
    equipo_final = []
    for posición, cantidad in formación.items():
        mejores = jugadores_ordenados[posición].head(cantidad)
        equipo_final.extend(mejores)
        presupuesto -= sum(jugador["ValueEUR"] for jugador in mejores)
    return equipo_final

1. ¿Por qué esta Lógica?
✅ Eficiencia: FAISS permite búsqueda rápida en millones de jugadores.
✅ Personalización: Los scores por posición reflejan roles tácticos reales.
✅ Transparencia: Cada selección incluye una explicación basada en datos.

1. Posibles Mejoras
Aprendizaje Automático: Usar un modelo que aprenda de selecciones previas.

Restricciones Dinámicas: Ajustar automáticamente el presupuesto por posición.
