# 🚨 HazardEx

Sistema multimodal para el análisis de contenido visual orientado a la detección de amenazas y situaciones de riesgo.

Este documento describe los pasos necesarios para instalar y ejecutar **HazardEx** en un entorno local, incluyendo la configuración del **backend**, **frontend**, almacenamiento con **MinIO** y los distintos modelos utilizados por el sistema.

---

## 📋 Requisitos previos

Antes de comenzar, asegúrate de disponer de los siguientes componentes:

* **GPU NVIDIA con soporte CUDA**. Se recomienda disponer de al menos **12 GB de VRAM**.

  * El sistema ha sido desarrollado y probado sobre una **NVIDIA GeForce RTX 4070**.
* **Python 3.11.9** y `venv`.
* **Node.js 24.19.0** y `npm`.
* **Docker**.
* **Ollama**, utilizado para la ejecución local de los modelos de lenguaje y visión.
* Aproximadamente **30 GB de espacio libre** para almacenar los modelos.

---

## 📥 Clonar el repositorio

```bash
git clone https://github.com/aramosgranodeoro/HazardEx
cd HazardEx
```

---

## ⚙️ Configuración del entorno

HazardEx utiliza variables de entorno para almacenar la configuración local y las credenciales necesarias para ejecutar la aplicación.

Es necesario crear un archivo `.env` tanto en:

```text
backend/.env
frontend/.env
```

> [!IMPORTANT]
> Los archivos `.env` no se incluyen en el repositorio para evitar publicar credenciales y configuraciones específicas del entorno.

---

### 🔧 Backend

Crea el archivo:

```text
backend/.env
```

con el siguiente contenido:

```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=<usuario>
MINIO_SECRET_KEY=<contrasena>
BUCKET_NAME=<nombre_bucket>

DOCUMENTS_FOLDER=<ruta_documentos>
INDEX_FOLDER=<ruta_chroma_db>

MODELO_EMBEDDINGS=intfloat/multilingual-e5-small

ADAPTERS_DIR=<ruta_adaptadores>
```

Las variables permiten configurar:

| Variable            | Descripción                                                 |
| ------------------- | ----------------------------------------------------------- |
| `MINIO_ENDPOINT`    | Dirección del servidor MinIO                                |
| `MINIO_ACCESS_KEY`  | Usuario de acceso a MinIO                                   |
| `MINIO_SECRET_KEY`  | Contraseña de acceso a MinIO                                |
| `BUCKET_NAME`       | Bucket utilizado para almacenar contenido multimedia        |
| `DOCUMENTS_FOLDER`  | Directorio con los documentos utilizados por el sistema RAG |
| `INDEX_FOLDER`      | Directorio donde se almacena la base de datos de ChromaDB   |
| `MODELO_EMBEDDINGS` | Modelo utilizado para generar embeddings                    |
| `ADAPTERS_DIR`      | Directorio donde se almacenan los adaptadores LoRA          |

---

### 🖥️ Frontend

Crea el archivo:

```text
frontend/.env
```

con la dirección del backend:

```env
VITE_BASE_URL=http://localhost:8000
```

---

## 🐍 Instalación del backend

Desde la raíz del proyecto:

```bash
cd backend

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

> [!NOTE]
> Los comandos anteriores utilizan la sintaxis de **Windows**.

---

## ⚛️ Instalación del frontend

Desde la raíz del proyecto:

```bash
cd frontend
npm install
```

---

## 🤖 Descarga y configuración de los modelos

HazardEx utiliza tanto modelos servidos mediante **Ollama** como modelos especializados almacenados localmente.

### Modelos de Ollama

Descarga los modelos necesarios mediante:

```bash
ollama pull llava:7b
ollama pull qwen3.5:latest
ollama pull internlm/interns1:mini-q8_0
ollama pull llama3.1:8b
```

Puedes comprobar los modelos instalados con:

```bash
ollama list
```

---

### 🧠 Modelos especializados y adaptadores LoRA

Los pesos de los modelos especializados y los adaptadores **LoRA** utilizados por HazardEx pueden descargarse desde Google Drive:

👉 [Descargar modelos y adaptadores de HazardEx](https://drive.google.com/drive/folders/1YBkf6Etv-CjJBao2oFSYbAwRhQgyyXYC?usp=sharing)

Una vez descargados:

1. Guarda los modelos en un directorio local.
2. Configura `ADAPTERS_DIR` en `backend/.env` con la ruta correspondiente.
3. Sitúa los pesos de los modelos **YOLO** en las rutas esperadas por el backend.

Por ejemplo:

```env
ADAPTERS_DIR=C:\modelos\HazardEx\adapters
```

---

## 🗄️ Puesta en marcha de MinIO

HazardEx utiliza **MinIO** como almacenamiento de objetos para guardar las imágenes, vídeos, miniaturas y resultados procesados por la aplicación.

Puedes iniciar MinIO mediante Docker.

### Windows

```bat
docker run -d ^
  --name minio ^
  -p 9000:9000 ^
  -p 9001:9001 ^
  -e "MINIO_ROOT_USER=<usuario>" ^
  -e "MINIO_ROOT_PASSWORD=<contrasena>" ^
  -v minio_data:/data ^
  minio/minio server /data --console-address ":9001"
```

Una vez iniciado:

* **API de MinIO:** `http://localhost:9000`
* **Consola de administración:** `http://localhost:9001`

Accede a la consola utilizando las credenciales definidas en:

```text
MINIO_ROOT_USER
MINIO_ROOT_PASSWORD
```

Estas credenciales deben coincidir con las configuradas en `backend/.env`.

---

## 🚀 Ejecución de HazardEx

Una vez instaladas las dependencias, descargados los modelos y puesto en marcha MinIO, pueden iniciarse el backend y el frontend.

### 1. Iniciar el backend

Abre una terminal:

```bash
cd backend
venv\Scripts\activate

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

El backend estará disponible en:

```text
http://localhost:8000
```

La documentación automática de la API puede consultarse en:

```text
http://localhost:8000/docs
```

---

### 2. Iniciar el frontend

Abre una segunda terminal:

```bash
cd frontend
npm run dev
```

---

## 🌐 Acceso a la aplicación

Una vez iniciados todos los servicios, abre en el navegador:

### 👉 http://localhost:5173

---

## 📦 Resumen de servicios

| Servicio           | Dirección                    |
| ------------------ | ---------------------------- |
| HazardEx Frontend  | `http://localhost:5173`      |
| HazardEx Backend   | `http://localhost:8000`      |
| Swagger / API Docs | `http://localhost:8000/docs` |
| MinIO API          | `http://localhost:9000`      |
| MinIO Console      | `http://localhost:9001`      |

---

## ✅ Orden recomendado de ejecución

Para iniciar HazardEx correctamente:

```text
1. Iniciar Docker / MinIO
        ↓
2. Comprobar que Ollama está disponible
        ↓
3. Iniciar el backend
        ↓
4. Iniciar el frontend
        ↓
5. Abrir http://localhost:5173
```

> [!TIP]
> Si alguno de los modelos de Ollama no está descargado, ejecuta los comandos `ollama pull` indicados anteriormente antes de iniciar el análisis.
